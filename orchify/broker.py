from orchify.event import BaseEvent, EventFeedback, FEEDBACK_TYPES
from orchify.utils import get_logger
import re
import threading as td
import asyncio
import uuid
import time
from collections import deque
from dataclasses import dataclass
from fnmatch import translate
from functools import lru_cache
from queue import Queue
from typing import Any, Callable, Optional, Union, get_args


logger = get_logger(__name__)


def _read_poll_interval() -> float:
    import os
    raw = os.getenv('ORCHIFY_POLL_INTERVAL')
    if not raw:
        return 0.05
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.05
    return value if value > 0 else 0.05


POLL_INTERVAL = _read_poll_interval()

#: Characters that make an fnmatch pattern a glob instead of a literal type.
_GLOB_CHARS = frozenset('*?[')

#: Upper bound on the per-event-type candidate cache (keys are event types).
_MAX_CANDIDATE_CACHE = 256


@lru_cache(maxsize=512)
def _compile_glob(pattern: str):
    '''Compile an fnmatch pattern once, so ``emit()`` never re-translates it.

    Args:
        pattern (str): An fnmatch glob such as ``'tool:*'``.

    Returns:
        re.Pattern: The compiled pattern; call ``.match(event_type)`` on it.
    '''
    return re.compile(translate(pattern))


def _set_async_event(event, loop) -> None:
    '''Set an asyncio.Event that belongs to another thread's loop, if possible.

    A closed loop (interpreter shutdown, a backend that was already stopped) must
    not turn a bookkeeping call into an exception, so the failure is swallowed.

    Args:
        event (Optional[asyncio.Event]): The event to set; None is a no-op.
        loop (Optional[asyncio.AbstractEventLoop]): The loop that owns ``event``.
    '''
    if event is None or loop is None:
        return
    try:
        loop.call_soon_threadsafe(event.set)
    except RuntimeError:
        pass


def _prepare_types(event_type: tuple):
    '''Split bound event types into literals, compiled globs and a match-all flag.

    Most hooks bind literal types (``'run:finish'``), which ``emit()`` can look up
    in a dict instead of glob-matching every hook. Only patterns that actually
    contain fnmatch metacharacters need the compiled-regex path.

    Args:
        event_type (tuple): The patterns a binding listens for.

    Returns:
        tuple: ``(literals, globs, star)`` where ``literals`` is a de-duplicated
        tuple of exact type strings, ``globs`` a tuple of compiled patterns and
        ``star`` is True when the binding listens for ``'*'``.
    '''
    literals = []
    globs = []
    star = False
    for pattern in event_type:
        if pattern == '*':
            star = True
        elif _GLOB_CHARS.isdisjoint(pattern):
            if pattern not in literals:
                literals.append(pattern)
        else:
            globs.append(_compile_glob(pattern))
    return tuple(literals), tuple(globs), star


_TERMINAL_EVENT_TYPES = ('run:finish', 'agent:abort')

_MAX_REMEMBERED_RUNS = 4096

DEFAULT_RUN_OUTCOME_LIMIT = 256


def _read_run_outcome_limit() -> int:
    '''Read how many finished-run outcomes the broker remembers.

    Returns:
        int: ``ORCHIFY_RUN_OUTCOME_LIMIT`` when it is a non-negative integer,
            otherwise ``DEFAULT_RUN_OUTCOME_LIMIT``. Zero disables the registry.
    '''
    import os
    raw = os.getenv('ORCHIFY_RUN_OUTCOME_LIMIT')
    if raw is None or str(raw).strip() == '':
        return DEFAULT_RUN_OUTCOME_LIMIT
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_RUN_OUTCOME_LIMIT
    return value if value >= 0 else DEFAULT_RUN_OUTCOME_LIMIT


def make_turn_id(run_id: str, turn: int) -> str:
    '''Build the stable id of one LLM turn inside a run.

    A turn is a single request/response round trip: ``run_id`` identifies the
    execution and ``turn`` is its 1-based index. The resulting id is step-scoped
    and resolves back to the run through ``Broker.run_id_of_turn()``.

    Args:
        run_id (str): Run the turn belongs to.
        turn (int): 1-based index of the turn inside the run.

    Returns:
        str: The turn id.
    '''
    return f'{run_id}#{turn}'


class ApprovalTimeout(Exception):
    '''Raised when a pending approval is not resolved before its timeout.'''


class RunStopped(Exception):
    '''Raised by ``wait_approval()`` when the run is asked to stop while parked.

    The approval is resolved as a rejection first, so a stopped run can never
    leave a live entry behind in the pool.
    '''


@dataclass
class PendingApproval:
    '''One manual-confirmation request parked in the broker's approval pool.

    Created by ``Broker.request_approval()`` (or ``ask_question()``) and resolved
    through ``Broker.resolve_approval()`` (or the ``approve()``/``reject()``/
    ``answer_approval()`` sugar). A resolved entry stays in the pool so late
    readers can inspect the decision, and is dropped by ``release_approval()`` or
    ``finish_task()``.

    ``run_id`` is the execution that asked; ``turn_id`` is the single LLM turn
    inside it that the request belongs to.

    Attributes:
        approval_id (str): Generated id (``a`` plus ten hex characters).
        run_id (str): Run that parked on this request.
        agent_name (str): Agent that asked.
        agent_code (str): Code of that agent.
        turn_id (str): LLM turn the request belongs to.
        kind (str): ``'tool'`` for a tool gate, ``'question'`` for an open question.
        tool_id (str): Model-assigned tool call id, when the request is a tool.
        tool_name (str): Name of the tool being confirmed.
        args (dict): Arguments the tool would be called with.
        step (int): Agent step that parked.
        timeout (Optional[float]): Seconds the waiter may park; None waits forever.
        created_at (float): ``time.time()`` when the entry was created.
        resolved (bool): True once a decision or answer arrived.
        approved (Optional[bool]): The decision; None while unresolved.
        reason (str): Rejection reason or timeout explanation.
        timed_out (bool): True when nobody answered before ``timeout``.
        answer (str): Free-text answer of a ``kind='question'`` entry.
    '''

    approval_id: str
    run_id: str
    agent_name: str = ''
    agent_code: str = ''
    turn_id: str = ''
    kind: str = 'tool'
    tool_id: str = ''
    tool_name: str = ''
    args: Optional[dict] = None
    step: int = 1
    timeout: Optional[float] = None
    created_at: float = 0.0
    resolved: bool = False
    approved: Optional[bool] = None
    reason: str = ''
    timed_out: bool = False
    answer: str = ''

    def __post_init__(self):
        if self.args is None:
            self.args = {}
        if not self.created_at:
            self.created_at = time.time()

    @property
    def decision(self) -> Optional[bool]:
        '''True approved, False rejected, None still pending.'''
        return self.approved

    @property
    def task_name(self) -> str:
        '''Deprecated alias of run_id (kept for older call sites).'''
        return self.run_id

    def __repr__(self) -> str:
        state = 'pending' if not self.resolved else ('approved' if self.approved else 'rejected')
        return (f"<PendingApproval id={self.approval_id!r} kind={self.kind!r} "
                f"tool={self.tool_name!r} run={self.run_id!r} {state}>")


class HookBinding:
    '''One registered hook: a callable plus its match rules.

    Returned by ``Broker.register_hook()`` and ``Broker.hook()``; pass it to
    ``Broker.remove_binding()`` to dispose exactly this binding.

    Attributes:
        func (Callable): Called with each matching event.
        event_type (list[str]): Event globs the hook listens for.
        turn_id (Optional[str]): Legacy alias of ``run_id``.
        mode (str): ``'normal'``, ``'once'`` or ``'agent'``.
        count (int): How many events the hook has received.
        disposed (bool): True once the binding is retired.
        agent_name (Optional[str]): Only fire for this agent.
        match (Optional[Callable]): Extra predicate over the event.
        owner (str): Owner id (usually a plugin) used by ``remove_hooks()``.
        plugin (bool): True for a hook registered by a plugin, which makes it
            subject to the enabled-plugin filter.
        priority (int): Dispatch order; higher runs first, ties keep registration
            order.
        background (bool): True runs the hook later on the shared dispatcher, so
            a slow hook never delays the emitting run.
        seq (int): Registration sequence, used to merge the per-type index back
            into registration order.
        literals (tuple[str, ...]): Event types matched by exact string.
        globs (tuple[re.Pattern, ...]): Compiled patterns for glob event types.
        star (bool): True when the binding listens for ``'*'`` (everything).
    '''

    __slots__ = ('func', 'event_type', 'run_id', 'mode', 'count', 'disposed', 'agent_name', 'match',
                 'owner', 'plugin', 'priority', 'background',
                 'seq', 'literals', 'globs', 'star')

    def __init__(
        self,
        func: Callable,
        event_type: tuple,
        run_id: Optional[str],
        mode: str,
        agent_name: Optional[str] = None,
        match: Optional[Callable] = None,
        owner: str = '',
        plugin: bool = False,
        priority: int = 0,
        background: bool = False,
    ):
        self.func = func
        self.event_type = event_type
        self.run_id = run_id
        self.mode = mode
        self.count = 0
        self.disposed = False
        self.agent_name = agent_name
        self.match = match
        self.owner = owner
        self.plugin = bool(plugin)
        self.priority = int(priority or 0)
        self.background = bool(background)
        self.seq = 0
        self.literals, self.globs, self.star = _prepare_types(event_type)

    @property
    def turn_id(self) -> Optional[str]:
        '''Deprecated alias of run_id: hooks are bound to a run, not a turn.'''
        return self.run_id

    def __repr__(self):
        return (f"<HookBinding event_type={self.event_type!r} run_id={self.run_id!r} "
                f"mode={self.mode!r} agent_name={self.agent_name!r} count={self.count} "
                f"priority={self.priority} background={self.background} disposed={self.disposed}>")


class Broker:
    '''Shared registry for runs, hooks, feedback and approvals.

    A single ``orchify_broker`` instance is created at import time and used by
    every Agent, Group and plugin: agents register their run in it, hooks observe
    the events it carries, plugins push control feedback through it, and callers
    park on its approval pool. All public methods are thread-safe.

    Attributes:
        hooks (list[HookBinding]): Registered hook bindings.
        runs (dict[str, Any]): Active runs, keyed by run id.
        run_contexts (dict[str, dict]): Context dict of every active run.
    '''

    def __init__(self):
        self.hooks: list[HookBinding] = []
        self.runs: dict[str, Any] = {}
        self.run_contexts: dict[str, dict] = {}

        # Hook dispatch index (rebuilt from ``hooks`` whenever a binding is
        # disposed): exact-type buckets + one glob bucket, both kept in
        # registration order so ``emit()`` can merge them in O(matched).
        self._hook_seq = 0
        self._literal_index: dict[str, list[HookBinding]] = {}
        self._glob_bindings: list[HookBinding] = []
        self._agent_bindings: dict[str, list[HookBinding]] = {}
        self._priority_count = 0
        # Merged candidate list per event type; cleared with the index.
        self._candidates_cache: dict[str, tuple] = {}

        self._feedback: dict[str, deque] = {}
        self._feedback_lock = td.Lock()
        self._pause_events: dict[str, asyncio.Event] = {}
        self._pause_loops: dict[str, Any] = {}
        self._pause_lock = td.Lock()

        self._turn_index: dict[str, str] = {}
        self._turns_by_run: dict[str, list] = {}
        self._turn_lock = td.Lock()

        self._approvals: dict[str, PendingApproval] = {}
        self._approvals_by_run: dict[str, set] = {}
        self._approval_events: dict[str, asyncio.Event] = {}
        self._approval_loops: dict[str, Any] = {}
        self._approval_lock = td.Lock()

        self._emit_lock = td.RLock()

        self._run_done: dict[str, td.Event] = {}
        #: ``run_id -> [(asyncio.Event, loop), ...]``; a run may be awaited from
        #: more than one loop, so every waiter gets its own event and loop.
        self._run_done_async: dict[str, list] = {}
        self._run_done_lock = td.Lock()

        self._run_outcome_limit = _read_run_outcome_limit()
        self._run_outcomes: dict[str, dict] = {}
        self._run_outcomes_lock = td.Lock()

        self._hook_queue: Queue = Queue()
        self._hook_pending = 0
        self._hook_cond = td.Condition()
        self._hook_worker: Optional[td.Thread] = None
        self._hook_worker_lock = td.Lock()

    def record_run_outcome(
        self,
        run_id: str,
        event_type: str,
        payload: Optional[dict] = None,
    ) -> Optional[dict]:
        '''Remember the terminal event of a run so late callers can read it.

        ``emit()`` calls this for every event in ``_TERMINAL_EVENT_TYPES``, which
        removes the race that made ``Agent.ask()`` depend on catching a
        ``run:finish`` hook: a run that ends before the caller starts waiting
        still leaves its outcome behind. The registry keeps the most recent
        ``ORCHIFY_RUN_OUTCOME_LIMIT`` runs (default 256, 0 disables it).

        Args:
            run_id (str): Run the outcome belongs to.
            event_type (str): Terminal event type ('run:finish' or 'agent:abort').
            payload (Optional[dict]): Event payload, usually carrying 'answer',
                'stopped' or 'error'.

        Returns:
            Optional[dict]: The stored record, or None when the registry is
                disabled or no run id was given.
        '''
        if not run_id or not self._run_outcome_limit:
            return None
        record = {
            'run_id': run_id,
            'event_type': event_type,
            'payload': dict(payload or {}),
            'created_at': time.time(),
        }
        with self._run_outcomes_lock:
            existing = self._run_outcomes.pop(run_id, None)
            if existing is not None:
                record['created_at'] = existing.get('created_at', record['created_at'])
            self._run_outcomes[run_id] = record
            while len(self._run_outcomes) > self._run_outcome_limit:
                oldest = next(iter(self._run_outcomes))
                if oldest == run_id and len(self._run_outcomes) == 1:
                    break
                self._run_outcomes.pop(oldest, None)
        return record

    def run_outcome(self, run_id: str) -> Optional[dict]:
        '''Return the remembered terminal event of a run.

        Args:
            run_id (str): Run to look up.

        Returns:
            Optional[dict]: The record built by ``record_run_outcome()``, or None
                when the run is unknown, still running or already evicted.
        '''
        if not run_id:
            return None
        with self._run_outcomes_lock:
            record = self._run_outcomes.get(run_id)
            return dict(record) if record is not None else None

    def run_outcomes(self) -> list:
        '''List the remembered terminal outcomes, oldest first.

        Returns:
            list[dict]: One record per remembered run.
        '''
        with self._run_outcomes_lock:
            return [dict(record) for record in self._run_outcomes.values()]

    def clear_run_outcome(self, run_id: str) -> bool:
        '''Forget the terminal outcome of a run.

        Args:
            run_id (str): Run to forget.

        Returns:
            bool: True when a record was dropped.
        '''
        with self._run_outcomes_lock:
            return self._run_outcomes.pop(run_id, None) is not None

    @property
    def run_outcome_limit(self) -> int:
        '''Return how many finished-run outcomes the registry keeps.

        Returns:
            int: Configured bound (0 means the registry is disabled).
        '''
        return self._run_outcome_limit


    def _execute_hook(self, func: Callable, event: BaseEvent):
        try: func(event)
        except Exception as e:
            name = getattr(func, '__name__', None) or type(func).__name__
            logger.error("Error executing hook '%s' for event '%s': %s",
                         name, event.event_type, e)

    def _hook_worker_loop(self) -> None:
        '''Drain the background hook queue until the process exits.'''
        while True:
            item = self._hook_queue.get()
            try:
                binding, event = item
                if not binding.disposed:
                    self._execute_hook(binding.func, event)
            except Exception as exc:
                logger.error('Error in the background hook dispatcher: %s', exc)
            finally:
                with self._hook_cond:
                    self._hook_pending = max(0, self._hook_pending - 1)
                    self._hook_cond.notify_all()

    def _dispatch_hook(self, binding: HookBinding, event: BaseEvent) -> None:
        '''Hand a background hook to the dispatcher thread.

        Args:
            binding (HookBinding): Binding registered with ``background=True``.
            event (BaseEvent): The event to deliver later.
        '''
        with self._hook_worker_lock:
            worker = self._hook_worker
            if worker is None or not worker.is_alive():
                worker = td.Thread(target=self._hook_worker_loop,
                                   name='OrchifyHookDispatcher', daemon=True)
                self._hook_worker = worker
                worker.start()
        with self._hook_cond:
            self._hook_pending += 1
        self._hook_queue.put((binding, event))

    def drain_hooks(self, timeout: Optional[float] = None) -> bool:
        '''Wait until every queued background hook has run.

        Args:
            timeout (Optional[float]): Seconds to wait; None waits forever.

        Returns:
            bool: True when the queue drained, False on timeout.
        '''
        with self._hook_cond:
            return self._hook_cond.wait_for(lambda: self._hook_pending == 0, timeout)

    def pending_hooks(self) -> int:
        '''Number of background hooks queued but not executed yet.

        Returns:
            int: Pending background hook count.
        '''
        with self._hook_cond:
            return self._hook_pending

    def _index_binding(self, binding: HookBinding) -> None:
        '''Add one binding to the dispatch index; call it under ``_emit_lock``.

        Args:
            binding (HookBinding): The binding to index (its match rules are
                already prepared by ``HookBinding.__init__``).
        '''
        self._hook_seq += 1
        binding.seq = self._hook_seq
        for literal in binding.literals:
            self._literal_index.setdefault(literal, []).append(binding)
        if binding.star or binding.globs:
            self._glob_bindings.append(binding)
        if binding.mode == 'agent' and binding.run_id is not None:
            self._agent_bindings.setdefault(binding.run_id, []).append(binding)
        if binding.priority:
            self._priority_count += 1
        self._candidates_cache.clear()

    def _rebuild_hooks(self) -> None:
        '''Drop disposed bindings and rebuild the dispatch index.

        Called under ``_emit_lock`` only when a binding was actually disposed, so
        the per-event path never copies the hook list.
        '''
        self.hooks = [b for b in self.hooks if not b.disposed]
        self._literal_index = {}
        self._glob_bindings = []
        self._agent_bindings = {}
        self._priority_count = 0
        self._hook_seq = 0
        self._candidates_cache.clear()
        for binding in self.hooks:
            self._index_binding(binding)

    def _candidates_for(self, event_type: str):
        '''Return the bindings that can match ``event_type``, in registration order.

        Literal bindings come from the per-type bucket and glob bindings from the
        wildcard bucket; a binding listed for both appears once. This is what
        keeps ``emit()`` proportional to the hooks that can match instead of to
        every registered hook. The common case (no wildcard bindings at all)
        returns the literal bucket as is, without building anything.

        The merged list is cached per event type: a streaming run emits the same
        few types thousands of times, and the index only changes when a hook is
        registered or removed (both clear the cache). Disposed bindings stay in
        the cached list; ``_candidate_ok`` filters them out.

        Args:
            event_type (str): The emitted event type.

        Returns:
            tuple: Candidate bindings, still to be filtered by run/agent.
        '''
        cached = self._candidates_cache.get(event_type)
        if cached is not None:
            return cached
        literals = self._literal_index.get(event_type)
        globs = self._glob_bindings
        if not globs:
            result = tuple(literals or ())
        elif not literals:
            result = tuple(globs)
        else:
            merged = []
            i = 0
            j = 0
            n = len(literals)
            m = len(globs)
            while i < n and j < m:
                a = literals[i]
                b = globs[j]
                if a is b:
                    merged.append(a)
                    i += 1
                    j += 1
                elif a.seq < b.seq:
                    merged.append(a)
                    i += 1
                else:
                    merged.append(b)
                    j += 1
            if i < n:
                merged.extend(literals[i:])
            if j < m:
                merged.extend(globs[j:])
            result = tuple(merged)
        if len(self._candidates_cache) > _MAX_CANDIDATE_CACHE:
            self._candidates_cache.clear()
        self._candidates_cache[event_type] = result
        return result

    def _matches(self, binding: HookBinding, event: BaseEvent) -> bool:
        '''True when ``event`` matches ``binding`` (pure predicate, no bookkeeping).

        Args:
            binding (HookBinding): The binding to test.
            event (BaseEvent): The event to test against.

        Returns:
            bool: True when the hook should fire.
        '''
        return self._candidate_ok(binding, event, None, check_plugins=False)

    def _candidate_ok(self, binding: HookBinding, event: BaseEvent,
                      enabled_plugins, check_plugins: bool = True) -> bool:
        '''Full match test for one candidate binding.'''
        if binding.disposed:
            return False
        if binding.run_id is not None and event.run_id != binding.run_id:
            return False
        if binding.agent_name is not None and event.agent_name != binding.agent_name:
            return False
        if not binding.star and event.event_type not in binding.literals:
            if not any(rx.match(event.event_type) for rx in binding.globs):
                return False
        if binding.match is not None and not binding.match(event):
            return False
        if check_plugins and enabled_plugins is not None and binding.plugin \
                and binding.owner and binding.owner not in enabled_plugins:
            return False
        return True

    def emit(self, event: BaseEvent) -> None:
        '''Deliver an event to every matching hook.

        Hook selection and bookkeeping (``count``, ``once`` disposal, pruning of
        disposed and agent-scoped bindings) happen under ``_emit_lock``; the hook
        callables themselves run after the lock is released, so a slow hook can no
        longer stall hook registration or another run's bookkeeping. Normal hooks
        for one event run in priority order (higher first, ties keep registration
        order) on the emitting thread; hooks registered with ``background=True``
        are handed to the shared dispatcher instead, so they never delay the
        emitting run.

        A hook registered by a plugin (``plugin=True``) fires only while that
        plugin is enabled for the run; hooks registered by the application always
        fire, whatever their owner id.

        Args:
            event (BaseEvent): The event to deliver.

        Raises:
            TypeError: When ``event`` is not a ``BaseEvent``.
        '''
        if not isinstance(event, BaseEvent):
            raise TypeError(f'emit() requires a BaseEvent, got {type(event)}')

        if event.event_type in _TERMINAL_EVENT_TYPES:
            self.record_run_outcome(event.run_id, event.event_type, event.payload)

        with self._emit_lock:
            enabled_plugins = None
            if event.run_id and event.run_id in self.run_contexts:
                ctx = self.run_contexts[event.run_id]
                event.metadata.setdefault('run_context', ctx)
                enabled_plugins = ctx.get('enabled_plugins')

            event_type = event.event_type
            candidates = []
            for b in self._candidates_for(event_type):
                if self._candidate_ok(b, event, enabled_plugins):
                    candidates.append(b)

            once_matched = False
            for binding in candidates:
                binding.count += 1
                if binding.mode == 'once':
                    binding.disposed = True
                    once_matched = True

            is_finish = event_type == 'run:finish'
            drop_agents = is_finish and event.run_id in self._agent_bindings
            if drop_agents:
                for binding in self._agent_bindings.get(event.run_id, ()):
                    binding.disposed = True
            if once_matched or drop_agents:
                self._rebuild_hooks()

        if self._priority_count:
            if any(b.priority for b in candidates):
                candidates.sort(key=lambda b: -b.priority)
        for binding in candidates:
            if binding.background:
                self._dispatch_hook(binding, event)
            else:
                self._execute_hook(binding.func, event)


    def register_hook(
        self,
        event_type: Union[str, list, tuple, set],
        hook: Callable,
        run_id: Optional[str] = None,
        mode: str = 'normal',
        agent_name: Optional[str] = None,
        match: Optional[Callable] = None,
        owner: str = '',
        turn_id: Optional[str] = None,
        plugin: bool = False,
        priority: int = 0,
        background: bool = False,
    ) -> HookBinding:
        '''Bind a hook to events.

        Args:
            event_type (Union[str, List[str], Tuple[str, ...], Set[str]]): Event glob
                or globs to listen for.
            hook (Callable): Called with each matching event.
            run_id (Optional[str]): Restrict the hook to one execution; a hook bound
                to a run is disposed when that run finishes.
            mode (str): ``'normal'``, ``'once'`` or ``'agent'``.
            agent_name (Optional[str]): Only fire for events of this agent.
            match (Optional[Callable]): Extra predicate over the event.
            owner (str): Owner id, used by ``remove_hooks()``.
            turn_id (Optional[str]): Deprecated alias of ``run_id``.
            plugin (bool): True when a plugin registers this hook, which subjects
                it to the enabled-plugin filter in ``emit()``.
            priority (int): Higher priority hooks run first; equal priorities keep
                registration order.
            background (bool): Run the hook on the shared dispatcher instead of the
                emitting thread, so a slow hook cannot delay the run. Use
                ``drain_hooks()`` to wait for queued background hooks.

        Returns:
            HookBinding: The binding, which can be passed to ``remove_binding()``.
        '''
        if run_id is None:
            run_id = turn_id
        if mode == 'background':
            # ``mode='background'`` predates the ``background`` flag: it was
            # accepted here and silently ran the hook on the emitting thread,
            # which put telemetry and persistence work in the middle of a run.
            # Keep the spelling working, but route it through the dispatcher.
            mode = 'normal'
            background = True
        if not callable(hook):
            raise ValueError(f'Hook must be callable, got {type(hook)}')
        if mode not in ('normal', 'once', 'agent'):
            raise ValueError(f"mode must be 'normal', 'once', 'agent' or 'background', got {mode!r}")
        if mode == 'agent' and run_id is None:
            raise ValueError("Hook mode 'agent' requires a run_id so it can be destroyed on that run's finish.")
        if match is not None and not callable(match):
            raise ValueError(f'Match filter must be callable, got {type(match)}')
        types = tuple(event_type) if isinstance(event_type, (tuple, list, set)) else (event_type,)
        if not types or not all(isinstance(t, str) for t in types):
            raise ValueError(f'event_type must be a str or a list/tuple/set of strings, got {event_type!r}')
        binding = HookBinding(hook, types, run_id, mode, agent_name=agent_name, match=match,
                              owner=owner, plugin=plugin, priority=priority,
                              background=background)
        with self._emit_lock:
            self.hooks.append(binding)
            self._index_binding(binding)
        return binding

    def remove_hooks(self, owner: str) -> int:
        '''Dispose and remove every hook binding of one owner.

        Args:
            owner (str): Owner id, typically a plugin name.

        Returns:
            int: Number of bindings removed.
        '''
        with self._emit_lock:
            doomed = [b for b in self.hooks if b.owner == owner]
            if not doomed:
                return 0
            kept = [b for b in self.hooks if b.owner != owner]
            self.hooks = kept
            self._rebuild_hooks()
        return len(doomed)

    def remove_binding(self, binding: HookBinding) -> bool:
        '''Dispose and remove one hook binding.

        ``remove_hooks()`` is owner-based and would remove every binding of that
        owner; this targets the exact binding a caller holds, which is what
        ``OrchifyManager.close()`` uses to detach its global tap.

        Args:
            binding (HookBinding): The binding returned by ``register_hook()``.

        Returns:
            bool: True when the binding was present and removed.
        '''
        with self._emit_lock:
            if binding not in self.hooks:
                binding.disposed = True
                return False
            binding.disposed = True
            self._rebuild_hooks()
        return True

    def hook(
        self,
        event_type: Union[str, list, tuple, set] = '*',
        run_id: Optional[str] = None,
        mode: str = 'normal',
        once: bool = False,
        agent_name: Optional[str] = None,
        match: Optional[Callable] = None,
        owner: str = '',
        turn_id: Optional[str] = None,
        plugin: bool = False,
        priority: int = 0,
        background: bool = False,
    ) -> Callable:
        '''Decorator registering a hook.

        Args:
            event_type (Union[str, List[str], Tuple[str, ...], Set[str]]): Event glob
                or globs; the hook fires for any of them.
            run_id (Optional[str]): Restrict the hook to one execution.
            mode (str): ``'normal'``, ``'once'`` or ``'agent'``.
            once (bool): Shorthand for ``mode='once'``.
            agent_name (Optional[str]): Only fire for events of this agent.
            match (Optional[Callable]): Extra predicate over the event.
            owner (str): Owner id, used by ``remove_hooks()``.
            turn_id (Optional[str]): Deprecated alias of ``run_id``.
            plugin (bool): True when a plugin registers this hook.
            priority (int): Higher priority hooks run first.
            background (bool): Run the hook on the shared dispatcher.

        Returns:
            Callable: A decorator that registers the function and returns it.
        '''
        def decorator(func: Callable) -> Callable:
            '''Register ``func`` and return it unchanged.

            Args:
                func (Callable): The hook implementation.

            Returns:
                Callable: ``func``, unchanged.
            '''
            return self.register_hook(event_type, func, run_id=run_id, mode='once' if once else mode,
                                      agent_name=agent_name, match=match, owner=owner, turn_id=turn_id,
                                      plugin=plugin, priority=priority, background=background)
        return decorator


    def event(
        self,
        event_type: str,
        *,
        agent_name: str = '',
        agent_code: str = '',
        turn_id: str = '',
        turn: int = 1,
        payload: dict = None,
        run_id: str = '',
        source: str = 'plugin',
        metadata: dict = None,
    ) -> BaseEvent:
        '''Build an event on behalf of a plugin.

        Args:
            event_type (str): Event type to emit.
            agent_name (str): Name of the acting agent.
            agent_code (str): Code of the acting agent.
            turn_id (str): LLM turn the event belongs to.
            turn (int): 1-based step index.
            payload (Optional[dict]): Event payload.
            run_id (str): Run the event belongs to.
            source (str): Emitter label; defaults to ``'plugin'``.
            metadata (Optional[dict]): Extra metadata carried by the event.

        Returns:
            BaseEvent: The event, ready to pass to ``emit()``.
        '''
        return BaseEvent(
            agent_name=agent_name,
            agent_code=agent_code,
            turn_id=turn_id,
            turn=turn,
            event_type=event_type,
            payload=payload,
            run_id=run_id,
            source=source,
            metadata=metadata,
        )

    def get_run_context(self, run_id: str) -> Optional[dict]:
        '''Return the context dict of a run.

        Args:
            run_id (str): The run id.

        Returns:
            Optional[dict]: The context of that run, or None when it is unknown.
        '''
        return self.run_contexts.get(run_id)


    def register_turn(self, turn_id: str, run_id: str) -> None:
        '''Record which run one LLM turn belongs to.

        Args:
            turn_id (str): The turn id (see ``make_turn_id``).
            run_id (str): The run that turn belongs to.
        '''
        if not turn_id or not run_id:
            return
        with self._turn_lock:
            self._turn_index[turn_id] = run_id
            turns = self._turns_by_run.setdefault(run_id, [])
            if turn_id not in turns:
                turns.append(turn_id)

    def run_id_of_turn(self, turn_id: str) -> Optional[str]:
        '''Resolve a turn_id (or a run_id, for convenience) to its run_id.'''
        with self._turn_lock:
            run_id = self._turn_index.get(turn_id)
        if run_id is not None:
            return run_id
        if turn_id in self.run_contexts:
            return turn_id
        for name, ctx in self.run_contexts.items():
            if ctx.get('run_id') == turn_id or ctx.get('turn_id') == turn_id:
                return name
        return None

    def active_runs(self) -> list:
        '''run_ids of every execution currently registered.'''
        return list(self.run_contexts.keys())


    def feedback(
        self,
        ftype: str,
        *,
        run_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        task_name: Optional[str] = None,
        payload: dict = None,
    ) -> bool:
        '''Push feedback from a hook or plugin into a running task.

        Target the run with ``run_id`` (the unique handle of one execution, returned
        by ``Agent.run``/``Group.run`` and carried as ``event.run_id``). ``turn_id``
        addresses one LLM turn inside a run and resolves to its run; ``task_name`` is
        a deprecated alias of ``run_id``.

        Supported types:

        * ``'control:stop'``: abort the run at the next checkpoint.
        * ``'control:pause'``: suspend the run until ``'control:resume'``.
        * ``'control:resume'``: wake a paused run.
        * ``'control:retry'``: re-issue the current request after a failure with
          ``payload={'kwargs': {...}, 'max_retries': n}``.
        * ``'control:override_answer'``: answer directly with
          ``payload={'content': str}``, skipping the next LLM call.
        * ``'control:deny_tool'``: block a tool call and feed the reason back as its
          result, with ``payload={'tool_id'|'tool_name', 'reason'}``.
        * ``'control:inject_tool_result'``: replace a pending tool execution with
          ``payload={'tool_id'|'tool_name', 'result'}``.
        * ``'control:update_messages'``: mutate the conversation history at the next
          checkpoint with ``payload={'op': 'append'|'truncate'|'replace',
          'messages': [...], 'keep': n}``.
        * ``'control:switch_model'``: override the model for the next request with
          ``payload={'model': str}``.

        Args:
            ftype (str): One of the control types above.
            run_id (Optional[str]): Target run.
            turn_id (Optional[str]): Target turn, resolved to its run.
            task_name (Optional[str]): Deprecated alias of ``run_id``.
            payload (Optional[dict]): Type-specific data.

        Returns:
            bool: True once the message was queued, False when the target run is not
            active.
        '''
        if ftype not in get_args(FEEDBACK_TYPES):
            raise ValueError(f"Invalid feedback type: '{ftype}'")

        target = run_id or task_name
        if target is None and turn_id is not None:
            target = self.run_id_of_turn(turn_id)

        if target is None or target not in self.run_contexts:
            return False

        if ftype == 'control:resume':
            if self._resume_run(target):
                return True
            fb = EventFeedback(ftype='control:resume', payload=payload)
            with self._feedback_lock:
                self._feedback.setdefault(target, deque()).append(fb)
            self._wake_run_waiters(target)
            return True

        fb = EventFeedback(ftype=ftype, payload=payload)
        with self._feedback_lock:
            self._feedback.setdefault(target, deque()).append(fb)
        if ftype == 'control:stop':
            self._wake_run_waiters(target)
        return True

    def drain_feedback(self, run_id: str, types: Optional[set] = None) -> list:
        '''Pop queued feedback for a run.

        Args:
            run_id (str): The run id.
            types (Optional[set]): When given, only items whose ``ftype`` is in the
                set are consumed; the rest stay queued for later checkpoints.

        Returns:
            list: The consumed ``EventFeedback`` items.
        '''
        task_name = run_id
        with self._feedback_lock:
            q = self._feedback.get(task_name)
            if not q:
                return []
            if types is None:
                self._feedback.pop(task_name, None)
                return list(q)
            kept = deque()
            out = []
            while q:
                item = q.popleft()
                if item.ftype in types:
                    out.append(item)
                else:
                    kept.append(item)
            if kept:
                self._feedback[task_name] = kept
            else:
                self._feedback.pop(task_name, None)
            return out

    def pop_feedback(self, run_id: str, types: Optional[set] = None) -> Optional[EventFeedback]:
        '''Pop the first queued feedback item of the given types.

        Unlike ``drain_feedback()`` this consumes at most one item and leaves every
        other type queued, which is what a mid-stream checkpoint wants.

        Args:
            run_id (str): The run id.
            types (Optional[set]): Accepted ``ftype`` values; None accepts any.

        Returns:
            Optional[EventFeedback]: The popped item, or None when none matched.
        '''
        task_name = run_id
        with self._feedback_lock:
            q = self._feedback.get(task_name)
            if not q:
                return None
            for idx, item in enumerate(q):
                if types is None or item.ftype in types:
                    del q[idx]
                    if not q:
                        self._feedback.pop(task_name, None)
                    return item
            return None

    def arm_resume(self, run_id: str) -> bool:
        '''Pre-register the pause waiter for a run.

        Call this before emitting ``run:paused`` so that a ``control:resume`` sent in
        reaction to that event finds a waiter instead of racing past it (the original
        lost-resume bug).

        Args:
            run_id (str): The run about to be parked.

        Returns:
            bool: True when a waiter was armed; False when one was already armed.
        '''
        loop = asyncio.get_running_loop()
        with self._pause_lock:
            if run_id in self._pause_events:
                return False
            self._pause_events[run_id] = asyncio.Event()
            self._pause_loops[run_id] = loop
        return True

    async def wait_resume(self, run_id: str, timeout: Optional[float] = None) -> str:
        '''Park the current run until a ``control:resume`` arrives for it.

        Used by the agent to suspend between steps (human-in-the-loop). The waiter
        must already have been armed by ``arm_resume()``, so a resume arriving
        between the ``run:paused`` event and this call is not lost.

        Args:
            run_id (str): The parked run.
            timeout (Optional[float]): Seconds to park; None waits until a decision
                or a stop arrives.

        Returns:
            str: ``'resumed'`` when woken by ``control:resume``, ``'stopped'`` when a
            ``control:stop`` arrived while parked, ``'timeout'`` when ``timeout``
            elapsed without a decision. A timeout does not silently continue the run:
            the caller decides the run's final status, so a paused task can never
            hang forever.
        '''
        loop = asyncio.get_running_loop()
        with self._pause_lock:
            event = self._pause_events.get(run_id)
            if event is None:
                event = asyncio.Event()
                self._pause_events[run_id] = event
                self._pause_loops[run_id] = loop
        deadline = (time.monotonic() + float(timeout)) if timeout else None
        try:
            while True:
                if self.pop_feedback(run_id, {'control:resume'}) is not None:
                    return 'resumed'
                if self.pop_feedback(run_id, {'control:stop'}) is not None:
                    return 'stopped'
                if event.is_set():
                    return 'resumed'
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return 'timeout'
                try:
                    await asyncio.wait_for(event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return 'timeout'
        finally:
            with self._pause_lock:
                if self._pause_events.get(run_id) is event:
                    self._pause_events.pop(run_id, None)
                    self._pause_loops.pop(run_id, None)

    def _resume_run(self, run_id: str) -> bool:
        with self._pause_lock:
            event = self._pause_events.get(run_id)
            loop = self._pause_loops.get(run_id)
        if event is None or loop is None:
            return False
        loop.call_soon_threadsafe(event.set)
        return True

    def _wake_run_waiters(self, run_id: str) -> None:
        '''Wake every coroutine parked on a run: its pause wait and its approvals.

        Queued feedback (``control:stop`` above all) has to interrupt a parked
        waiter immediately; without this wake-up the waiter could only notice the
        feedback on its next poll. Callers must queue the feedback *before*
        calling this, since a waiter that wakes up re-reads the queue first.

        Args:
            run_id (str): The run whose waiters should be re-checked.
        '''
        with self._pause_lock:
            pause_event = self._pause_events.get(run_id)
            pause_loop = self._pause_loops.get(run_id)
        _set_async_event(pause_event, pause_loop)

        with self._approval_lock:
            waiters = [(self._approval_events.get(aid), self._approval_loops.get(aid))
                       for aid in tuple(self._approvals_by_run.get(run_id, ()))]
        for approval_event, approval_loop in waiters:
            _set_async_event(approval_event, approval_loop)


    def request_approval(
        self,
        run_id: str,
        *,
        kind: str = 'tool',
        tool_id: str = '',
        tool_name: str = '',
        args: Optional[dict] = None,
        approval_id: Optional[str] = None,
        timeout: Optional[float] = None,
        turn_id: str = '',
        agent_name: str = '',
        agent_code: str = '',
        step: int = 1,
    ) -> PendingApproval:
        '''Register a manual-confirmation request in the approval pool.

        This is the synchronous half of the gate: the agent creates the entry, emits
        ``run:approval:required`` so a UI or hook can render it, then parks. The
        decision comes back through ``resolve_approval()``, ``approve()`` or
        ``reject()``, which may be called from any thread.

        Args:
            run_id (str): Run that is parking.
            kind (str): ``'tool'`` or ``'question'``.
            tool_id (str): Model-assigned tool call id.
            tool_name (str): Name of the tool being confirmed.
            args (Optional[dict]): Arguments the tool would be called with.
            approval_id (Optional[str]): Explicit id; generated when omitted.
            timeout (Optional[float]): Seconds the waiter may park.
            turn_id (str): LLM turn the request belongs to.
            agent_name (str): Agent that asked.
            agent_code (str): Code of that agent.
            step (int): Agent step that parked.

        Returns:
            PendingApproval: The registered entry.
        '''
        approval = PendingApproval(
            approval_id=approval_id or f'a{uuid.uuid4().hex[:10]}',
            run_id=run_id,
            agent_name=agent_name,
            agent_code=agent_code,
            turn_id=turn_id,
            kind=kind,
            tool_id=tool_id,
            tool_name=tool_name,
            args=dict(args or {}),
            step=step,
            timeout=timeout,
        )
        with self._approval_lock:
            self._approvals[approval.approval_id] = approval
            self._approvals_by_run.setdefault(run_id, set()).add(approval.approval_id)
        return approval

    def get_approval(self, approval_id: str) -> Optional[PendingApproval]:
        '''Return an approval entry by id.

        Args:
            approval_id (str): The id returned by ``request_approval()``.

        Returns:
            Optional[PendingApproval]: The entry, or None when unknown or released.
        '''
        with self._approval_lock:
            return self._approvals.get(approval_id)

    def pending_approvals(self, run_id: Optional[str] = None, *,
                          task_name: Optional[str] = None) -> list:
        '''Unresolved approvals, optionally filtered to one run.'''
        run_id = run_id if run_id is not None else task_name
        with self._approval_lock:
            return [a for a in self._approvals.values()
                    if not a.resolved and (run_id is None or a.run_id == run_id)]

    def approvals(self, run_id: Optional[str] = None, include_resolved: bool = True, *,
                  task_name: Optional[str] = None) -> list:
        '''Every approval in the pool, optionally filtered to one run.'''
        run_id = run_id if run_id is not None else task_name
        with self._approval_lock:
            return [a for a in self._approvals.values()
                    if (run_id is None or a.run_id == run_id)
                    and (include_resolved or not a.resolved)]

    def resolve_approval(
        self,
        approval_id: str,
        approved: bool,
        *,
        reason: str = '',
        timed_out: bool = False,
        remove: bool = False,
    ) -> bool:
        '''Deliver a decision for a pending approval.

        Args:
            approval_id (str): The entry to resolve.
            approved (bool): True to allow the action, False to deny it.
            reason (str): Explanation fed back to the agent.
            timed_out (bool): Mark the decision as a timeout rather than a refusal.
            remove (bool): Drop the entry from the pool once resolved.

        Returns:
            bool: True when the decision was accepted; False when the id is unknown
            or was already decided, which lets a UI tell "nothing to do" from
            "decision accepted". Wakes the parked task, which then proceeds, or is
            denied when ``approved`` is False.
        '''
        with self._approval_lock:
            approval = self._approvals.get(approval_id)
            if approval is None or approval.resolved:
                return False
            approval.resolved = True
            approval.approved = bool(approved)
            approval.reason = reason or ''
            approval.timed_out = bool(timed_out)
            event = self._approval_events.pop(approval_id, None)
            loop = self._approval_loops.pop(approval_id, None)
            if remove:
                self._approvals.pop(approval_id, None)
                self._forget_approval_locked(approval)
        if event is not None and loop is not None:
            loop.call_soon_threadsafe(event.set)
        return True

    def _forget_approval_locked(self, approval: PendingApproval) -> None:
        '''Drop an approval id from its run's index; call under ``_approval_lock``.'''
        ids = self._approvals_by_run.get(approval.run_id)
        if ids is None:
            return
        ids.discard(approval.approval_id)
        if not ids:
            self._approvals_by_run.pop(approval.run_id, None)

    def approve(self, approval_id: str, *, reason: str = '', remove: bool = False) -> bool:
        '''Convenience wrapper: accept a pending approval.'''
        return self.resolve_approval(approval_id, True, reason=reason, remove=remove)

    def ask_question(
        self,
        run_id: str,
        question: str,
        *,
        tool_id: str = '',
        tool_name: str = 'ask_user',
        timeout: Optional[float] = None,
        turn_id: str = '',
        agent_name: str = '',
        agent_code: str = '',
        step: int = 1,
    ) -> PendingApproval:
        '''Register an open question (``kind='question'``) in the same pool.

        Modelled as a ``PendingApproval`` so that waiting, timeouts, stop-while-parked
        and the manager views are shared with tool approvals; the answer itself is
        free text and lands in ``PendingApproval.answer`` through
        ``answer_approval()``.

        Args:
            run_id (str): Run that is asking.
            question (str): The question put to the human operator.
            tool_id (str): Model-assigned tool call id.
            tool_name (str): Name of the question tool.
            timeout (Optional[float]): Seconds the waiter may park.
            turn_id (str): LLM turn the question belongs to.
            agent_name (str): Agent that asked.
            agent_code (str): Code of that agent.
            step (int): Agent step that parked.

        Returns:
            PendingApproval: The registered entry.
        '''
        return self.request_approval(
            run_id,
            kind='question',
            tool_id=tool_id,
            tool_name=tool_name,
            args={'question': str(question)},
            timeout=timeout,
            turn_id=turn_id,
            agent_name=agent_name,
            agent_code=agent_code,
            step=step,
        )

    def answer_approval(self, approval_id: str, answer: str, *, remove: bool = False) -> bool:
        '''Deliver a free-text answer to a pending question.

        Args:
            approval_id (str): The question entry to answer.
            answer (str): The free-text answer.
            remove (bool): Drop the entry from the pool once answered.

        Returns:
            bool: True when the answer was accepted; False when the id is unknown or
            the entry was already resolved, so a UI can tell "already answered" from
            "answer accepted".
        '''
        with self._approval_lock:
            approval = self._approvals.get(approval_id)
            if approval is None or approval.resolved:
                return False
            approval.resolved = True
            approval.approved = True
            approval.answer = '' if answer is None else str(answer)
            approval.reason = ''
            approval.timed_out = False
            event = self._approval_events.pop(approval_id, None)
            loop = self._approval_loops.pop(approval_id, None)
            if remove:
                self._approvals.pop(approval_id, None)
                self._forget_approval_locked(approval)
        if event is not None and loop is not None:
            loop.call_soon_threadsafe(event.set)
        return True

    def pending_questions(self, run_id: Optional[str] = None) -> list:
        '''Unresolved kind='question' entries, optionally filtered to one run.'''
        with self._approval_lock:
            return [a for a in self._approvals.values()
                    if not a.resolved and a.kind == 'question'
                    and (run_id is None or a.run_id == run_id)]

    def reject(self, approval_id: str, *, reason: str = 'denied', remove: bool = False) -> bool:
        '''Convenience wrapper: deny a pending approval.'''
        return self.resolve_approval(approval_id, False, reason=reason, remove=remove)

    def release_approval(self, approval_id: str) -> bool:
        '''Drop an approval from the pool (used to clean up after a task ends).'''
        with self._approval_lock:
            approval = self._approvals.pop(approval_id, None)
            if approval is None:
                return False
            self._forget_approval_locked(approval)
            return True

    async def wait_approval(
        self,
        approval_id: str,
        timeout: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> PendingApproval:
        '''Park the current run until the approval is resolved.

        Registers this run's approval event before returning to the caller's next
        await point, so a decision delivered immediately after the
        ``run:approval:required`` event is never lost.

        Args:
            approval_id (str): The entry to wait for.
            timeout (Optional[float]): Seconds to wait; None waits forever.
            run_id (Optional[str]): Run that is parking, so a ``control:stop`` can
                interrupt the wait.

        Returns:
            PendingApproval: The resolved entry.

        Raises:
            ApprovalTimeout: When ``timeout`` seconds elapse without a decision.
            RunStopped: When the run is asked to stop while parked.
            KeyError: When the entry was released out from under the caller.
        '''
        loop = asyncio.get_running_loop()
        with self._approval_lock:
            approval = self._approvals.get(approval_id)
            if approval is None:
                raise KeyError(f'Unknown approval id: {approval_id!r}')
            if approval.resolved:
                return approval
            event = asyncio.Event()
            self._approval_events[approval_id] = event
            self._approval_loops[approval_id] = loop
        run_id = run_id or approval.run_id
        deadline = (time.monotonic() + float(timeout)) if timeout else None
        try:
            while True:
                with self._approval_lock:
                    approval = self._approvals.get(approval_id)
                if approval is None:
                    raise KeyError(f'Approval {approval_id!r} was released while waiting.')
                if approval.resolved:
                    return approval
                if run_id and self.pop_feedback(run_id, {'control:stop'}) is not None:
                    self.resolve_approval(approval_id, False, reason='run stopped')
                    raise RunStopped(f'Run stopped while awaiting approval {approval_id!r}.')
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise ApprovalTimeout(
                        f'Approval {approval_id!r} ({approval.tool_name or approval.kind}) '
                        f'not resolved within {timeout}s.')
                try:
                    await asyncio.wait_for(event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise ApprovalTimeout(
                        f'Approval {approval_id!r} ({approval.tool_name or approval.kind}) '
                        f'not resolved within {timeout}s.')
        finally:
            with self._approval_lock:
                if self._approval_events.get(approval_id) is event:
                    self._approval_events.pop(approval_id, None)
                    self._approval_loops.pop(approval_id, None)


    def start_task(self, run_id: str, task: Any, context: Optional[dict] = None):
        '''Register a run before it starts executing.

        Args:
            run_id (str): The run id.
            task (Any): The asyncio task driving the run.
            context (Optional[dict]): Run context (agent name/code, started_at,
                enabled plugins and plugin config) exposed to hooks and plugins. The
                key ``enabled_plugins`` filters which plugin-owned hooks may fire.
        '''
        self.runs[run_id] = task
        if context is not None:
            self.run_contexts[run_id] = context
        with self._run_done_lock:
            existing = self._run_done.get(run_id)
            if existing is None or existing.is_set():
                self._run_done[run_id] = td.Event()

    def finish_task(self, run_id: str):
        '''Retire a run and release everything it parked on.

        Removes the run, its context and its turn index, drops its queued feedback,
        wakes any pause waiter, and releases every approval of the run so a parked
        approval can never leak past the end of the run.

        Args:
            run_id (str): The run id; unknown ids are ignored.
        '''
        if run_id in self.runs:
            del self.runs[run_id]
        if run_id in self.run_contexts:
            del self.run_contexts[run_id]
        with self._turn_lock:
            for tid in self._turns_by_run.pop(run_id, ()):
                if self._turn_index.get(tid) == run_id:
                    self._turn_index.pop(tid, None)
        with self._feedback_lock:
            self._feedback.pop(run_id, None)
        with self._pause_lock:
            event = self._pause_events.pop(run_id, None)
            loop = self._pause_loops.pop(run_id, None)
        _set_async_event(event, loop)
        with self._approval_lock:
            stale = list(self._approvals_by_run.pop(run_id, ()))
            waiters = []
            for aid in stale:
                self._approvals.pop(aid, None)
                waiters.append((self._approval_events.pop(aid, None),
                                self._approval_loops.pop(aid, None)))
        for a_event, a_loop in waiters:
            _set_async_event(a_event, a_loop)
        with self._run_done_lock:
            done = self._run_done.get(run_id)
            if done is not None:
                done.set()
            async_waiters = self._run_done_async.pop(run_id, ())
            if len(self._run_done) > _MAX_REMEMBERED_RUNS:
                for rid in [r for r, ev in self._run_done.items() if ev.is_set()]:
                    if len(self._run_done) <= _MAX_REMEMBERED_RUNS:
                        break
                    self._run_done.pop(rid, None)
                    self._run_done_async.pop(rid, None)
        for async_event, async_loop in async_waiters:
            _set_async_event(async_event, async_loop)

    def is_run_active(self, run_id: str) -> bool:
        '''True while `run_id` is registered (its run has not finished).'''
        return run_id in self.run_contexts

    def run_done_event(self, run_id: str) -> Optional[td.Event]:
        '''The completion event of `run_id`, or None if the run is unknown.'''
        with self._run_done_lock:
            return self._run_done.get(run_id)

    def wait_run(self, run_id: str, timeout: Optional[float] = None) -> bool:
        '''Block the calling thread until a run finishes.

        Must not be called from the orchestrator loop thread: that thread has to stay
        free to execute the run itself. Use ``Agent.wait()``, which checks this.

        Args:
            run_id (str): The run id.
            timeout (Optional[float]): Seconds to wait; None waits forever.

        Returns:
            bool: True when the run is finished (including one that had already
            finished), False for an unknown run id or on timeout.
        '''
        with self._run_done_lock:
            event = self._run_done.get(run_id)
        if event is None:
            return False
        return event.wait(timeout)

    async def await_run(self, run_id: str, timeout: Optional[float] = None) -> bool:
        '''Await a run without blocking the calling thread.

        The completion flag is a thread event, but ``finish_task()`` also sets a
        per-run ``asyncio.Event`` owned by the caller's loop, so awaiting a run
        costs nothing until it finishes. Use this from async code (``await`` a
        group's members) and ``wait_run`` from synchronous code.

        Args:
            run_id (str): The run id.
            timeout (Optional[float]): Seconds to wait; None waits forever.

        Returns:
            bool: True when the run is finished (including one that had already
            finished), False for an unknown run id or on timeout.
        '''
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._run_done_lock:
                thread_event = self._run_done.get(run_id)
                if thread_event is None:
                    return False
                if thread_event.is_set():
                    return True
                entry = (asyncio.Event(), loop)
                self._run_done_async.setdefault(run_id, []).append(entry)
            try:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                try:
                    await asyncio.wait_for(entry[0].wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return False
            finally:
                with self._run_done_lock:
                    waiters = self._run_done_async.get(run_id)
                    if waiters is not None:
                        try:
                            waiters.remove(entry)
                        except ValueError:
                            pass
                        if not waiters:
                            self._run_done_async.pop(run_id, None)
            if entry[0].is_set():
                return True


orchify_broker = Broker()
