from orchify.event import BaseEvent, EventFeedback, FEEDBACK_TYPES
import threading as td
import asyncio
import uuid
import time
from collections import deque
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Callable, Optional, Union, get_args


# Default polling interval (seconds) used while a run is parked on a pause or an
# approval gate. The waiter must wake up periodically to observe control:stop,
# so it can never be a bare `await event.wait()`. Override with ORCHIFY_POLL_INTERVAL.
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

# How many finished runs the broker remembers so that a caller may still wait on
# (or inspect) a run that already ended. Bounded to keep long sessions flat.
_MAX_REMEMBERED_RUNS = 4096


def make_turn_id(run_id: str, turn: int) -> str:
    '''
    Build the stable id of one LLM turn inside a run.

    A turn is a single request/response round trip: `run_id` identifies the
    execution, `turn` is its 1-based index. The resulting id is step-scoped and
    resolves back to the run through Broker.run_id_of_turn().
    '''
    return f'{run_id}#{turn}'


class ApprovalTimeout(Exception):
    '''Raised when a pending approval is not resolved before its timeout.'''


class RunStopped(Exception):
    '''
    Raised by wait_approval() when the run is asked to stop while parked.

    The approval is resolved as a rejection first, so a stopped run can never
    leave a live entry behind in the pool.
    '''


@dataclass
class PendingApproval:
    '''
    One manual-confirmation request parked in the broker's approval pool.

    Created by Broker.request_approval(); resolved through
    Broker.resolve_approval() (or the approve()/reject() sugar). A resolved
    entry stays in the pool so late readers can inspect the decision, and is
    dropped by release_approval() / finish_task().

    `run_id` is the execution that asked (see the module docstring of
    orchify.agent for the run/turn model); `turn_id` is the single LLM turn
    inside it that the request belongs to.
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
    __slots__ = ('func', 'event_type', 'run_id', 'mode', 'count', 'disposed', 'agent_name', 'match', 'owner')

    def __init__(
        self,
        func: Callable,
        event_type: tuple,
        run_id: Optional[str],
        mode: str,
        agent_name: Optional[str] = None,
        match: Optional[Callable] = None,
        owner: str = '',
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

    @property
    def turn_id(self) -> Optional[str]:
        '''Deprecated alias of run_id: hooks are bound to a run, not a turn.'''
        return self.run_id

    def __repr__(self):
        return (f"<HookBinding event_type={self.event_type!r} run_id={self.run_id!r} "
                f"mode={self.mode!r} agent_name={self.agent_name!r} count={self.count} disposed={self.disposed}>")


class Broker:
    def __init__(self):
        self.hooks: list[HookBinding] = []
        self.runs: dict[str, Any] = {}
        self.run_contexts: dict[str, dict] = {}

        self._feedback: dict[str, deque] = {}
        self._feedback_lock = td.Lock()
        self._pause_events: dict[str, asyncio.Event] = {}
        self._pause_loops: dict[str, Any] = {}
        self._pause_lock = td.Lock()

        # turn_id -> run_id, so a caller holding a single turn can still address
        # the run it belongs to. Populated by register_turn().
        self._turn_index: dict[str, str] = {}
        self._turn_lock = td.Lock()

        self._approvals: dict[str, PendingApproval] = {}
        self._approval_events: dict[str, asyncio.Event] = {}
        self._approval_loops: dict[str, Any] = {}
        self._approval_lock = td.Lock()

        self._emit_lock = td.RLock()

        # run_id -> threading.Event, set by finish_task(). This lets a plain
        # caller thread block on a run (`wait_run`) instead of busy-waiting on
        # `runs`. Finished entries are kept (bounded) so waiting on a run that
        # already ended still returns True.
        self._run_done: dict[str, td.Event] = {}
        self._run_done_lock = td.Lock()

    # ---------- emit ----------

    def _execute_hook(self, func: Callable, event: BaseEvent):
        try: func(event)
        except Exception as e:
            name = getattr(func, '__name__', None) or type(func).__name__
            print(f"Error executing hook '{name}' for event '{event.event_type}': {e}")

    def _matches(self, binding: HookBinding, event: BaseEvent) -> bool:
        if binding.run_id is not None and event.run_id != binding.run_id:
            return False
        if binding.agent_name is not None and event.agent_name != binding.agent_name:
            return False
        if not any(fnmatchcase(event.event_type, t) for t in binding.event_type):
            return False
        if binding.match is not None and not binding.match(event):
            return False
        return True

    def emit(self, event: BaseEvent) -> None:
        '''
        Deliver `event` to every matching hook.

        Hook *selection* and bookkeeping (`count`, `once` disposal, pruning of
        disposed/agent bindings) happen under `_emit_lock`; the hook callables
        themselves run AFTER the lock is released, so a slow hook can no longer
        stall hook registration or another run's emit() bookkeeping. Hooks for
        one event still run in registration order, on the emitting thread.
        '''
        if not isinstance(event, BaseEvent):
            raise TypeError(f'emit() requires a BaseEvent, got {type(event)}')

        with self._emit_lock:
            # expose the run context to hooks via event metadata
            enabled_plugins = None
            if event.run_id and event.run_id in self.run_contexts:
                ctx = self.run_contexts[event.run_id]
                event.metadata.setdefault('run_context', ctx)
                enabled_plugins = ctx.get('enabled_plugins')

            candidates = []
            for b in self.hooks:
                if b.disposed or not self._matches(b, event):
                    continue
                # filter by enabled plugins: skip hooks owned by disabled plugins
                if enabled_plugins is not None and b.owner and b.owner not in enabled_plugins:
                    continue
                candidates.append(b)

            for binding in candidates:
                binding.count += 1
                if binding.mode == 'once':
                    binding.disposed = True

            is_finish = event.event_type == 'run:finish'
            self.hooks = [
                b for b in self.hooks
                if not b.disposed and not (is_finish and b.mode == 'agent' and b.run_id == event.run_id)
            ]

        # Run the hooks outside the lock (candidates were snapshotted inside it).
        for binding in candidates:
            self._execute_hook(binding.func, event)

    # ---------- registration API ----------

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
    ) -> HookBinding:
        '''
        Bind a hook to events. `run_id` restricts it to one execution; hooks are
        run-scoped (mode 'agent' hooks are disposed when that run finishes).
        `turn_id` is accepted as a deprecated alias of `run_id` - it used to name
        the run, and still resolves the same way.
        '''
        if run_id is None:
            run_id = turn_id
        if not callable(hook):
            raise ValueError(f'Hook must be callable, got {type(hook)}')
        if mode == 'agent' and run_id is None:
            raise ValueError("Hook mode 'agent' requires a run_id so it can be destroyed on that run's finish.")
        if match is not None and not callable(match):
            raise ValueError(f'Match filter must be callable, got {type(match)}')
        types = tuple(event_type) if isinstance(event_type, (tuple, list, set)) else (event_type,)
        if not types or not all(isinstance(t, str) for t in types):
            raise ValueError(f'event_type must be a str or a list/tuple/set of strings, got {event_type!r}')
        binding = HookBinding(hook, types, run_id, mode, agent_name=agent_name, match=match, owner=owner)
        with self._emit_lock:
            self.hooks.append(binding)
        return binding

    def remove_hooks(self, owner: str) -> int:
        '''Dispose and remove every hook binding owned by `owner` (typically a plugin).
        Returns the number of removed bindings.'''
        with self._emit_lock:
            kept = [b for b in self.hooks if b.owner != owner]
            removed = len(self.hooks) - len(kept)
            self.hooks = kept
        return removed

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
    ) -> Callable:
        '''Decorator registering a hook. `event_type` may be a single string or a
        list/tuple/set of strings; the hook fires for any of them. `turn_id` is a
        deprecated alias of `run_id`.'''
        def decorator(func: Callable) -> Callable:
            return self.register_hook(event_type, func, run_id=run_id, mode='once' if once else mode,
                                      agent_name=agent_name, match=match, owner=owner, turn_id=turn_id)
        return decorator

    # ---------- plugin event factory ----------

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
        return self.run_contexts.get(run_id)

    # ---------- run / turn identity ----------

    def register_turn(self, turn_id: str, run_id: str) -> None:
        '''
        Record that `turn_id` (one LLM turn) belongs to `run_id` (one
        execution), so feedback addressed at a single turn can still find the
        run to deliver it to.
        '''
        if not turn_id or not run_id:
            return
        with self._turn_lock:
            self._turn_index[turn_id] = run_id

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

    # ---------- feedback (reverse control channel) ----------

    def feedback(
        self,
        ftype: str,
        *,
        run_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        task_name: Optional[str] = None,
        payload: dict = None,
    ) -> bool:
        '''
        Push feedback from a hook/plugin back into a running agent task.

        Target the run with `run_id` - the unique handle of one execution
        (returned by Agent.run / Group.run, and carried as `event.run_id`).
        `turn_id` addresses one LLM turn inside a run and resolves to its run;
        `task_name` is a deprecated alias of `run_id`. Returns False if the
        target run is not active, True once queued.

        Supported types:
          - 'control:stop':           abort the run at the next checkpoint.
          - 'control:pause':          suspend the run until 'control:resume'.
          - 'control:resume':         wake a paused run.
          - 'control:retry':          payload={'kwargs': {...}, 'max_retries': n}
                                      re-issue the current request on failure with
                                      the given request overrides.
          - 'control:override_answer': payload={'content': str} answers directly,
                                      skipping the LLM call for the next request.
          - 'control:deny_tool':      payload={'tool_id'|'tool_name', 'reason'} blocks
                                      the tool call and feeds the reason back as result.
          - 'control:inject_tool_result': payload={'tool_id'|'tool_name', 'result'}
                                      replaces a pending tool execution with the
                                      given result.
          - 'control:update_messages': payload={'op': 'append'|'truncate'|'replace',
                                      'messages': [...], 'keep': n} mutates the
                                      conversation history at the next checkpoint.
          - 'control:switch_model':   payload={'model': str} overrides the model
                                      for the next request.
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
            # No waiter armed yet: the resume raced ahead of the park. Queue it as
            # feedback so wait_resume() observes it the moment it parks.
            fb = EventFeedback(ftype='control:resume', payload=payload)
            with self._feedback_lock:
                self._feedback.setdefault(target, deque()).append(fb)
            return True

        fb = EventFeedback(ftype=ftype, payload=payload)
        with self._feedback_lock:
            self._feedback.setdefault(target, deque()).append(fb)
        return True

    def drain_feedback(self, run_id: str, types: Optional[set] = None) -> list:
        '''
        Pop queued feedback for a run. If `types` is given, only items with an
        ftype in that set are consumed; the rest stay queued for later checkpoints.
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
        '''Pop only the first queued feedback item whose ftype is in `types`.

        Unlike drain_feedback() this consumes at most one item and leaves every
        other type queued, which is what a mid-stream checkpoint wants.
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
        '''
        Pre-register the pause waiter for a run.

        Call this BEFORE emitting 'run:paused' so that a 'control:resume' sent in
        reaction to that event finds a waiter instead of racing past it (the
        original lost-resume bug). Returns False when the run is already armed.
        '''
        loop = asyncio.get_running_loop()
        with self._pause_lock:
            if run_id in self._pause_events:
                return False
            self._pause_events[run_id] = asyncio.Event()
            self._pause_loops[run_id] = loop
        return True

    async def wait_resume(self, run_id: str, timeout: Optional[float] = None) -> str:
        '''
        Park the current run until a 'control:resume' feedback arrives for it.

        Used by the agent to suspend between steps (human-in-the-loop approval).
        The waiter must already have been armed by arm_resume() so that a resume
        arriving between the 'run:paused' event and this call is not lost.

        Returns:
            'resumed'  - woken by control:resume.
            'stopped'  - a control:stop arrived while parked.
            'timeout'  - `timeout` seconds elapsed without a decision.

        A timeout does NOT silently continue the run: the caller decides what
        the run's final status is, so a paused task can never hang forever.
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
                # a resume that raced ahead of the park is queued as feedback
                if self.pop_feedback(run_id, {'control:resume'}) is not None:
                    return 'resumed'
                if self.pop_feedback(run_id, {'control:stop'}) is not None:
                    return 'stopped'
                if event.is_set():
                    return 'resumed'
                if deadline is not None and time.monotonic() >= deadline:
                    return 'timeout'
                await asyncio.sleep(POLL_INTERVAL)
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

    # ---------- approval pool (manual confirmation) ----------

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
        '''
        Register a manual-confirmation request in the approval pool.

        This is the synchronous half of the gate: the agent creates the entry,
        emits 'run:approval:required' so a UI/hook can render it, then parks.
        The decision comes back through resolve_approval() / approve() /
        reject(), which may be called from any thread.
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
        return approval

    def get_approval(self, approval_id: str) -> Optional[PendingApproval]:
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
        '''
        Deliver a decision for a pending approval.

        Returns False when the id is unknown or was already decided, so a UI can
        distinguish "nothing to do" from "decision accepted". Wakes the parked
        task, which then proceeds (approved=False denies the action instead).
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
        if event is not None and loop is not None:
            loop.call_soon_threadsafe(event.set)
        return True

    def approve(self, approval_id: str, *, reason: str = '', remove: bool = False) -> bool:
        '''Convenience wrapper: accept a pending approval.'''
        return self.resolve_approval(approval_id, True, reason=reason, remove=remove)

    def reject(self, approval_id: str, *, reason: str = 'denied', remove: bool = False) -> bool:
        '''Convenience wrapper: deny a pending approval.'''
        return self.resolve_approval(approval_id, False, reason=reason, remove=remove)

    def release_approval(self, approval_id: str) -> bool:
        '''Drop an approval from the pool (used to clean up after a task ends).'''
        with self._approval_lock:
            return self._approvals.pop(approval_id, None) is not None

    async def wait_approval(
        self,
        approval_id: str,
        timeout: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> PendingApproval:
        '''
        Park the current run until the approval is resolved.

        Registers this run's approval event BEFORE returning to the caller's
        next await point, so a decision delivered immediately after the
        'run:approval:required' event is never lost. Raises ApprovalTimeout
        once `timeout` seconds elapse without a decision, and re-raises
        KeyError if the entry was released out from under us.
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
                    # A stop while parked resolves the request as a rejection and
                    # surfaces the stop so the caller can end the run instead of
                    # silently swallowing it.
                    self.resolve_approval(approval_id, False, reason='run stopped')
                    raise RunStopped(f'Run stopped while awaiting approval {approval_id!r}.')
                if deadline is not None and time.monotonic() >= deadline:
                    raise ApprovalTimeout(
                        f'Approval {approval_id!r} ({approval.tool_name or approval.kind}) '
                        f'not resolved within {timeout}s.')
                await asyncio.sleep(POLL_INTERVAL)
        finally:
            with self._approval_lock:
                if self._approval_events.get(approval_id) is event:
                    self._approval_events.pop(approval_id, None)
                    self._approval_loops.pop(approval_id, None)

    # ---------- run tracking ----------

    def start_task(self, run_id: str, task: Any, context: Optional[dict] = None):
        self.runs[run_id] = task
        if context is not None:
            self.run_contexts[run_id] = context
        with self._run_done_lock:
            # The same run is registered twice (Agent.run before dispatching, and
            # _async_thread_process once the coroutine starts). A waiter may
            # already hold the current event, so keep it unless the previous run
            # under this id had already finished.
            existing = self._run_done.get(run_id)
            if existing is None or existing.is_set():
                self._run_done[run_id] = td.Event()

    def finish_task(self, run_id: str):
        if run_id in self.runs:
            del self.runs[run_id]
        if run_id in self.run_contexts:
            del self.run_contexts[run_id]
        with self._turn_lock:
            for tid in [t for t, r in self._turn_index.items() if r == run_id]:
                self._turn_index.pop(tid, None)
        with self._feedback_lock:
            self._feedback.pop(run_id, None)
        with self._pause_lock:
            event = self._pause_events.pop(run_id, None)
            loop = self._pause_loops.pop(run_id, None)
        if event is not None and loop is not None:
            loop.call_soon_threadsafe(event.set)
        with self._approval_lock:
            stale = [aid for aid, a in self._approvals.items() if a.run_id == run_id]
            for aid in stale:
                self._approvals.pop(aid, None)
            waiters = [(aid, self._approval_events.pop(aid, None),
                        self._approval_loops.pop(aid, None)) for aid in stale]
        for _aid, a_event, a_loop in waiters:
            if a_event is not None and a_loop is not None:
                a_loop.call_soon_threadsafe(a_event.set)
        with self._run_done_lock:
            done = self._run_done.get(run_id)
            if done is not None:
                done.set()
            # keep the finished handles bounded, oldest already-set first
            if len(self._run_done) > _MAX_REMEMBERED_RUNS:
                for rid in [r for r, ev in self._run_done.items() if ev.is_set()]:
                    if len(self._run_done) <= _MAX_REMEMBERED_RUNS:
                        break
                    self._run_done.pop(rid, None)

    def is_run_active(self, run_id: str) -> bool:
        '''True while `run_id` is registered (its run has not finished).'''
        return run_id in self.run_contexts

    def run_done_event(self, run_id: str) -> Optional[td.Event]:
        '''The completion event of `run_id`, or None if the run is unknown.'''
        with self._run_done_lock:
            return self._run_done.get(run_id)

    def wait_run(self, run_id: str, timeout: Optional[float] = None) -> bool:
        '''
        Block the CALLING thread until `run_id` finishes.

        Returns True when the run is finished (including one that had already
        finished), False for an unknown run or on timeout. Must not be called
        from the orchestrator loop thread (that thread has to stay free to
        execute the run itself) - use `Agent.wait()` which checks this.
        '''
        with self._run_done_lock:
            event = self._run_done.get(run_id)
        if event is None:
            return False
        return event.wait(timeout)



orchify_broker = Broker()
