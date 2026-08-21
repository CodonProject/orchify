from orchify.event import BaseEvent, AgentEvent, ToolEvent, RuntimeEvent, EventFeedback, FEEDBACK_TYPES, EVENT_TYPES
import inspect
import threading as td
import asyncio
from collections import deque
from fnmatch import fnmatchcase
from typing import Any, Callable, Optional, Literal, Union
from typing import get_args


BUILTIN_EVENT_TYPES = set(get_args(EVENT_TYPES)) | {'*'}

HOOK_MODES = Literal['normal', 'once', 'agent']


class HookBinding:
    __slots__ = ('func', 'event_type', 'turn_id', 'mode', 'count', 'disposed', 'agent_name', 'match', 'owner')

    def __init__(
        self,
        func: Callable,
        event_type: str,
        turn_id: Optional[str],
        mode: str,
        agent_name: Optional[str] = None,
        match: Optional[Callable] = None,
        owner: str = '',
    ):
        self.func = func
        self.event_type = event_type
        self.turn_id = turn_id
        self.mode = mode
        self.count = 0
        self.disposed = False
        self.agent_name = agent_name
        self.match = match
        self.owner = owner

    def __repr__(self):
        return (f"<HookBinding event_type={self.event_type!r} turn_id={self.turn_id!r} "
                f"mode={self.mode!r} agent_name={self.agent_name!r} count={self.count} disposed={self.disposed}>")


class Broker:
    def __init__(self):
        self.hooks: dict[str, list[HookBinding]] = {event_type: [] for event_type in get_args(EVENT_TYPES)}
        self.hooks['*'] = []

        self.requests: dict[str, td.Event] = {}
        self.runs: dict[str, Any] = {}
        self.run_contexts: dict[str, dict] = {}

        self._feedback: dict[str, deque] = {}
        self._feedback_lock = td.Lock()
        self._pause_events: dict[str, asyncio.Event] = {}
        self._pause_loops: dict[str, Any] = {}
        self._pause_lock = td.Lock()

        self._event_owners: dict[str, str] = {}

        self._emit_lock = td.RLock()

    # ---------- hook registration ----------

    def declare(self, event_type: str, owner: str = '') -> None:
        '''Dynamically register a custom event type (e.g. namespaced plugin types).'''
        if event_type not in self.hooks:
            self.hooks[event_type] = []
        self._event_owners.setdefault(event_type, owner)

    def _all_hooks(self):
        for bindings in self.hooks.values():
            yield from bindings

    def _execute_hook(self, func: Callable, event: BaseEvent):
        try: func(event)
        except Exception as e:
            print(f"Error executing hook '{func.__name__}' for event '{event.event_type}': {e}")

    def _matches(self, binding: HookBinding, event: BaseEvent) -> bool:
        if binding.turn_id is not None and event.turn_id != binding.turn_id:
            return False
        if binding.agent_name is not None and event.agent_name != binding.agent_name:
            return False
        types = binding.event_type if isinstance(binding.event_type, (tuple, list)) else (binding.event_type,)
        if not any(fnmatchcase(event.event_type, t) for t in types):
            return False
        if binding.match is not None and not binding.match(event):
            return False
        return True

    # ---------- emit ----------

    def emit(self, event: BaseEvent) -> None:
        if not isinstance(event, BaseEvent):
            raise TypeError(f'emit() requires a BaseEvent, got {type(event)}')

        with self._emit_lock:
            if event.event_type not in self.hooks:
                self.hooks[event.event_type] = []

            # expose the run context to hooks via event metadata
            if event.run_id and event.run_id in self.run_contexts:
                event.metadata.setdefault('run_context', self.run_contexts[event.run_id])

            # determine enabled plugins for this run (None = all enabled)
            enabled_plugins = None
            if event.run_id and event.run_id in self.run_contexts:
                enabled_plugins = self.run_contexts[event.run_id].get('enabled_plugins')

            candidates = []
            seen = set()
            for b in self._all_hooks():
                if not b.disposed and b not in seen and self._matches(b, event):
                    # filter by enabled plugins: skip hooks owned by disabled plugins
                    if enabled_plugins is not None and b.owner and b.owner not in enabled_plugins:
                        continue
                    seen.add(b)
                    candidates.append(b)

            for binding in candidates:
                binding.count += 1
                self._execute_hook(binding.func, event)
                if binding.mode == 'once':
                    binding.disposed = True

            if event.event_type == 'run:finish':
                for et in list(self.hooks):
                    self.hooks[et] = [
                        b for b in self.hooks[et]
                        if not b.disposed and not (b.mode == 'agent' and b.turn_id == event.turn_id)
                    ]
            else:
                for et in list(self.hooks):
                    self.hooks[et] = [b for b in self.hooks[et] if not b.disposed]

    # ---------- registration API ----------

    def register_hook(
        self,
        event_type: Union[str, list, tuple, set],
        hook: Callable,
        turn_id: Optional[str] = None,
        mode: str = 'normal',
        agent_name: Optional[str] = None,
        match: Optional[Callable] = None,
        owner: str = '',
    ) -> HookBinding:
        if not callable(hook):
            raise ValueError(f'Hook must be callable, got {type(hook)}')
        signature = inspect.signature(hook)
        if len(signature.parameters) != 1 or list(signature.parameters.values())[0].annotation not in [BaseEvent, AgentEvent, ToolEvent, RuntimeEvent, inspect._empty]:
            raise ValueError(f'Hook must accept a single argument of type BaseEvent, AgentEvent, ToolEvent, or RuntimeEvent, got {signature}')
        if mode == 'agent' and turn_id is None:
            raise ValueError("Hook mode 'agent' requires a turn_id so it can be destroyed on that run's finish.")
        if match is not None and not callable(match):
            raise ValueError(f'Match filter must be callable, got {type(match)}')
        types = tuple(event_type) if isinstance(event_type, (tuple, list, set)) else (event_type,)
        if not types or not all(isinstance(t, str) for t in types):
            raise ValueError(f'event_type must be a str or a list/tuple/set of strings, got {event_type!r}')
        with self._emit_lock:
            for et in types:
                self.declare(et, owner=owner)
            binding = HookBinding(hook, types[0] if len(types) == 1 else types, turn_id, mode,
                                  agent_name=agent_name, match=match, owner=owner)
            for et in types:
                self.hooks[et].append(binding)
        return binding

    def remove_hooks(self, owner: str) -> int:
        '''
        Dispose and remove every hook binding owned by `owner` (typically a plugin).
        Also cleans up non-builtin event types that were declared by the owner and
        now have no remaining bindings. Returns the number of removed bindings.
        '''
        removed = 0
        with self._emit_lock:
            for event_type in list(self.hooks):
                bindings = self.hooks[event_type]
                kept = [b for b in bindings if b.owner != owner]
                removed += len(bindings) - len(kept)
                self.hooks[event_type] = kept
            for event_type in list(self._event_owners):
                if self._event_owners.get(event_type) == owner \
                        and event_type not in BUILTIN_EVENT_TYPES \
                        and not self.hooks.get(event_type):
                    del self.hooks[event_type]
                    del self._event_owners[event_type]
        return removed

    def hook(
        self,
        event_type: Union[str, list, tuple, set] = '*',
        turn_id: Optional[str] = None,
        mode: str = 'normal',
        once: bool = False,
        agent_name: Optional[str] = None,
        match: Optional[Callable] = None,
        owner: str = '',
    ) -> Callable:
        '''Decorator registering a hook. `event_type` may be a single string or a
        list/tuple/set of strings; the hook fires for any of them.'''
        def decorator(func: Callable) -> Callable:
            return self.register_hook(event_type, func, turn_id=turn_id, mode='once' if once else mode,
                                      agent_name=agent_name, match=match, owner=owner)
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
        self.declare(event_type)
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

    def get_run_context(self, name: str) -> Optional[dict]:
        return self.run_contexts.get(name)

    def find_run_context(self, turn_id: str = None, agent_name: str = None) -> tuple[Optional[str], Optional[dict]]:
        '''Find an active run context by turn_id or agent_name.
        Returns (task_name, context) or (None, None).'''
        for name, ctx in self.run_contexts.items():
            if turn_id and ctx.get('turn_id') == turn_id:
                return name, ctx
            if agent_name and ctx.get('agent_name') == agent_name:
                return name, ctx
        return None, None

    # ---------- feedback (reverse control channel) ----------

    def feedback(
        self,
        ftype: str,
        *,
        task_name: Optional[str] = None,
        turn_id: Optional[str] = None,
        comment: Optional[str] = None,
        payload: dict = None,
    ) -> bool:
        '''
        Push feedback from a hook/plugin back into a running agent task.
        Target the run by its task_name or turn_id. Returns False if the target
        run is not active, True once queued.

        Supported types:
          - 'control:stop':           abort the run at the next checkpoint.
          - 'control:continue':       no-op (reserved for flow control).
          - 'control:pause':          suspend the run until 'control:resume'.
          - 'control:resume':         wake a paused run (by task_name/turn_id).
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

        if task_name is None and turn_id is not None:
            for name, ctx in self.run_contexts.items():
                if ctx.get('turn_id') == turn_id:
                    task_name = name
                    break

        if task_name is None or task_name not in self.run_contexts:
            return False

        if ftype == 'control:resume':
            return self._resume_run(task_name)

        fb = EventFeedback(ftype=ftype, comment=comment, payload=payload)
        with self._feedback_lock:
            self._feedback.setdefault(task_name, deque()).append(fb)
        return True

    def drain_feedback(self, task_name: str, types: Optional[set] = None) -> list:
        '''
        Pop queued feedback for a task. If `types` is given, only items with an
        ftype in that set are consumed; the rest stay queued for later checkpoints.
        '''
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

    async def wait_resume(self, task_name: str) -> None:
        '''
        Park the current task until a 'control:resume' feedback arrives for it.
        Used by the agent to suspend between steps (human-in-the-loop approval).
        '''
        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        with self._pause_lock:
            self._pause_events[task_name] = event
            self._pause_loops[task_name] = loop
        try:
            await event.wait()
        finally:
            with self._pause_lock:
                self._pause_events.pop(task_name, None)
                self._pause_loops.pop(task_name, None)

    def _resume_run(self, task_name: str) -> bool:
        with self._pause_lock:
            event = self._pause_events.get(task_name)
            loop = self._pause_loops.get(task_name)
        if event is None or loop is None:
            return False
        loop.call_soon_threadsafe(event.set)
        return True

    # ---------- request / run tracking ----------

    def start_req(self, code: str) -> td.Event | None:
        if code in self.requests.keys(): return
        event = td.Event()
        self.requests[code] = event
        return event

    def stop_req(self, code: str) -> bool:
        if not code in self.requests.keys(): return False
        event: td.Event = self.requests[code]
        event.set()
        return True

    def finish_req(self, code: str) -> bool:
        if not code in self.requests.keys(): return False
        self.requests.pop(code)
        return True

    def start_td(self, t: td.Thread):
        self.runs[t.name] = t
        if not t.is_alive(): t.start()

    def finish_td(self, t: td.Thread):
        if not t.name in self.runs.keys(): return
        del self.runs[t.name]

    def start_task(self, name: str, task: asyncio.Task, context: Optional[dict] = None):
        self.runs[name] = task
        if context is not None:
            self.run_contexts[name] = context

    def finish_task(self, name: str):
        if name in self.runs:
            del self.runs[name]
        if name in self.run_contexts:
            del self.run_contexts[name]
        with self._feedback_lock:
            self._feedback.pop(name, None)
        with self._pause_lock:
            event = self._pause_events.pop(name, None)
            self._pause_loops.pop(name, None)
        if event is not None:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(event.set)


orchify_broker = Broker()