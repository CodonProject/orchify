from orchify.event import BaseEvent, AgentEvent, RuntimeEvent, EVENT_TYPES
import inspect
import threading as td
import asyncio
from typing import Any, Callable, Optional, Literal
from typing import get_args


HOOK_MODES = Literal['normal', 'once', 'agent']


class HookBinding:
    __slots__ = ('func', 'event_type', 'turn_id', 'mode', 'count', 'disposed')

    def __init__(self, func: Callable, event_type: str, turn_id: Optional[str], mode: str):
        self.func = func
        self.event_type = event_type
        self.turn_id = turn_id
        self.mode = mode
        self.count = 0
        self.disposed = False

    def __repr__(self):
        return f"<HookBinding event_type={self.event_type!r} turn_id={self.turn_id!r} mode={self.mode!r} count={self.count} disposed={self.disposed}>"


class Broker:
    def __init__(self):
        self.hooks: dict[str, list[HookBinding]] = {event_type: [] for event_type in get_args(EVENT_TYPES)}
        self.hooks['*'] = []

        self.requests: dict[str, td.Event] = {}
        self.runs: dict[str, Any] = {}

        self._emit_lock = td.Lock()

    def _execute_hook(self, func: Callable, event: BaseEvent):
        try: func(event)
        except Exception as e:
            print(f"Error executing hook '{func.__name__}' for event '{event.event_type}': {e}")

    def _matches(self, binding: HookBinding, event: BaseEvent) -> bool:
        if binding.turn_id is not None and event.turn_id != binding.turn_id:
            return False
        return True

    def emit(self, event: BaseEvent) -> None:
        with self._emit_lock:
            for binding in list(self.hooks.get(event.event_type, [])):
                if binding.disposed: continue
                if not self._matches(binding, event): continue
                binding.count += 1
                self._execute_hook(binding.func, event)
                if binding.mode == 'once':
                    binding.disposed = True

            for binding in list(self.hooks['*']):
                if binding.disposed: continue
                if not self._matches(binding, event): continue
                binding.count += 1
                self._execute_hook(binding.func, event)
                if binding.mode == 'once':
                    binding.disposed = True

            if event.event_type == 'run:finish':
                for event_type in self.hooks:
                    self.hooks[event_type] = [
                        b for b in self.hooks[event_type]
                        if not b.disposed and not (b.mode == 'agent' and b.turn_id == event.turn_id)
                    ]
            else:
                for event_type in self.hooks:
                    self.hooks[event_type] = [b for b in self.hooks[event_type] if not b.disposed]

    def register_hook(
        self,
        event_type: str,
        hook: Callable,
        turn_id: Optional[str] = None,
        mode: str = 'normal',
    ) -> None:
        if event_type not in self.hooks.keys():
            raise ValueError(f"Invalid event type: '{event_type}'")
        if not callable(hook):
            raise ValueError(f'Hook must be callable, got {type(hook)}')
        signature = inspect.signature(hook)
        if len(signature.parameters) != 1 or list(signature.parameters.values())[0].annotation not in [BaseEvent, AgentEvent, RuntimeEvent, inspect._empty]:
            raise ValueError(f'Hook must accept a single argument of type BaseEvent, AgentEvent, or RuntimeEvent, got {signature}')
        if mode == 'agent' and turn_id is None:
            raise ValueError("Hook mode 'agent' requires a turn_id so it can be destroyed on that run's finish.")
        binding = HookBinding(hook, event_type, turn_id, mode)
        self.hooks[event_type].append(binding)

    def hook(self, event_type: str = '*', turn_id: Optional[str] = None, mode: str = 'normal', once: bool = False) -> Callable:
        def decorator(func: Callable) -> Callable:
            self.register_hook(event_type, func, turn_id=turn_id, mode='once' if once else mode)
            return func
        return decorator

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

    def start_task(self, name: str, task: asyncio.Task):
        self.runs[name] = task

    def finish_task(self, name: str):
        if name in self.runs:
            del self.runs[name]


orchify_broker = Broker()