from orchify.event import BaseEvent, AgentEvent, RuntimeEvent, EVENT_TYPES
import inspect
import threading as td
import asyncio
from typing import Any
from typing import get_args


class Broker:
    def __init__(self):
        self.hooks: dict[str, list[callable]] = {event_type: [] for event_type in get_args(EVENT_TYPES)}
        self.hooks['*'] = []

        self.requests: dict[str, td.Event] = {}
        self.runs: dict[str, Any] = {}

        self._emit_lock = td.Lock()
    
    def _execute_hook(self, hook: callable, event: BaseEvent):
        try: hook(event)
        except Exception as e:
            print(f"Error executing hook '{hook.__name__}' for event '{event.event_type}': {e}")

    def emit(self, event: BaseEvent) -> None:
        self._emit_lock.acquire()
        for hook in self.hooks.get(event.event_type, []):
            self._execute_hook(hook, event)
        for hook in self.hooks['*']:
            self._execute_hook(hook, event)
        self._emit_lock.release()
    
    def register_hook(self, event_type: str, hook: callable) -> None:
        if event_type not in self.hooks.keys():
            raise ValueError(f"Invalid event type: '{event_type}'")
        if not callable(hook):
            raise ValueError(f'Hook must be callable, got {type(hook)}')
        signature = inspect.signature(hook)
        if len(signature.parameters) != 1 or list(signature.parameters.values())[0].annotation not in [BaseEvent, AgentEvent, RuntimeEvent, inspect._empty]:
            raise ValueError(f'Hook must accept a single argument of type BaseEvent, AgentEvent, or RuntimeEvent, got {signature}')
        self.hooks[event_type].append(hook)
    
    def hook(self, event_type: str = '*') -> callable:
        def decorator(func: callable) -> callable:
            self.register_hook(event_type, func)
            return func
        return decorator
    
    def start_req(self, code: str) -> td.Event|None:
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