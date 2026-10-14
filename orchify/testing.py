from typing import Any, Callable, Dict, List, Optional

import asyncio
import threading
import time

from .broker import orchify_broker
from .llm.base import LLMInterface, Chunk, FinalStatus, Response


class FakeLLM(LLMInterface):
    '''Scripted, offline LLM for tests.

    Each plan is a list of steps; a step with ``content`` yields one stream
    chunk, a step with ``final`` yields the final status, and either may carry a
    ``delay`` in seconds. The last plan repeats once the script runs out, so a
    single-plan script answers every request.

    Attributes:
        script (List[List[dict]]): The plans, one per request.
        calls (List[dict]): Recorded requests, each with ``messages`` and
            ``model``.
    '''

    def __init__(self, script: Optional[List[List[dict]]] = None):
        '''Create the fake.

        Args:
            script (Optional[List[List[dict]]]): Plans to replay; an empty script
                answers every request with an empty final status.
        '''
        super().__init__()
        self.script: List[List[dict]] = list(script or [[]])
        self.calls: List[dict] = []

    async def _core_request(self, messages=None, model=None, tools=None, **kwargs):
        '''Yield the next scripted chunk, then its final status.

        Args:
            messages (Optional[List[dict]]): Conversation sent by the agent.
            model (Optional[str]): Model name the agent asked for.
            tools (Optional[List[dict]]): Tool schemas the agent advertised.
            **kwargs (Any): Extra request parameters, ignored.

        Yields:
            Response: Chunk responses followed by the final one.
        '''
        self.calls.append({'messages': [dict(m) for m in (messages or [])], 'model': model})
        plan = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        for entry in plan:
            if entry.get('delay'):
                await asyncio.sleep(entry['delay'])
            if entry.get('content') is not None:
                yield Response(current_chunk=Chunk(is_cot=False, content=entry['content']))
            if entry.get('final') is not None:
                yield Response(
                    current_chunk=None,
                    final_status=FinalStatus(**entry['final']),
                    is_final=True,
                )


def final(content: str, tool_calls: Optional[List[dict]] = None) -> dict:
    '''Build one final step for a :class:`FakeLLM` plan.

    Args:
        content (str): Answer text the fake reports.
        tool_calls (Optional[List[dict]]): Tool calls to request, usually built
            with :func:`tool_call`.

    Returns:
        dict: A plan step that yields a final status.
    '''
    return {'final': {'content': content, 'reasoning': '', 'tool_calls': tool_calls or []}}


def tool_call(name: str, args: str, call_id: str = 'c1') -> dict:
    '''Build one tool call for :func:`final`.

    Args:
        name (str): Tool name the model asks to run.
        args (str): JSON string of the tool arguments.
        call_id (str): Identifier echoed back in the tool result.

    Returns:
        dict: A tool call in the provider's wire format.
    '''
    return {'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': args}}


class Recorder:
    '''Collect broker events in tests, with optional per-type callbacks.

    The hook is registered owner-less so it fires regardless of the
    enabled-plugin filter.

    Attributes:
        events (List[Any]): Every event received, oldest first.
        lock (threading.Lock): Guards ``events``.
        callbacks (Dict[str, Callable]): Callback per event type.
        binding (Any): The hook binding, or None once closed.
    '''

    def __init__(self, callbacks: Optional[Dict[str, Callable]] = None):
        '''Start recording.

        Args:
            callbacks (Optional[Dict[str, Callable]]): Called with each event of
                the matching event type.
        '''
        self.events: List[Any] = []
        self.lock = threading.Lock()
        self.callbacks = callbacks or {}
        self.binding = orchify_broker.register_hook('*', self._on_event, owner='')

    def _on_event(self, event: Any) -> None:
        '''Record one event and dispatch its callback.

        Args:
            event (Any): The event emitted by the broker.
        '''
        with self.lock:
            self.events.append(event)
        callback = self.callbacks.get(event.event_type)
        if callback is not None:
            callback(event)

    def types(self) -> List[str]:
        '''Return the recorded event types, oldest first.

        Returns:
            List[str]: One event type per recorded event.
        '''
        with self.lock:
            return [event.event_type for event in self.events]

    def of(self, event_type: str) -> List[Any]:
        '''Return the recorded events of one type.

        Args:
            event_type (str): Event type to keep.

        Returns:
            List[Any]: Matching events, oldest first.
        '''
        with self.lock:
            return [event for event in self.events if event.event_type == event_type]

    def close(self) -> None:
        '''Detach the hook, leaving the recorded events in place.'''
        binding, self.binding = self.binding, None
        if binding is not None:
            orchify_broker.remove_binding(binding)

    def __enter__(self) -> 'Recorder':
        '''Return this recorder, so it can be used as a context manager.

        Returns:
            Recorder: This recorder.
        '''
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        '''Detach the hook on the way out of a ``with`` block.

        Args:
            exc_type (Any): Exception class, when one is propagating.
            exc (Any): Exception instance, when one is propagating.
            traceback (Any): Traceback, when an exception is propagating.

        Returns:
            bool: Always False, so exceptions keep propagating.
        '''
        self.close()
        return False


def wait_until(predicate: Callable[[], bool], timeout: float = 8.0) -> bool:
    '''Poll ``predicate`` until it is true or the deadline passes.

    Args:
        predicate (Callable[[], bool]): Condition to wait for.
        timeout (float): Seconds to keep polling.

    Returns:
        bool: The value of ``predicate`` at the deadline.
    '''
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def wait_finish(recorder: 'Recorder', timeout: float = 8.0) -> Any:
    '''Wait for the first ``run:finish`` event and return it.

    Args:
        recorder (Recorder): Recorder watching the broker.
        timeout (float): Seconds to wait.

    Returns:
        Any: The ``run:finish`` event.

    Raises:
        AssertionError: When no run finished before the deadline.
    '''
    assert wait_until(lambda: bool(recorder.of('run:finish')), timeout), \
        f'run never finished; events={recorder.types()}'
    return recorder.of('run:finish')[0]


def finish_payload(recorder: 'Recorder') -> dict:
    '''Return the payload of the first ``run:finish`` event.

    Args:
        recorder (Recorder): Recorder watching the broker.

    Returns:
        dict: The run:finish payload.
    '''
    return wait_finish(recorder).payload


__all__ = [
    'FakeLLM',
    'final',
    'tool_call',
    'Recorder',
    'wait_until',
    'wait_finish',
    'finish_payload',
]
