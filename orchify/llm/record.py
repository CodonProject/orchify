"""Record what an LLM returned and replay it later.

A recorded session is the cheapest way to debug an agent: the transcript does
not change, no tokens are spent, and a replay can run inside a test suite.

Example:
    >>> from orchify.llm import RecordLLM, ReplayLLM
    >>> llm = RecordLLM(OpenAICompat())
    >>> agent = Agent(name='demo', llm=llm)
    >>> agent.ask('write a haiku')
    >>> llm.save('runs/haiku.json')
    >>> replay = ReplayLLM.from_file('runs/haiku.json')
    >>> Agent(name='demo', llm=replay).ask('write a haiku')
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from .base import Chunk, FinalStatus, LLMInterface, Response

__all__ = ['RecordLLM', 'ReplayLLM', 'ReplayError', 'RecordingLLM']


class ReplayError(RuntimeError):
    '''Raised when a replay cannot answer a request.'''


_CHUNK_FIELDS = ('is_cot', 'content', 'is_assembly_tool', 'assembly_chunk',
                 'total_content', 'total_cot_content', 'total_tool_call',
                 'is_cot_end')
_FINAL_FIELDS = ('content', 'reasoning', 'tool_calls', 'finish_reason',
                 'completion_tokens', 'prompt_tokens', 'prompt_cache_hit_tokens',
                 'prompt_cache_miss_tokens', 'total_tokens')


def _encode(response: Response) -> Dict[str, Any]:
    '''Turn one streamed response into a JSON-able record.

    Args:
        response (Response): A response yielded by a provider.

    Returns:
        Dict[str, Any]: The encoded entry.
    '''
    chunk = response.current_chunk
    final = response.final_status
    entry: Dict[str, Any] = {'is_final': bool(response.is_final)}
    if chunk is not None:
        payload = {name: getattr(chunk, name) for name in _CHUNK_FIELDS}
        payload['total_tool_call'] = [dict(call) for call in payload['total_tool_call'] or []]
        entry['chunk'] = payload
    if final is not None:
        payload = {name: getattr(final, name) for name in _FINAL_FIELDS}
        payload['tool_calls'] = [dict(call) for call in payload['tool_calls'] or []]
        entry['final'] = payload
    return entry


def _decode(entry: Dict[str, Any]) -> Response:
    '''Rebuild a response from an encoded entry.

    Args:
        entry (Dict[str, Any]): One encoded response.

    Returns:
        Response: The replayed response.
    '''
    chunk_data = entry.get('chunk')
    final_data = entry.get('final')
    chunk = None
    final = None
    if chunk_data is not None:
        payload = dict(chunk_data)
        payload['total_tool_call'] = [dict(call) for call in payload.get('total_tool_call') or []]
        chunk = Chunk(**payload)
    if final_data is not None:
        payload = dict(final_data)
        payload['tool_calls'] = [dict(call) for call in payload.get('tool_calls') or []]
        final = FinalStatus(**payload)
    return Response(current_chunk=chunk, final_status=final,
                    is_final=bool(entry.get('is_final')))


class RecordingLLM(LLMInterface):
    '''Wrap a provider and remember every response it streamed.

    Attributes:
        provider (LLMInterface): The wrapped provider.
        calls (List[Dict[str, Any]]): Recorded calls, one per request.
    '''

    name = 'record'

    def __init__(self, provider: LLMInterface) -> None:
        '''Wrap a provider.

        Args:
            provider (LLMInterface): Provider whose answers are recorded.
        '''
        super().__init__()
        self.provider = provider
        self.calls: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None

    def __getattr__(self, item: str) -> Any:
        '''Delegate unknown attributes (``model``, ``messages``) to the provider.'''
        if item.startswith('_') or item == 'provider':
            raise AttributeError(item)
        return getattr(self.provider, item)

    async def _core_request(self, **kwargs: Any):
        '''Stream from the provider while recording each response.

        Args:
            **kwargs (Any): Provider request arguments.

        Yields:
            Response: Provider responses, unchanged.
        '''
        request = {key: value for key, value in kwargs.items() if key != 'scope'}
        record: Dict[str, Any] = {'request': json.loads(json.dumps(request, default=str)),
                                  'responses': []}
        self.calls.append(record)
        self._current = record
        try:
            async for response in self.provider._core_request(**kwargs):
                record['responses'].append(_encode(response))
                yield response
        finally:
            self._current = None

    def reset(self) -> None:
        '''Forget everything recorded so far.'''
        self.calls = []

    def as_dict(self) -> Dict[str, Any]:
        '''Return the recording as a JSON-able document.

        Returns:
            Dict[str, Any]: ``{'version': 1, 'calls': [...]}``.
        '''
        return {'version': 1, 'calls': json.loads(json.dumps(self.calls, default=str))}

    def save(self, path: str) -> str:
        '''Write the recording to ``path``.

        Args:
            path (str): Destination JSON file.

        Returns:
            str: The path written.
        '''
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(self.as_dict(), handle, ensure_ascii=False, indent=2)
        return path


RecordLLM = RecordingLLM


class ReplayLLM(LLMInterface):
    '''Serve recorded responses instead of calling a provider.

    Attributes:
        calls (List[Dict[str, Any]]): The recording being replayed.
        strict (bool): Compare incoming requests with the recorded ones.
        position (int): Index of the next call to serve.
    '''

    name = 'replay'

    def __init__(self, calls: Iterable[Dict[str, Any]] = (), *,
                 strict: bool = False) -> None:
        '''Load a recording.

        Args:
            calls (Iterable[Dict[str, Any]]): Recorded calls.
            strict (bool): Raise when a request differs from the recording.
        '''
        super().__init__()
        self.calls = [dict(call) for call in calls]
        self.strict = bool(strict)
        self.position = 0
        self.requests: List[Dict[str, Any]] = []

    @classmethod
    def from_file(cls, path: str, **kwargs: Any) -> 'ReplayLLM':
        '''Load a recording written by :meth:`RecordingLLM.save`.

        Args:
            path (str): Recorded JSON file.
            **kwargs (Any): Forwarded to :class:`ReplayLLM`.

        Returns:
            ReplayLLM: Ready to serve.

        Raises:
            ReplayError: When the file is missing or damaged.
        '''
        if not os.path.exists(path):
            raise ReplayError(f'recording not found: {path}')
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                document = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ReplayError(f'cannot read recording {path}: {exc}') from exc
        calls = document.get('calls') if isinstance(document, dict) else document
        if not isinstance(calls, list):
            raise ReplayError(f'recording {path} has no calls')
        return cls(calls, **kwargs)

    def reset(self) -> None:
        '''Rewind to the first recorded call.'''
        self.position = 0
        self.requests = []

    @property
    def remaining(self) -> int:
        '''Number of calls left to replay.'''
        return max(0, len(self.calls) - self.position)

    async def _core_request(self, **kwargs: Any):
        '''Serve the next recorded call.

        Args:
            **kwargs (Any): Provider request arguments.

        Yields:
            Response: The recorded responses.

        Raises:
            ReplayError: When the recording ran out or the request changed.
        '''
        if self.position >= len(self.calls):
            raise ReplayError(
                f'recording exhausted after {len(self.calls)} call(s); '
                f'request #{self.position + 1} has no answer')
        record = self.calls[self.position]
        request = {key: value for key, value in kwargs.items() if key != 'scope'}
        self.requests.append(json.loads(json.dumps(request, default=str)))
        if self.strict:
            expected = record.get('request')
            if expected is not None and json.loads(json.dumps(request, default=str)) != expected:
                raise ReplayError(
                    f'request #{self.position + 1} differs from the recording; '
                    'disable strict mode to replay anyway')
        self.position += 1
        for entry in record.get('responses') or []:
            yield _decode(entry)
