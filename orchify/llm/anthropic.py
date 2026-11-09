"""Native Anthropic (Messages API) streaming adapter.

The OpenAI-compatible adapter covers any endpoint that speaks the OpenAI wire
format. Claude's native API is different enough (top-level ``system``, typed
content blocks, ``tool_use``/``tool_result`` blocks, thinking blocks) to deserve
its own translation layer; this module implements the streaming path only, which
is what the agent runtime uses.

Usage::

    from orchify import Agent
    from orchify.llm.anthropic import AnthropicLLM

    agent = Agent(llm=AnthropicLLM(api_key='sk-ant-...'), model='claude-sonnet-4-5')

Credentials and endpoint come from ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``
when the constructor arguments are omitted.
"""

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from orchify.llm.base import Chunk, FinalStatus, LLMInterface, Response
from orchify.llm.retry import stream_sse
from orchify.utils import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = 'https://api.anthropic.com'
DEFAULT_MODEL = 'claude-sonnet-4-5'
DEFAULT_MAX_TOKENS = 4096
DEFAULT_VERSION = '2023-06-01'

#: Anthropic ``stop_reason`` -> the provider-neutral reason the runtime reports.
_FINISH_REASONS = {
    'end_turn': 'stop',
    'stop_sequence': 'stop',
    'max_tokens': 'length',
    'tool_use': 'tool_calls',
    'pause_turn': 'stop',
    'refusal': 'content_filter',
}

_RETRYABLE_HINT = 'anthropic'


def _env(name: str, default: str = '') -> str:
    '''Read an environment variable, falling back to ``default``.'''
    value = os.environ.get(name)
    return value if value else default


def _json_arguments(arguments: Any) -> Dict[str, Any]:
    '''Parse a tool-call argument string into a JSON object.

    Args:
        arguments (Any): JSON string, dict, or None.

    Returns:
        Dict[str, Any]: The parsed arguments (empty when unparsable).
    '''
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _text_of(content: Any) -> str:
    '''Flatten a message content value into plain text.'''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part for part in content if isinstance(part, dict) and part.get('type') == 'text']
        return '\n'.join(str(part.get('text') or '') for part in parts)
    return '' if content is None else str(content)


def _user_blocks(content: Any) -> List[Dict[str, Any]]:
    '''Convert an OpenAI-style user content value into Anthropic content blocks.

    Text parts become ``text`` blocks; ``image_url`` parts become ``image``
    blocks with a base64 or URL source, so multimodal messages survive the
    conversion.

    Args:
        content (Any): String or list of content parts.

    Returns:
        List[Dict[str, Any]]: Anthropic content blocks.
    '''
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}] if content else []
    blocks: List[Dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get('type')
        if kind == 'text':
            text = str(part.get('text') or '')
            if text:
                blocks.append({'type': 'text', 'text': text})
        elif kind == 'image_url':
            url = part.get('image_url')
            if isinstance(url, dict):
                url = url.get('url')
            url = str(url or '')
            if url.startswith('data:'):
                header, _, data = url.partition(',')
                media_type = header[5:].split(';')[0] or 'image/png'
                blocks.append({'type': 'image', 'source': {
                    'type': 'base64', 'media_type': media_type, 'data': data}})
            elif url:
                blocks.append({'type': 'image', 'source': {'type': 'url', 'url': url}})
    return blocks


def to_anthropic_messages(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    '''Translate OpenAI-style messages into Anthropic ``system`` + ``messages``.

    System messages are hoisted into the top-level system prompt. An assistant
    ``tool_calls`` turn becomes ``tool_use`` blocks, and every ``role: tool``
    message becomes a ``tool_result`` block on a user turn; consecutive tool
    results are merged into one user message because the Messages API requires
    alternating roles.

    Args:
        messages (List[Dict[str, Any]]): Conversation in OpenAI wire format.

    Returns:
        Tuple[str, List[Dict[str, Any]]]: The system prompt and the messages.
    '''
    system_parts: List[str] = []
    converted: List[Dict[str, Any]] = []
    for message in messages or []:
        role = message.get('role') or 'user'
        content = message.get('content')
        if role in ('system', 'developer'):
            text = _text_of(content)
            if text:
                system_parts.append(text)
            continue
        if role == 'tool':
            block = {
                'type': 'tool_result',
                'tool_use_id': str(message.get('tool_call_id') or ''),
                'content': _text_of(content) or '(empty)',
            }
            last = converted[-1] if converted else None
            if last is not None and last.get('role') == 'user' and _only_tool_results(last):
                last['content'].append(block)
            else:
                converted.append({'role': 'user', 'content': [block]})
            continue
        if role == 'assistant':
            blocks: List[Dict[str, Any]] = []
            text = _text_of(content)
            if text:
                blocks.append({'type': 'text', 'text': text})
            for call in message.get('tool_calls') or []:
                function = call.get('function') or {}
                blocks.append({
                    'type': 'tool_use',
                    'id': str(call.get('id') or ''),
                    'name': str(function.get('name') or ''),
                    'input': _json_arguments(function.get('arguments')),
                })
            if blocks:
                converted.append({'role': 'assistant', 'content': blocks})
            continue
        blocks = _user_blocks(content)
        if blocks:
            converted.append({'role': 'user', 'content': blocks})
    return '\n\n'.join(system_parts), converted


def _only_tool_results(message: Dict[str, Any]) -> bool:
    '''Whether a converted message consists only of tool_result blocks.'''
    return all(block.get('type') == 'tool_result' for block in message.get('content') or [])


def to_anthropic_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    '''Translate OpenAI function schemas into Anthropic tool definitions.

    Args:
        tools (Optional[List[Dict[str, Any]]]): OpenAI tool schemas.

    Returns:
        List[Dict[str, Any]]: Anthropic tool definitions.
    '''
    converted: List[Dict[str, Any]] = []
    for entry in tools or []:
        function = entry.get('function') if isinstance(entry, dict) else None
        if not function:
            continue
        converted.append({
            'name': str(function.get('name') or ''),
            'description': str(function.get('description') or ''),
            'input_schema': function.get('parameters') or {'type': 'object', 'properties': {}},
        })
    return converted


class AnthropicLLM(LLMInterface):
    '''Streaming Claude backend for the Orchify agent runtime.

    Args:
        api_key (Optional[str]): Anthropic API key; defaults to ``ANTHROPIC_API_KEY``.
        base_url (Optional[str]): API root; defaults to ``ANTHROPIC_BASE_URL`` or
            ``https://api.anthropic.com``.
        model (str): Default model id.
        max_tokens (int): Default output limit (the Messages API requires one).
        version (str): ``anthropic-version`` header value.
        thinking_budget (Optional[int]): Token budget when thinking is enabled.
        max_retries (int): Extra attempts for transient failures.
        backoff (float): Base retry delay in seconds.
    '''

    name = 'anthropic'

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS,
                 version: str = DEFAULT_VERSION, thinking_budget: Optional[int] = None,
                 max_retries: int = 2, backoff: float = 0.5):
        super().__init__()
        self.api_key = api_key or _env('ANTHROPIC_API_KEY')
        self.base_url = (base_url or _env('ANTHROPIC_BASE_URL', DEFAULT_BASE_URL)).rstrip('/')
        self.model = model
        self.max_tokens = int(max_tokens)
        self.version = version
        self.thinking_budget = thinking_budget
        self.max_retries = max(0, int(max_retries))
        self.backoff = max(0.0, float(backoff))

    @property
    def full_url(self) -> str:
        '''The streaming Messages API endpoint.'''
        if self.base_url.endswith('/v1'):
            return f'{self.base_url}/messages'
        return f'{self.base_url}/v1/messages'

    def missing_credentials(self) -> Optional[str]:
        '''Explain how to authenticate when ``ANTHROPIC_API_KEY`` is missing.'''
        if self.api_key:
            return None
        return "Anthropic needs a key: set ANTHROPIC_API_KEY or pass api_key='sk-ant-...'."

    def _build_payload(self, messages: List[Dict[str, Any]], model: str,
                       tools: Optional[List[Dict[str, Any]]], temperature: float,
                       top_p: float, max_tokens: Optional[int],
                       thinking: str) -> Dict[str, Any]:
        '''Build the Messages API request body.

        Args:
            messages (List[Dict[str, Any]]): Conversation in OpenAI format.
            model (str): Model id.
            tools (Optional[List[Dict[str, Any]]]): Tool schemas.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability.
            max_tokens (Optional[int]): Output limit override.
            thinking (str): ``'enabled'`` turns on extended thinking.

        Returns:
            Dict[str, Any]: The JSON request body.
        '''
        system, converted = to_anthropic_messages(messages)
        payload: Dict[str, Any] = {
            'model': model,
            'max_tokens': int(max_tokens or self.max_tokens),
            'messages': converted,
            'stream': True,
        }
        if system:
            payload['system'] = system
        if temperature is not None:
            payload['temperature'] = float(temperature)
        if top_p is not None and float(top_p) < 1.0:
            payload['top_p'] = float(top_p)
        anthropic_tools = to_anthropic_tools(tools)
        if anthropic_tools:
            payload['tools'] = anthropic_tools
        if thinking == 'enabled' and self.thinking_budget:
            payload['thinking'] = {'type': 'enabled', 'budget_tokens': int(self.thinking_budget)}
        return payload

    async def _core_request(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        json_format: bool = False,
        max_tokens: Optional[int] = None,
        thinking: str = 'auto',
        effort: str = 'medium',
        extra_data: Optional[dict] = None,
    ) -> AsyncGenerator[Response, None]:
        '''Stream one Claude completion, translated into Orchify responses.'''
        if not self.api_key:
            raise RuntimeError(self.missing_credentials() or 'Anthropic API key is missing')
        model = model or self.model
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': self.version,
            'content-type': 'application/json',
        }
        payload = self._build_payload(messages, model, tools, temperature, top_p,
                                      max_tokens, thinking)
        if extra_data:
            payload.update(extra_data)

        total_content = ''
        total_cot_content = ''
        tool_calls: Dict[int, Dict[str, Any]] = {}
        arg_parts: Dict[int, List[str]] = {}
        dirty: set = set()
        block_index = -1
        open_block: Dict[str, Any] = {}
        finish_reason = ''
        usage: Dict[str, int] = {}

        def snapshot() -> List[Dict[str, Any]]:
            while dirty:
                index = dirty.pop()
                tool_calls[index]['function']['arguments'] = ''.join(arg_parts[index])
            return list(tool_calls.values())

        def chunk(content: str = '', *, is_cot: bool = False, **kwargs: Any) -> Response:
            return Response(
                current_chunk=Chunk(
                    is_cot=is_cot,
                    content=content,
                    total_content=total_content,
                    total_cot_content=total_cot_content,
                    total_tool_call=snapshot(),
                    **kwargs,
                ),
                is_final=False,
            )

        async for raw in stream_sse(self.full_url, headers, payload,
                                    max_retries=self.max_retries, backoff=self.backoff,
                                    label='anthropic'):
            line = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if not data or data == '[DONE]':
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            kind = event.get('type')

            if kind == 'message_start':
                usage.update((event.get('message') or {}).get('usage') or {})
                continue
            if kind == 'content_block_start':
                block_index = int(event.get('index') or 0)
                block = event.get('content_block') or {}
                block_type = block.get('type')
                open_block = {'type': block_type, 'index': block_index}
                if block_type == 'tool_use':
                    tool_calls[block_index] = {
                        'id': str(block.get('id') or ''),
                        'type': 'function',
                        'function': {'name': str(block.get('name') or ''), 'arguments': ''},
                    }
                    arg_parts[block_index] = []
                elif block_type in ('text', 'thinking'):
                    if block_type == 'text' and block.get('text'):
                        total_content += str(block['text'])
                        yield chunk(content=str(block['text']))
                    if block_type == 'thinking' and block.get('thinking'):
                        total_cot_content += str(block['thinking'])
                        yield chunk(is_cot=True, content=str(block['thinking']))
                continue
            if kind == 'content_block_delta':
                delta = event.get('delta') or {}
                delta_type = delta.get('type')
                if delta_type == 'text_delta':
                    text = str(delta.get('text') or '')
                    if text:
                        total_content += text
                        yield chunk(content=text)
                elif delta_type == 'thinking_delta':
                    thought = str(delta.get('thinking') or '')
                    if thought:
                        total_cot_content += thought
                        yield chunk(is_cot=True, content=thought)
                elif delta_type == 'input_json_delta':
                    partial = str(delta.get('partial_json') or '')
                    if partial:
                        index = int(event.get('index') or 0)
                        arg_parts.setdefault(index, []).append(partial)
                        dirty.add(index)
                        yield chunk(is_assembly_tool=True, assembly_chunk=partial)
                continue
            if kind == 'content_block_stop':
                if open_block.get('type') == 'text' and total_cot_content:
                    yield chunk(is_cot_end=True)
                open_block = {}
                continue
            if kind == 'message_delta':
                delta = event.get('delta') or {}
                finish_reason = _FINISH_REASONS.get(str(delta.get('stop_reason') or ''), '')
                usage.update(event.get('usage') or {})
                continue
            if kind == 'error':
                detail = event.get('error') or {}
                raise RuntimeError(f"Anthropic error: {detail.get('type')}: {detail.get('message')}")

        prompt_tokens = int(usage.get('input_tokens') or 0)
        cache_hit = int(usage.get('cache_read_input_tokens') or 0)
        cache_miss = int(usage.get('cache_creation_input_tokens') or 0) or max(0, prompt_tokens - cache_hit)
        completion_tokens = int(usage.get('output_tokens') or 0)
        yield Response(
            current_chunk=None,
            final_status=FinalStatus(
                content=total_content,
                reasoning=total_cot_content,
                tool_calls=snapshot(),
                finish_reason=finish_reason,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                prompt_cache_hit_tokens=cache_hit,
                prompt_cache_miss_tokens=cache_miss,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            is_final=True,
        )


__all__ = ['AnthropicLLM', 'to_anthropic_messages', 'to_anthropic_tools',
           'DEFAULT_BASE_URL', 'DEFAULT_MODEL']
