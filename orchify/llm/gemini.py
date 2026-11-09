"""Native Google Gemini (generateContent) streaming adapter.

Gemini's REST wire format differs from both OpenAI and Anthropic: ``contents``
with ``parts``, ``functionCall``/``functionResponse`` parts, and a
``systemInstruction`` object. This adapter translates in both directions so the
agent runtime can talk to Gemini without a compatibility proxy.

Usage::

    from orchify import Agent
    from orchify.llm.gemini import GeminiLLM

    agent = Agent(llm=GeminiLLM(model='gemini-2.5-flash'), model='gemini-2.5-flash')

Credentials and endpoint come from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) and
``GEMINI_BASE_URL``.
"""

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from orchify.llm.base import Chunk, FinalStatus, LLMInterface, Response
from orchify.llm.retry import stream_sse
from orchify.utils import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com'
DEFAULT_MODEL = 'gemini-2.5-flash'

#: Gemini ``finishReason`` -> the provider-neutral reason the runtime reports.
_FINISH_REASONS = {
    'STOP': 'stop',
    'MAX_TOKENS': 'length',
    'SAFETY': 'content_filter',
    'RECITATION': 'content_filter',
    'PROHIBITED_CONTENT': 'content_filter',
    'BLOCKLIST': 'content_filter',
    'OTHER': '',
    'FINISH_REASON_UNSPECIFIED': '',
}


def _env(*names: str) -> str:
    '''Return the first non-empty environment variable among ``names``.'''
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ''


def _text_of(content: Any) -> str:
    '''Flatten a message content value into plain text.'''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(
            str(part.get('text') or '') for part in content
            if isinstance(part, dict) and part.get('type') == 'text'
        )
    return '' if content is None else str(content)


def _user_parts(content: Any) -> List[Dict[str, Any]]:
    '''Convert an OpenAI user content value into Gemini parts.

    Args:
        content (Any): String or list of content parts.

    Returns:
        List[Dict[str, Any]]: Gemini parts (text and inline image data).
    '''
    if isinstance(content, str):
        return [{'text': content}] if content else []
    parts: List[Dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get('type') == 'text':
            text = str(part.get('text') or '')
            if text:
                parts.append({'text': text})
        elif part.get('type') == 'image_url':
            url = part.get('image_url')
            if isinstance(url, dict):
                url = url.get('url')
            url = str(url or '')
            if url.startswith('data:'):
                header, _, data = url.partition(',')
                mime = header[5:].split(';')[0] or 'image/png'
                parts.append({'inlineData': {'mimeType': mime, 'data': data}})
    return parts


def to_gemini_contents(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    '''Translate OpenAI-style messages into Gemini ``system`` + ``contents``.

    Args:
        messages (List[Dict[str, Any]]): Conversation in OpenAI wire format.

    Returns:
        Tuple[str, List[Dict[str, Any]]]: The system instruction text and contents.
    '''
    system_parts: List[str] = []
    contents: List[Dict[str, Any]] = []
    names_by_call: Dict[str, str] = {}
    for message in messages or []:
        role = message.get('role') or 'user'
        content = message.get('content')
        if role in ('system', 'developer'):
            text = _text_of(content)
            if text:
                system_parts.append(text)
            continue
        if role == 'tool':
            name = names_by_call.get(str(message.get('tool_call_id') or ''), 'tool')
            contents.append({'role': 'user', 'parts': [{
                'functionResponse': {'name': name, 'response': {'result': _text_of(content) or '(empty)'}},
            }]})
            continue
        if role == 'assistant':
            parts: List[Dict[str, Any]] = []
            text = _text_of(content)
            if text:
                parts.append({'text': text})
            for call in message.get('tool_calls') or []:
                function = call.get('function') or {}
                name = str(function.get('name') or '')
                names_by_call[str(call.get('id') or '')] = name
                try:
                    args = json.loads(function.get('arguments') or '{}')
                except (TypeError, ValueError):
                    args = {}
                parts.append({'functionCall': {'name': name, 'args': args if isinstance(args, dict) else {}}})
            if parts:
                contents.append({'role': 'model', 'parts': parts})
            continue
        parts = _user_parts(content)
        if parts:
            contents.append({'role': 'user', 'parts': parts})
    return '\n\n'.join(system_parts), contents


def to_gemini_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    '''Translate OpenAI function schemas into Gemini function declarations.

    Args:
        tools (Optional[List[Dict[str, Any]]]): OpenAI tool schemas.

    Returns:
        List[Dict[str, Any]]: Gemini ``tools`` value.
    '''
    declarations: List[Dict[str, Any]] = []
    for entry in tools or []:
        function = entry.get('function') if isinstance(entry, dict) else None
        if not function:
            continue
        declarations.append({
            'name': str(function.get('name') or ''),
            'description': str(function.get('description') or ''),
            'parameters': function.get('parameters') or {'type': 'object', 'properties': {}},
        })
    return [{'functionDeclarations': declarations}] if declarations else []


class GeminiLLM(LLMInterface):
    '''Streaming Gemini backend for the Orchify agent runtime.

    Args:
        api_key (Optional[str]): API key; defaults to ``GEMINI_API_KEY`` or
            ``GOOGLE_API_KEY``.
        base_url (Optional[str]): API root; defaults to ``GEMINI_BASE_URL``.
        model (str): Default model id.
        max_tokens (Optional[int]): Output limit (``maxOutputTokens``).
        max_retries (int): Extra attempts for transient failures.
        backoff (float): Base retry delay in seconds.
    '''

    name = 'gemini'

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: str = DEFAULT_MODEL, max_tokens: Optional[int] = None,
                 max_retries: int = 2, backoff: float = 0.5):
        super().__init__()
        self.api_key = api_key or _env('GEMINI_API_KEY', 'GOOGLE_API_KEY')
        self.base_url = (base_url or os.environ.get('GEMINI_BASE_URL') or DEFAULT_BASE_URL).rstrip('/')
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max(0, int(max_retries))
        self.backoff = max(0.0, float(backoff))

    def url_for(self, model: str) -> str:
        '''Return the streaming endpoint for ``model``.

        Args:
            model (str): Model id, e.g. ``gemini-2.5-flash``.

        Returns:
            str: The ``:streamGenerateContent`` URL with ``alt=sse``.
        '''
        return f'{self.base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse'

    def missing_credentials(self) -> Optional[str]:
        '''Explain how to authenticate when no Gemini key is configured.'''
        if self.api_key:
            return None
        return 'Gemini needs a key: set GEMINI_API_KEY (or GOOGLE_API_KEY) or pass api_key=...'

    def _build_payload(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
                       temperature: float, top_p: float, json_format: bool,
                       max_tokens: Optional[int]) -> Dict[str, Any]:
        '''Build the ``generateContent`` request body.'''
        system, contents = to_gemini_contents(messages)
        payload: Dict[str, Any] = {'contents': contents}
        if system:
            payload['systemInstruction'] = {'parts': [{'text': system}]}
        gemini_tools = to_gemini_tools(tools)
        if gemini_tools:
            payload['tools'] = gemini_tools
        config: Dict[str, Any] = {}
        if temperature is not None:
            config['temperature'] = float(temperature)
        if top_p is not None and float(top_p) < 1.0:
            config['topP'] = float(top_p)
        limit = max_tokens or self.max_tokens
        if limit:
            config['maxOutputTokens'] = int(limit)
        if json_format:
            config['responseMimeType'] = 'application/json'
        if config:
            payload['generationConfig'] = config
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
        '''Stream one Gemini completion, translated into Orchify responses.'''
        if not self.api_key:
            raise RuntimeError(self.missing_credentials() or 'Gemini API key is missing')
        model = model or self.model
        headers = {'x-goog-api-key': self.api_key, 'content-type': 'application/json'}
        payload = self._build_payload(messages, tools, temperature, top_p, json_format, max_tokens)
        if extra_data:
            payload.update(extra_data)

        total_content = ''
        total_cot_content = ''
        tool_calls: List[Dict[str, Any]] = []
        finish_reason = ''
        usage: Dict[str, Any] = {}
        counter = 0

        def chunk(content: str = '', *, is_cot: bool = False, **kwargs: Any) -> Response:
            return Response(
                current_chunk=Chunk(
                    is_cot=is_cot,
                    content=content,
                    total_content=total_content,
                    total_cot_content=total_cot_content,
                    total_tool_call=list(tool_calls),
                    **kwargs,
                ),
                is_final=False,
            )

        async for raw in stream_sse(self.url_for(model), headers, payload,
                                    max_retries=self.max_retries, backoff=self.backoff,
                                    label='gemini'):
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
            usage.update(event.get('usageMetadata') or {})
            for candidate in event.get('candidates') or []:
                finish_reason = _FINISH_REASONS.get(str(candidate.get('finishReason') or ''), finish_reason)
                parts = ((candidate.get('content') or {}).get('parts')) or []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if part.get('functionCall'):
                        call = part['functionCall'] or {}
                        counter += 1
                        arguments = json.dumps(call.get('args') or {}, ensure_ascii=False)
                        tool_calls.append({
                            'id': f'call_{counter}',
                            'type': 'function',
                            'function': {'name': str(call.get('name') or ''), 'arguments': arguments},
                        })
                        yield chunk(is_assembly_tool=True, assembly_chunk=arguments)
                        continue
                    text = part.get('text')
                    if not text:
                        continue
                    if part.get('thought'):
                        total_cot_content += str(text)
                        yield chunk(is_cot=True, content=str(text))
                    else:
                        total_content += str(text)
                        yield chunk(content=str(text))

        prompt_tokens = int(usage.get('promptTokenCount') or 0)
        cache_hit = int(usage.get('cachedContentTokenCount') or 0)
        completion_tokens = int(usage.get('candidatesTokenCount') or 0)
        yield Response(
            current_chunk=None,
            final_status=FinalStatus(
                content=total_content,
                reasoning=total_cot_content,
                tool_calls=list(tool_calls),
                finish_reason=finish_reason,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                prompt_cache_hit_tokens=cache_hit,
                prompt_cache_miss_tokens=max(0, prompt_tokens - cache_hit),
                total_tokens=int(usage.get('totalTokenCount') or (prompt_tokens + completion_tokens)),
            ),
            is_final=True,
        )


__all__ = ['GeminiLLM', 'to_gemini_contents', 'to_gemini_tools', 'DEFAULT_BASE_URL', 'DEFAULT_MODEL']
