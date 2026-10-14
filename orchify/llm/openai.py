import json
import asyncio
import os
import httpx

from typing import AsyncGenerator, Optional, Literal

from orchify.llm.base import Chunk, FinalStatus, Response, LLMInterface
from orchify.env import req_key, req_url, req_model
from orchify.backend import orchify_web_backend


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_MAX_RETRY_DELAY = 30.0


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw or raw.strip() == '':
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw or raw.strip() == '':
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _is_retryable(exc: Exception) -> bool:
    '''429/5xx and any transport error (including timeouts) are transient.'''
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


def _retry_delay(attempt: int, exc: Exception) -> float:
    '''Exponential backoff, honoring a Retry-After header when the server sent one.'''
    response = getattr(exc, 'response', None)
    if response is not None:
        raw = response.headers.get('Retry-After')
        if raw:
            try:
                return max(0.0, min(float(raw), _MAX_RETRY_DELAY))
            except (TypeError, ValueError):
                pass
    backoff = _read_float_env('ORCHIFY_LLM_RETRY_BACKOFF', 0.5)
    return min(backoff * (2 ** attempt), _MAX_RETRY_DELAY)


class OpenAICompat(LLMInterface):
    '''LLM backend for any OpenAI-compatible chat API.

    Reads ``API_KEY`` (or ``OPENAI_API_KEY``), ``BASE_URL`` and ``MODEL_ID``
    from the environment and streams ``/chat/completions`` responses, mapping
    provider SSE chunks onto :class:`Chunk` and :class:`FinalStatus`.

    Attributes:
        api_key (str): Bearer token sent with each request.
        base_url (str): API base URL.
        url (str): Path appended to ``base_url`` (``/chat/completions``).
    '''

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        '''Create the backend.

        Args:
            api_key (Optional[str]): API key; defaults to ``API_KEY`` (or
                ``OPENAI_API_KEY``) from the environment.
            base_url (Optional[str]): API root; defaults to ``BASE_URL`` from the
                environment, then to 'https://api.openai.com/v1'.
        '''
        super().__init__()
        self.api_key  = api_key if api_key is not None else req_key()
        self.base_url = base_url if base_url is not None else req_url('https://api.openai.com/v1')
        self.url = '/chat/completions'

    @property
    def full_url(self) -> str:
        '''Return the configured chat-completions URL.

        Returns:
            str: ``base_url`` with ``/chat/completions`` appended once.
        '''
        return f'{self.base_url}{self.url}'

    def _build_payload(
        self,
        messages: list[dict],
        model: str,
        tools: Optional[list[dict]],
        temperature: float,
        top_p: float,
        json_format: bool,
        max_tokens: Optional[int],
        thinking: Literal['disabled', 'enabled', 'auto'],
        effort: Literal['minimal', 'low', 'medium', 'high', 'xhigh'],
        extra_data: Optional[dict],
    ) -> dict:
        if isinstance(thinking, bool):
            thinking = 'enabled' if thinking else 'disabled'
        elif thinking in ('disable', 'off', 'false'):
            thinking = 'disabled'
        elif thinking in ('on', 'true'):
            thinking = 'enabled'

        data = {
            'model': model,
            'messages': messages,
            'stream': True,
            'stream_options': {'include_usage': True},
        }

        if tools:
            data['tools'] = tools

        is_openai_reasoning = model.startswith(('o1', 'o3'))
        if not is_openai_reasoning:
            data['temperature'] = temperature
            data['top_p'] = top_p

        if json_format:
            data['response_format'] = {'type': 'json_object'}

        if max_tokens is not None:
            data['max_tokens'] = max_tokens

        is_deepseek = 'api.deepseek.com' in self.base_url or 'deepseek' in model.lower()

        if is_openai_reasoning:
            if thinking != 'disabled':
                effort_map = {
                    'minimal': 'low',
                    'low': 'low',
                    'medium': 'medium',
                    'high': 'high',
                    'xhigh': 'high'
                }
                data['reasoning_effort'] = effort_map.get(effort, 'medium')
        elif is_deepseek:
            ds_thinking_type = 'adaptive' if thinking == 'auto' else thinking
            data['thinking'] = {'type': ds_thinking_type}
        else:
            if thinking in ('enabled', 'disabled'):
                data['thinking'] = {'type': thinking}

        if extra_data:
            data.update(extra_data)

        return data

    async def _resilient_lines(self, url: str, headers: dict, data: dict) -> AsyncGenerator[str, None]:
        '''Stream response lines, retrying transient failures.

        A retry is safe only while nothing has been streamed yet (the caller has
        seen no partial content): connect errors, timeouts before the first line and
        error statuses such as 429 or 503 all qualify. A connection that dies
        mid-answer is raised as-is, because the caller already consumed part of the
        answer and a silent replay could duplicate output.

        ``ORCHIFY_LLM_MAX_RETRIES`` (default 2) and ``ORCHIFY_LLM_RETRY_BACKOFF``
        (default 0.5 seconds, exponential) control this; a ``Retry-After`` header
        wins over the backoff.

        Args:
            url (str): Chat-completions URL.
            headers (dict): Request headers.
            data (dict): JSON request body.

        Yields:
            str: Non-empty response lines from the stream.
        '''
        attempt = 0
        while True:
            produced = False
            try:
                async for line in orchify_web_backend.post_stream_async(url, headers=headers, json_data=data):
                    produced = True
                    yield line
                return
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                max_retries = _read_int_env('ORCHIFY_LLM_MAX_RETRIES', 2)
                if produced or attempt >= max_retries or not _is_retryable(exc):
                    raise
                delay = _retry_delay(attempt, exc)
                attempt += 1
                if delay > 0:
                    await asyncio.sleep(delay)

    async def _core_request(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        json_format: bool = False,
        max_tokens: Optional[int] = None,
        thinking: Literal['disabled', 'enabled', 'auto'] = 'auto',
        effort: Literal['minimal', 'low', 'medium', 'high', 'xhigh'] = 'medium',
        extra_data: Optional[dict] = None,
    ) -> AsyncGenerator[Response, None]:
        model = model or req_model('gpt-4o')
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = self._build_payload(messages, model, tools, temperature, top_p,
                                   json_format, max_tokens, thinking, effort, extra_data)

        total_content = ''
        total_cot_content = ''
        tool_calls_map = {}

        completion_tokens = 0
        prompt_tokens = 0
        prompt_cache_hit_tokens = 0
        prompt_cache_miss_tokens = 0
        total_tokens = 0

        has_cot = False
        cot_ended = False

        def make_chunk(**kwargs) -> Response:
            return Response(
                current_chunk=Chunk(
                    total_content=total_content,
                    total_cot_content=total_cot_content,
                    total_tool_call=list(tool_calls_map.values()),
                    **kwargs
                ),
                is_final=False
            )

        try:
            stream_gen = self._resilient_lines(self.full_url, headers, data)
            async for line_bytes in stream_gen:
                if not line_bytes:
                    continue

                line_str = line_bytes.decode('utf-8') if isinstance(line_bytes, bytes) else line_bytes
                if not line_str.startswith('data: '):
                    continue

                raw_data = line_str[6:].strip()
                if raw_data == '[DONE]':
                    break

                try:
                    chunk = json.loads(raw_data)
                except json.JSONDecodeError:
                    continue

                usage = chunk.get('usage')
                if usage:
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)

                    prompt_details = usage.get('prompt_tokens_details', {})
                    cached_tokens = prompt_details.get('cached_tokens')
                    if cached_tokens is not None:
                        prompt_cache_hit_tokens = cached_tokens
                    else:
                        prompt_cache_hit_tokens = usage.get('prompt_cache_hit_tokens', 0)

                    prompt_cache_miss_tokens = usage.get('prompt_cache_miss_tokens')
                    if prompt_cache_miss_tokens is None:
                        prompt_cache_miss_tokens = max(0, prompt_tokens - prompt_cache_hit_tokens)

                choices = chunk.get('choices', [])
                if not choices:
                    continue

                delta = choices[0].get('delta', {})

                cot_chunk = delta.get('reasoning_content') or delta.get('reasoning') or ""
                content_chunk = delta.get('content') or ""

                tool_calls_delta = delta.get('tool_calls', [])
                is_assembly_tool = False
                assembly_chunk = ""

                if tool_calls_delta:
                    is_assembly_tool = True
                    for tool_call in tool_calls_delta:
                        idx = tool_call.get('index')
                        if idx is None:
                            continue

                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                'id': '',
                                'type': 'function',
                                'function': {'name': '', 'arguments': ''}
                            }

                        if tool_call.get('id'):
                            tool_calls_map[idx]['id'] = tool_call['id']
                        if tool_call.get('type'):
                            tool_calls_map[idx]['type'] = tool_call['type']

                        func_delta = tool_call.get('function', {})
                        if func_delta.get('name'):
                            tool_calls_map[idx]['function']['name'] += func_delta['name']
                        if func_delta.get('arguments'):
                            arg_part = func_delta['arguments']
                            tool_calls_map[idx]['function']['arguments'] += arg_part
                            assembly_chunk += arg_part

                if cot_chunk:
                    total_cot_content += cot_chunk
                if content_chunk:
                    total_content += content_chunk

                is_cot_end = False
                if cot_chunk:
                    has_cot = True
                elif has_cot and not cot_ended:
                    is_cot_end = True
                    cot_ended = True

                if is_cot_end:
                    yield make_chunk(is_cot=False, content="")
                    is_cot_end = False

                if cot_chunk:
                    yield make_chunk(is_cot=True, content=cot_chunk)

                if content_chunk:
                    yield make_chunk(is_cot=False, content=content_chunk, is_cot_end=is_cot_end)

                if is_assembly_tool and not (cot_chunk or content_chunk):
                    yield make_chunk(is_cot=False, content="", is_assembly_tool=True,
                                     assembly_chunk=assembly_chunk, is_cot_end=is_cot_end)

            if has_cot and not cot_ended:
                yield make_chunk(is_cot=False, content="", is_cot_end=True)
        except httpx.HTTPStatusError as e:
            try:
                error_detail = e.response.json()
            except Exception:
                try:
                    error_detail = e.response.text
                except Exception:
                    error_detail = ""
            error_msg = f"HTTP Error '{e.response.status_code}': {e}"
            if error_detail:
                error_msg += f' | Response detail: {error_detail}'
            raise httpx.HTTPStatusError(error_msg, request=e.request, response=e.response) from e

        yield Response(
            current_chunk=None,
            final_status=FinalStatus(
                content=total_content,
                reasoning=total_cot_content,
                tool_calls=list(tool_calls_map.values()),
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                prompt_cache_miss_tokens=prompt_cache_miss_tokens,
                total_tokens=total_tokens
            ),
            is_final=True
        )
