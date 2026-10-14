import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx

from orchify.backend import orchify_web_backend
from orchify.event import AgentEvent
from orchify.llm.openai import (OpenAICompat, RETRYABLE_STATUS, _is_retryable,
                                _read_float_env, _read_int_env, _retry_delay)


class _Env:
    '''Set environment variables for the duration of a ``with`` block.'''

    def __init__(self, **values):
        '''
        Args:
            **values: Variable name mapped to its value; ``None`` removes the
                variable instead of setting it.
        '''
        self.values = values

    def __enter__(self):
        self.saved = {name: os.environ.get(name) for name in self.values}
        for name, value in self.values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return False


class _Transport:
    '''Stand in for ``WebBackend.post_stream_async`` while a test runs.

    Attributes:
        script (list): One entry per attempt, either a list of lines to yield or
            an exception instance to raise. The last entry repeats forever.
        calls (int): Number of attempts served so far.
    '''

    def __init__(self, script):
        '''
        Args:
            script (list): Per-attempt lines or exceptions, tried in order.
        '''
        self.script = list(script)
        self.calls = 0
        self._saved = None

    async def __call__(self, url, headers, json_data):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(step, Exception):
            raise step
        for line in step:
            if isinstance(line, Exception):
                raise line
            yield line

    def __enter__(self):
        self._saved = orchify_web_backend.__dict__.get('post_stream_async')
        orchify_web_backend.post_stream_async = self
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._saved is None:
            orchify_web_backend.__dict__.pop('post_stream_async', None)
        else:
            orchify_web_backend.post_stream_async = self._saved
        return False


def _status_error(code, body=None, text=None, headers=None):
    request = httpx.Request('POST', 'https://api.openai.com/v1/chat/completions')
    if text is not None:
        response = httpx.Response(code, request=request, text=text, headers=headers or {})
    else:
        response = httpx.Response(code, request=request, json=body if body is not None else {},
                                  headers=headers or {})
    return httpx.HTTPStatusError(f'HTTP Error {code}', request=request, response=response)


def _sse(payload):
    return 'data: ' + (payload if isinstance(payload, str) else json.dumps(payload))


class _BrokenBody:
    '''A response whose body cannot be turned into text, for the error-detail path.'''

    status_code = 502

    def json(self):
        raise ValueError('not json')

    @property
    def text(self):
        raise ValueError('no text either')


def _delta(**delta):
    return _sse({'choices': [{'delta': delta}]})


def _run(llm, lines, **kwargs):
    '''Drive ``_core_request`` with a scripted stream and collect every item.'''

    async def _lines(url, headers, data):
        for line in lines:
            yield line

    llm._resilient_lines = _lines

    async def _drain():
        return [item async for item in
                llm._core_request(messages=[{'role': 'user', 'content': 'hi'}],
                                  model='gpt-4o', **kwargs)]

    return asyncio.run(_drain())


def _labelled(responses):
    '''Map responses onto (event_type, content) pairs, with 'final'/'assembly' markers.'''
    out = []
    for response in responses:
        if response.is_final:
            out.append(('final', response.final_status.content))
        elif response.current_chunk.is_assembly_tool:
            out.append(('assembly', response.current_chunk.assembly_chunk))
        else:
            event = AgentEvent.build_from_resp(response, agent_name='a', agent_code='c',
                                               turn_id='t', turn=1)
            out.append((event.event_type, event.content))
    return out


def test_full_url_and_payload_flavours():
    '''_build_payload adapts to OpenAI reasoning, DeepSeek and generic models.'''
    llm = OpenAICompat(api_key='k', base_url='https://api.openai.com/v1')
    assert llm.full_url == 'https://api.openai.com/v1/chat/completions'

    base = llm._build_payload([{'role': 'user', 'content': 'hi'}], 'gpt-4o', None,
                              0.7, 1.0, False, None, 'auto', 'medium', None)
    assert base['stream'] is True
    assert base['stream_options'] == {'include_usage': True}
    assert base['temperature'] == 0.7 and base['top_p'] == 1.0
    assert 'tools' not in base and 'thinking' not in base and 'response_format' not in base

    enabled = llm._build_payload([], 'gpt-4o', [{'type': 'function'}], 0.7, 1.0, True,
                                 128, True, 'medium', None)
    assert enabled['tools'] == [{'type': 'function'}]
    assert enabled['response_format'] == {'type': 'json_object'}
    assert enabled['max_tokens'] == 128
    assert enabled['thinking'] == {'type': 'enabled'}

    disabled = llm._build_payload([], 'gpt-4o', None, 0.7, 1.0, False, None, 'off', 'medium', None)
    assert disabled['thinking'] == {'type': 'disabled'}

    on_true = llm._build_payload([], 'gpt-4o', None, 0.7, 1.0, False, None, 'on', 'medium', None)
    assert on_true['thinking'] == {'type': 'enabled'}

    reasoning = llm._build_payload([], 'o3-mini', None, 0.7, 1.0, False, None, 'enabled', 'xhigh', None)
    assert 'temperature' not in reasoning and 'top_p' not in reasoning
    assert reasoning['reasoning_effort'] == 'high'
    assert 'thinking' not in reasoning

    minimal = llm._build_payload([], 'o1-preview', None, 0.7, 1.0, False, None, 'enabled', 'minimal', None)
    assert minimal['reasoning_effort'] == 'low'

    reasoning_off = llm._build_payload([], 'o1-preview', None, 0.7, 1.0, False, None, 'disabled', 'high', None)
    assert 'reasoning_effort' not in reasoning_off

    deepseek = llm._build_payload([], 'deepseek-chat', None, 0.7, 1.0, False, None, 'auto', 'medium', None)
    assert deepseek['thinking'] == {'type': 'adaptive'}

    by_url = OpenAICompat(api_key='k', base_url='https://api.deepseek.com/v1')
    by_url_thinking = by_url._build_payload([], 'gpt-4o', None, 0.7, 1.0, False, None, 'enabled', 'medium', None)
    assert by_url_thinking['thinking'] == {'type': 'enabled'}

    merged = llm._build_payload([], 'gpt-4o', None, 0.7, 1.0, False, None, 'auto', 'medium',
                                {'model': 'override', 'extra': 1})
    assert merged['model'] == 'override' and merged['extra'] == 1


def test_core_request_content_and_usage():
    '''Content deltas accumulate, junk lines are skipped and usage is recorded.'''
    lines = [
        _delta(content='Hel'),
        '',
        _sse('not json at all'),
        'event: ping',
        _delta(content='lo'),
        _sse({'choices': [], 'usage': {'prompt_tokens': 30, 'completion_tokens': 5,
                                       'total_tokens': 35,
                                       'prompt_tokens_details': {'cached_tokens': 20}}}),
        _sse('[DONE]'),
        _delta(content='after done'),
    ]
    responses = _run(OpenAICompat(api_key='k'), lines)
    assert _labelled(responses) == [('agent:answer', 'Hel'), ('agent:answer', 'lo'), ('final', 'Hello')]
    assert [r.current_chunk.total_content for r in responses[:-1]] == ['Hel', 'Hello']
    final = responses[-1].final_status
    assert final.completion_tokens == 5
    assert final.prompt_tokens == 30
    assert final.total_tokens == 35
    assert final.prompt_cache_hit_tokens == 20
    assert final.prompt_cache_miss_tokens == 10


def test_core_request_usage_variants():
    '''Cache-miss tokens fall back to prompt minus hit, DeepSeek style or implied.'''
    deepseek_style = _sse({'usage': {'prompt_tokens': 100, 'completion_tokens': 2,
                                     'total_tokens': 102, 'prompt_cache_hit_tokens': 40,
                                     'prompt_cache_miss_tokens': 60}})
    final = _run(OpenAICompat(api_key='k'), [deepseek_style])[-1].final_status
    assert final.prompt_cache_hit_tokens == 40 and final.prompt_cache_miss_tokens == 60

    implied = _sse({'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11,
                              'prompt_tokens_details': {'cached_tokens': 4}}})
    final = _run(OpenAICompat(api_key='k'), [implied])[-1].final_status
    assert final.prompt_cache_hit_tokens == 4 and final.prompt_cache_miss_tokens == 6


def test_core_request_reasoning_events():
    '''A reasoning block ends with an agent:reason:finish chunk, not an empty answer.'''
    responses = _run(OpenAICompat(api_key='k'),
                     [_delta(reasoning_content='think'), _delta(reasoning='ing'),
                      _delta(content='answer')])
    assert _labelled(responses) == [('agent:reason:step', 'think'),
                                    ('agent:reason:step', 'ing'),
                                    ('agent:reason:finish', ''),
                                    ('agent:answer', 'answer'),
                                    ('final', 'answer')]
    final = responses[-1].final_status
    assert final.reasoning == 'thinking' and final.content == 'answer'

    only_reasoning = _run(OpenAICompat(api_key='k'), [_delta(reasoning_content='deep')])
    assert _labelled(only_reasoning) == [('agent:reason:step', 'deep'),
                                         ('agent:reason:finish', ''),
                                         ('final', '')]


def test_core_request_tool_call_assembly():
    '''Split tool-call deltas assemble into one call, and index-less deltas are ignored.'''
    split = [
        _sse({'choices': [{'delta': {'tool_calls': [
            {'index': 0, 'id': 'call_1', 'type': 'function',
             'function': {'name': 'get_weather', 'arguments': '{"ci'}}]}}]}),
        _sse({'choices': [{'delta': {'tool_calls': [
            {'index': 0, 'function': {'arguments': 'ty": "Tokyo"}'}}]}}]}),
        _sse({'choices': [{'delta': {'tool_calls': [{'function': {'name': 'ignored'}}]}}]}),
        _sse('[DONE]'),
    ]
    responses = _run(OpenAICompat(api_key='k'), split)
    assert _labelled(responses) == [('assembly', '{"ci'), ('assembly', 'ty": "Tokyo"}'),
                                    ('assembly', ''), ('final', '')]
    calls = responses[-1].final_status.tool_calls
    assert calls == [{'id': 'call_1', 'type': 'function',
                      'function': {'name': 'get_weather', 'arguments': '{"city": "Tokyo"}'}}]
    assert responses[0].current_chunk.total_tool_call[0]['function']['name'] == 'get_weather'


def test_core_request_wraps_http_errors():
    '''Provider failures are re-raised with the response body as extra detail.'''
    llm = OpenAICompat(api_key='k')

    async def _raise(exc):
        raise exc
        yield ''

    llm._resilient_lines = lambda url, headers, data: _raise(_status_error(429, body={'error': 'slow down'}))

    async def _drain():
        return [item async for item in
                llm._core_request(messages=[], model='gpt-4o')]

    try:
        asyncio.run(_drain())
        raise AssertionError('expected the provider error to propagate')
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 429
        assert "HTTP Error '429'" in str(exc)
        assert 'slow down' in str(exc)

    llm._resilient_lines = lambda url, headers, data: _raise(_status_error(500, text='upstream boom'))

    try:
        asyncio.run(_drain())
        raise AssertionError('expected the provider error to propagate')
    except httpx.HTTPStatusError as exc:
        assert 'Response detail: upstream boom' in str(exc)

    broken = _status_error(502)
    broken.response = _BrokenBody()

    llm._resilient_lines = lambda url, headers, data: _raise(broken)

    try:
        asyncio.run(_drain())
        raise AssertionError('expected the provider error to propagate')
    except httpx.HTTPStatusError as exc:
        assert "HTTP Error '502'" in str(exc) and 'Response detail' not in str(exc)


def test_resilient_lines_retry_matrix():
    '''Retries stop at the first streamed line, at a non-retryable status and at the cap.'''
    llm = OpenAICompat(api_key='k')

    async def _collect():
        return [line async for line in llm._resilient_lines('u', {}, {})]

    with _Env(ORCHIFY_LLM_RETRY_BACKOFF='0', ORCHIFY_LLM_MAX_RETRIES='2'):
        transport = _Transport([_status_error(503), ['data: one', 'data: two']])
        with transport:
            assert asyncio.run(_collect()) == ['data: one', 'data: two']
        assert transport.calls == 2

        transport = _Transport([_status_error(400)])
        with transport:
            try:
                asyncio.run(_collect())
                raise AssertionError('expected a 400 to surface')
            except httpx.HTTPStatusError as exc:
                assert exc.response.status_code == 400
        assert transport.calls == 1

        seen = []

        async def _consume():
            async for line in llm._resilient_lines('u', {}, {}):
                seen.append(line)

        transport = _Transport([['data: partial', _status_error(503)]])
        with transport:
            try:
                asyncio.run(_consume())
                raise AssertionError('expected a mid-stream failure')
            except httpx.HTTPStatusError as exc:
                assert exc.response.status_code == 503
        assert seen == ['data: partial'] and transport.calls == 1

        transport = _Transport([httpx.ConnectError('refused')])
        with transport:
            try:
                asyncio.run(_collect())
                raise AssertionError('expected the transport error to surface')
            except httpx.TransportError:
                pass
        assert transport.calls == 3

    with _Env(ORCHIFY_LLM_RETRY_BACKOFF='0', ORCHIFY_LLM_MAX_RETRIES='1'):
        transport = _Transport([_status_error(429, headers={'Retry-After': '0'})])
        with transport:
            try:
                asyncio.run(_collect())
                raise AssertionError('expected a 429 to surface at the retry cap')
            except httpx.HTTPStatusError as exc:
                assert exc.response.status_code == 429
        assert transport.calls == 2

    with _Env(ORCHIFY_LLM_MAX_RETRIES='0'):
        transport = _Transport([_status_error(503)])
        with transport:
            try:
                asyncio.run(_collect())
                raise AssertionError('expected no retry when the cap is zero')
            except httpx.HTTPStatusError:
                pass
        assert transport.calls == 1

    with _Env(ORCHIFY_LLM_RETRY_BACKOFF='0.001', ORCHIFY_LLM_MAX_RETRIES='1'):
        transport = _Transport([_status_error(503), ['data: ok']])
        with transport:
            assert asyncio.run(_collect()) == ['data: ok']
        assert transport.calls == 2


def test_env_readers_and_retry_delay():
    '''Backoff honours Retry-After, the env override and the 30s ceiling.'''
    with _Env(ORCHIFY_TEST_FLOAT=None):
        assert _read_float_env('ORCHIFY_TEST_FLOAT', 1.5) == 1.5
    with _Env(ORCHIFY_TEST_FLOAT='0.25'):
        assert _read_float_env('ORCHIFY_TEST_FLOAT', 1.5) == 0.25
    for bad in ('junk', '-3', '   '):
        with _Env(ORCHIFY_TEST_FLOAT=bad):
            assert _read_float_env('ORCHIFY_TEST_FLOAT', 1.5) == 1.5
    with _Env(ORCHIFY_TEST_INT='7'):
        assert _read_int_env('ORCHIFY_TEST_INT', 2) == 7
    with _Env(ORCHIFY_TEST_INT='junk'):
        assert _read_int_env('ORCHIFY_TEST_INT', 2) == 2

    assert RETRYABLE_STATUS >= {408, 429, 500, 502, 503, 504}
    assert _is_retryable(_status_error(503)) and _is_retryable(_status_error(429))
    assert not _is_retryable(_status_error(400)) and not _is_retryable(ValueError('nope'))
    assert _is_retryable(httpx.ReadTimeout('slow', request=httpx.Request('POST', 'https://x/y')))

    plain = ValueError('nope')
    with _Env(ORCHIFY_LLM_RETRY_BACKOFF='2'):
        assert _retry_delay(1, plain) == 4.0
        assert _retry_delay(10, plain) == 30.0
    assert _retry_delay(0, _status_error(429, headers={'Retry-After': '1.25'})) == 1.25
    assert _retry_delay(0, _status_error(429, headers={'Retry-After': '999'})) == 30.0
    assert _retry_delay(0, _status_error(429, headers={'Retry-After': 'oops'})) == 0.5


def main():
    '''Run every test in this module and report a summary.'''
    tests = [
        test_full_url_and_payload_flavours,
        test_core_request_content_and_usage,
        test_core_request_usage_variants,
        test_core_request_reasoning_events,
        test_core_request_tool_call_assembly,
        test_core_request_wraps_http_errors,
        test_resilient_lines_retry_matrix,
        test_env_readers_and_retry_delay,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            import traceback
            failures.append(fn.__name__)
            print(f'FAIL {fn.__name__}: {exc}')
            traceback.print_exc()
    print()
    if failures:
        print(f'{len(failures)} failed: {failures}')
        return 1
    print(f'all {len(tests)} tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
