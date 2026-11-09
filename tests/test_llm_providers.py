"""Tests for the native Anthropic/Gemini adapters and the fallback chain.

The adapters talk to the network only through
``orchify_web_backend.post_stream_async``, so every test here replaces that one
method with a canned SSE stream and checks the translation in both directions:
no socket, no credentials, no provider.
"""

import asyncio
import json

import httpx
import pytest

from orchify.backend import orchify_web_backend
from orchify.llm.anthropic import (
    AnthropicLLM,
    to_anthropic_messages,
    to_anthropic_tools,
)
from orchify.llm.base import Chunk, Response
from orchify.llm.fallback import FallbackLLM
from orchify.llm.gemini import GeminiLLM, to_gemini_contents, to_gemini_tools
from orchify.llm.retry import MAX_RETRY_DELAY, is_retryable, retry_delay
from orchify.testing import FakeLLM, final


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def collect(llm, **kwargs):
    '''Run ``llm._core_request`` to completion and return every response.'''
    async def run():
        responses = []
        async for response in llm._core_request(**kwargs):
            responses.append(response)
        return responses

    return asyncio.run(run())


def patch_stream(monkeypatch, lines):
    '''Replace the SSE transport with a canned list of lines.'''
    seen = []

    async def fake(url, headers, payload):
        seen.append({'url': url, 'headers': headers, 'payload': payload})
        for line in lines:
            yield line

    monkeypatch.setattr(orchify_web_backend, 'post_stream_async', fake)
    return seen


def anthropic_lines(events):
    '''Render Anthropic events as an SSE stream (event: + data: lines).'''
    lines = []
    for event in events:
        lines.append('event: %s' % event.get('type'))
        lines.append('data: %s' % json.dumps(event))
        lines.append('')
    return lines


def gemini_lines(events):
    '''Render Gemini events as an SSE stream.'''
    return ['data: %s' % json.dumps(event) for event in events]


def text_of(responses):
    '''The cumulative answer text after each streamed content chunk.'''
    return [r.current_chunk.total_content for r in responses
            if r.current_chunk is not None and r.current_chunk.content
            and not r.current_chunk.is_cot]


def cot_of(responses):
    '''The cumulative reasoning text after each streamed thinking chunk.'''
    return [r.current_chunk.total_cot_content for r in responses
            if r.current_chunk is not None and r.current_chunk.is_cot]


def assembly_of(responses):
    '''The streamed tool-argument fragments of every response.'''
    return [r.current_chunk.assembly_chunk for r in responses
            if r.current_chunk is not None and r.current_chunk.is_assembly_tool]


# --------------------------------------------------------------------------- #
# retry helpers
# --------------------------------------------------------------------------- #

def test_retry_helpers_classify_and_back_off():
    request = httpx.Request('POST', 'https://example.test/v1/messages')
    rate_limited = httpx.HTTPStatusError('boom', request=request,
                                         response=httpx.Response(429, request=request))
    not_found = httpx.HTTPStatusError('boom', request=request,
                                      response=httpx.Response(404, request=request))
    assert is_retryable(rate_limited) is True
    assert is_retryable(not_found) is False
    assert is_retryable(httpx.ConnectError('down')) is True
    assert is_retryable(ValueError('bad')) is False
    assert retry_delay(0, 0.5) == 0.5
    assert retry_delay(2, 0.5) == 2.0
    assert retry_delay(20, 0.5) == MAX_RETRY_DELAY
    assert retry_delay(0, 0.0) == 0.0


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #

def test_anthropic_message_conversion_keeps_tool_blocks_together():
    messages = [
        {'role': 'system', 'content': 'be nice'},
        {'role': 'user', 'content': 'what is the weather?'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'call_1', 'type': 'function',
            'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'},
        }]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'sunny'},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': '22C'},
    ]
    system, converted = to_anthropic_messages(messages)

    assert system == 'be nice'
    assert converted[0] == {'role': 'user', 'content': [{'type': 'text', 'text': 'what is the weather?'}]}
    assert converted[1]['content'][0] == {
        'type': 'tool_use', 'id': 'call_1', 'name': 'get_weather', 'input': {'city': 'Paris'},
    }
    # Both tool results are merged into one user message, as the API requires.
    assert len(converted) == 3
    assert [block['content'] for block in converted[2]['content']] == ['sunny', '22C']
    assert to_anthropic_tools([{'type': 'function', 'function': {
        'name': 'get_weather', 'description': 'weather', 'parameters': {'type': 'object'},
    }}]) == [{'name': 'get_weather', 'description': 'weather', 'input_schema': {'type': 'object'}}]


def test_anthropic_stream_translates_text_usage_and_finish_reason(monkeypatch):
    events = [
        {'type': 'message_start', 'message': {'usage': {
            'input_tokens': 12, 'output_tokens': 1, 'cache_read_input_tokens': 5,
        }}},
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hel'}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'lo'}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 3}},
        {'type': 'message_stop'},
    ]
    seen = patch_stream(monkeypatch, anthropic_lines(events))
    llm = AnthropicLLM(api_key='sk-test', model='claude-test')
    responses = collect(llm, messages=[{'role': 'user', 'content': 'hi'}])

    assert text_of(responses) == ['Hel', 'Hello']
    assert seen[0]['url'] == 'https://api.anthropic.com/v1/messages'
    assert seen[0]['headers']['x-api-key'] == 'sk-test'
    assert seen[0]['payload']['model'] == 'claude-test'
    final_status = responses[-1].final_status
    assert final_status.content == 'Hello'
    assert final_status.finish_reason == 'stop'
    assert final_status.prompt_tokens == 12
    assert final_status.completion_tokens == 3
    assert final_status.prompt_cache_hit_tokens == 5
    assert final_status.prompt_cache_miss_tokens == 7


def test_anthropic_stream_assembles_tool_use_arguments(monkeypatch):
    events = [
        {'type': 'message_start', 'message': {'usage': {'input_tokens': 30}}},
        {'type': 'content_block_start', 'index': 0, 'content_block': {
            'type': 'tool_use', 'id': 'toolu_1', 'name': 'get_weather', 'input': {},
        }},
        {'type': 'content_block_delta', 'index': 0,
         'delta': {'type': 'input_json_delta', 'partial_json': '{"city":'}},
        {'type': 'content_block_delta', 'index': 0,
         'delta': {'type': 'input_json_delta', 'partial_json': '"Paris"}'}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'}, 'usage': {'output_tokens': 9}},
    ]
    patch_stream(monkeypatch, anthropic_lines(events))
    llm = AnthropicLLM(api_key='sk-test')
    responses = collect(llm, messages=[{'role': 'user', 'content': 'weather?'}])

    assert assembly_of(responses) == ['{"city":', '"Paris"}']
    final_status = responses[-1].final_status
    assert final_status.finish_reason == 'tool_calls'
    assert final_status.tool_calls == [{
        'id': 'toolu_1', 'type': 'function',
        'function': {'name': 'get_weather', 'arguments': '{"city":"Paris"}'},
    }]


def test_anthropic_stream_separates_thinking_from_the_answer(monkeypatch):
    events = [
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'thinking', 'thinking': ''}},
        {'type': 'content_block_delta', 'index': 0,
         'delta': {'type': 'thinking_delta', 'thinking': 'let me think'}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': 'answer'}},
        {'type': 'content_block_stop', 'index': 1},
    ]
    patch_stream(monkeypatch, anthropic_lines(events))
    llm = AnthropicLLM(api_key='sk-test')
    responses = collect(llm, messages=[{'role': 'user', 'content': 'q'}])

    cot = [r.current_chunk.content for r in responses
           if r.current_chunk is not None and r.current_chunk.is_cot]
    assert cot == ['let me think']
    assert cot_of(responses) == ['let me think']
    assert any(r.current_chunk is not None and r.current_chunk.is_cot_end for r in responses)
    final_status = responses[-1].final_status
    assert final_status.reasoning == 'let me think'
    assert final_status.content == 'answer'


def test_anthropic_reports_unusable_stream_errors(monkeypatch):
    patch_stream(monkeypatch, anthropic_lines([
        {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'try later'}},
    ]))
    llm = AnthropicLLM(api_key='sk-test')
    with pytest.raises(RuntimeError, match='overloaded_error'):
        collect(llm, messages=[{'role': 'user', 'content': 'q'}])


def test_anthropic_without_a_key_explains_the_fix(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    llm = AnthropicLLM()
    assert 'ANTHROPIC_API_KEY' in llm.missing_credentials()
    with pytest.raises(RuntimeError, match='ANTHROPIC_API_KEY'):
        collect(llm, messages=[{'role': 'user', 'content': 'q'}])


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #

def test_gemini_conversion_maps_tool_calls_and_responses():
    messages = [
        {'role': 'system', 'content': 'be nice'},
        {'role': 'user', 'content': 'weather?'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'call_1', 'type': 'function',
            'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'},
        }]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'sunny'},
    ]
    system, contents = to_gemini_contents(messages)

    assert system == 'be nice'
    assert contents[1]['role'] == 'model'
    assert contents[1]['parts'][0]['functionCall'] == {'name': 'get_weather', 'args': {'city': 'Paris'}}
    assert contents[2]['role'] == 'user'
    assert contents[2]['parts'][0]['functionResponse'] == {
        'name': 'get_weather', 'response': {'result': 'sunny'},
    }
    assert to_gemini_tools([{'type': 'function', 'function': {
        'name': 'get_weather', 'description': 'weather', 'parameters': {'type': 'object'},
    }}]) == [{'functionDeclarations': [{
        'name': 'get_weather', 'description': 'weather', 'parameters': {'type': 'object'},
    }]}]


def test_gemini_stream_translates_text_and_usage(monkeypatch):
    events = [
        {'candidates': [{'content': {'parts': [{'text': 'Hi'}], 'role': 'model'}}]},
        {'candidates': [{'content': {'parts': [{'text': ' there'}], 'role': 'model'},
                         'finishReason': 'STOP'}],
         'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 4,
                           'cachedContentTokenCount': 2, 'totalTokenCount': 14}},
    ]
    seen = patch_stream(monkeypatch, gemini_lines(events))
    llm = GeminiLLM(api_key='gm-test', model='gemini-test')
    responses = collect(llm, messages=[{'role': 'user', 'content': 'hi'}])

    assert text_of(responses) == ['Hi', 'Hi there']
    assert seen[0]['url'] == ('https://generativelanguage.googleapis.com'
                              '/v1beta/models/gemini-test:streamGenerateContent?alt=sse')
    assert seen[0]['headers']['x-goog-api-key'] == 'gm-test'
    final_status = responses[-1].final_status
    assert final_status.content == 'Hi there'
    assert final_status.finish_reason == 'stop'
    assert final_status.prompt_tokens == 10
    assert final_status.completion_tokens == 4
    assert final_status.prompt_cache_hit_tokens == 2
    assert final_status.prompt_cache_miss_tokens == 8
    assert final_status.total_tokens == 14


def test_gemini_stream_maps_function_calls_and_thoughts(monkeypatch):
    events = [
        {'candidates': [{'content': {'parts': [{'text': 'thinking', 'thought': True}],
                                     'role': 'model'}}]},
        {'candidates': [{'content': {'parts': [{'functionCall': {
            'name': 'get_weather', 'args': {'city': 'Paris'},
        }}], 'role': 'model'}, 'finishReason': 'STOP'}]},
    ]
    patch_stream(monkeypatch, gemini_lines(events))
    llm = GeminiLLM(api_key='gm-test')
    responses = collect(llm, messages=[{'role': 'user', 'content': 'weather?'}])

    cot = [r.current_chunk.content for r in responses
           if r.current_chunk is not None and r.current_chunk.is_cot]
    assert cot == ['thinking']
    assert assembly_of(responses) == ['{"city": "Paris"}']
    final_status = responses[-1].final_status
    assert final_status.reasoning == 'thinking'
    assert final_status.tool_calls == [{
        'id': 'call_1', 'type': 'function',
        'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'},
    }]


# --------------------------------------------------------------------------- #
# FallbackLLM
# --------------------------------------------------------------------------- #

class FlakyLLM(FakeLLM):
    '''A FakeLLM that fails a configurable number of times before answering.'''

    def __init__(self, script=None, error=None, fail_times=1):
        super().__init__(script or [[final('recovered')]])
        self.error = error if error is not None else httpx.ConnectError('network down')
        self.fail_times = fail_times
        self.attempts = 0

    async def _core_request(self, messages=None, model=None, tools=None, **kwargs):
        self.attempts += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        async for response in super()._core_request(messages=messages, model=model,
                                                    tools=tools, **kwargs):
            yield response


class PartialLLM(FakeLLM):
    '''A FakeLLM that emits one chunk and then loses the connection.'''

    def __init__(self):
        super().__init__([[final('never')]])
        self.attempts = 0

    async def _core_request(self, messages=None, model=None, tools=None, **kwargs):
        self.attempts += 1
        yield Response(current_chunk=Chunk(is_cot=False, content='partial', total_content='partial'),
                       is_final=False)
        raise httpx.ConnectError('connection reset mid-stream')


class KeylessLLM(FakeLLM):
    '''A FakeLLM that reports missing credentials.'''

    def missing_credentials(self):
        return 'no key for this provider'


def _ask(llm, text='hi'):
    responses = collect(llm, messages=[{'role': 'user', 'content': text}])
    return responses[-1].final_status.content if responses else ''


def test_fallback_switches_after_a_transient_failure_and_cools_down():
    primary = FlakyLLM([[final('never')]], fail_times=1)
    secondary = FakeLLM([[final('backup answer')]])
    llm = FallbackLLM([primary, secondary], max_failures=1, cooldown=60.0)

    assert _ask(llm) == 'backup answer'
    state = llm.snapshot()
    assert primary.attempts == 1
    assert state['switches'] == 1
    assert state['last_provider'] == 'FakeLLM'
    assert state['cooldown_left'][0] > 0
    assert state['available'] == [1]

    # While cooling down the failing provider is not touched at all.
    assert _ask(llm) == 'backup answer'
    assert primary.attempts == 1
    assert llm.snapshot()['attempts'] == 3


def test_fallback_counts_failures_across_requests_before_cooling_down():
    primary = FlakyLLM([[final('eventually')]], fail_times=2)
    llm = FallbackLLM([primary], max_failures=2, cooldown=30.0)

    # One failure per request: the provider is only cooled down on the second.
    for _ in range(2):
        with pytest.raises(httpx.ConnectError, match='network down'):
            _ask(llm)
    assert llm.snapshot()['cooldown_left'][0] > 0

    # Cooling down does not remove the provider: a lone provider is still tried.
    assert _ask(llm) == 'eventually'
    state = llm.snapshot()
    assert primary.attempts == 3
    assert state['failures'] == [0]
    assert state['cooldown_left'][0] == 0.0


def test_fallback_does_not_retry_a_permanent_error():
    primary = FlakyLLM(error=ValueError('bad request'))
    secondary = FakeLLM([[final('never')]])
    llm = FallbackLLM([primary, secondary])

    with pytest.raises(ValueError, match='bad request'):
        _ask(llm)
    assert len(secondary.calls) == 0
    assert llm.snapshot()['cooldown_left'][0] == 0.0


def test_fallback_refuses_to_replay_a_partial_answer():
    primary = PartialLLM()
    secondary = FakeLLM([[final('never')]])
    llm = FallbackLLM([primary, secondary])

    with pytest.raises(httpx.ConnectError, match='mid-stream'):
        _ask(llm)
    assert len(secondary.calls) == 0


def test_fallback_skips_a_provider_without_credentials():
    keyless = KeylessLLM([[final('never')]])
    secondary = FakeLLM([[final('served')]])
    llm = FallbackLLM([keyless, secondary])

    assert _ask(llm) == 'served'
    assert len(keyless.calls) == 0
    assert llm.snapshot()['attempts'] == 1
    assert llm.snapshot()['cooldown_left'][0] == float('inf')


def test_fallback_reports_when_nobody_can_answer():
    keyless = KeylessLLM([[final('never')]])
    llm = FallbackLLM([keyless])

    with pytest.raises(RuntimeError, match='No LLM provider is available'):
        _ask(llm)


def test_fallback_reset_clears_cooldowns():
    primary = FlakyLLM([[final('ok again')]], fail_times=1)
    llm = FallbackLLM([primary], max_failures=1, cooldown=600.0)

    with pytest.raises(httpx.ConnectError, match='network down'):
        _ask(llm)
    assert llm.snapshot()['cooldown_left'][0] > 0

    llm.reset()
    assert llm.snapshot()['cooldown_left'][0] == 0.0
    assert _ask(llm) == 'ok again'


