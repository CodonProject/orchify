"""RecordLLM / ReplayLLM: capture a session once, replay it for free."""

import os
import uuid

import pytest

from orchify import Agent, tool
from orchify.llm import RecordLLM, ReplayLLM, ReplayError
from orchify.testing import FakeLLM, final, tool_call


def _directory():
    '''A writable scratch directory inside the test tree.'''
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data',
                        'record-%s' % uuid.uuid4().hex[:8])
    os.makedirs(path, exist_ok=True)
    return path


def _final_with_usage(content='recorded answer'):
    '''A finishing step that also reports token counters.'''
    return {'final': {'content': content, 'reasoning': '', 'tool_calls': [],
                      'prompt_tokens': 100, 'completion_tokens': 20,
                      'total_tokens': 120,
                      'prompt_cache_hit_tokens': 10,
                      'prompt_cache_miss_tokens': 90}}


def test_record_then_replay_reproduces_the_answer():
    '''A recorded run replays the same content without touching the provider.'''
    directory = _directory()
    try:
        inner = FakeLLM([[_final_with_usage()]])
        recorder = RecordLLM(inner)
        agent = Agent(name='rec', llm=recorder)
        answer = agent.ask('hello')
        assert answer == 'recorded answer'
        assert len(recorder.calls) == 1
        request = recorder.calls[0]['request']
        assert request['messages'][-1]['content'] == 'hello'
        path = recorder.save(os.path.join(directory, 'session.json'))
        assert os.path.exists(path)

        replay = ReplayLLM.from_file(path)
        replayed = Agent(name='rec', llm=replay).ask('hello')
        assert replayed == 'recorded answer'
        assert replay.position == 1 and replay.remaining == 0
        assert inner.calls  # the first provider saw the request

        second_inner = FakeLLM([[_final_with_usage('other')]])
        second_replay = ReplayLLM.from_file(path)
        second_agent = Agent(name='rec', llm=second_replay)
        assert second_agent.ask('hello') == 'recorded answer'
        del second_inner
    finally:
        _drop(directory)


def _collect(llm, text='hello'):
    '''Drive an LLM directly (the agent swallows provider errors).'''
    import asyncio

    async def consume():
        messages = [{'role': 'user', 'content': text}]
        return [response async for response in llm.request(messages=messages)]

    return asyncio.run(consume())


def test_replay_strict_mode_detects_a_changed_prompt():
    '''Strict replay refuses a request whose messages differ.'''
    directory = _directory()
    try:
        recorder = RecordLLM(FakeLLM([[_final_with_usage()]]))
        _collect(recorder, 'hello')
        path = recorder.save(os.path.join(directory, 'session.json'))
        replay = ReplayLLM.from_file(path, strict=True)
        with pytest.raises(ReplayError, match='differs from the recording'):
            _collect(replay, 'goodbye')
    finally:
        _drop(directory)


def test_replay_raises_when_the_recording_runs_out():
    '''An empty recording cannot answer anything.'''
    empty = ReplayLLM([])
    with pytest.raises(ReplayError, match='exhausted'):
        _collect(empty)


def test_missing_recording_file_is_reported():
    '''A bad path raises the same error type as an exhausted recording.'''
    with pytest.raises(ReplayError, match='recording not found'):
        ReplayLLM.from_file('does/not/exist.json')


def test_replay_reproduces_tool_calls():
    '''Streamed chunks and the final tool call survive the round trip.'''
    directory = _directory()
    calls = []

    @tool()
    def echo(text: str) -> str:
        '''Echo the text back.

        Args:
            text (str): Text to echo.

        Returns:
            str: The same text.
        '''
        calls.append(text)
        return text

    plan = [{'content': 'let me check',
             'final': {'content': 'done after tool', 'reasoning': '',
                       'tool_calls': [tool_call('echo', '{"text": "ping"}')]}}]
    plan.append(_final_with_usage('done after tool'))
    try:
        recorder = RecordLLM(FakeLLM([plan[0:1], [plan[1]]]))
        first = Agent(name='rec', llm=recorder, tools=[echo])
        assert first.ask('use the tool') == 'done after tool'
        assert calls == ['ping']
        path = recorder.save(os.path.join(directory, 'tool.json'))

        replay = ReplayLLM.from_file(path)
        calls.clear()
        second = Agent(name='rec', llm=replay, tools=[echo])
        assert second.ask('use the tool') == 'done after tool'
        assert calls == ['ping']
        assert len(replay.requests) == 2
    finally:
        _drop(directory)


def test_recording_round_trips_through_as_dict():
    '''``as_dict`` is JSON-able and ``ReplayLLM`` accepts it directly.'''
    recorder = RecordLLM(FakeLLM([[_final_with_usage()]]))
    Agent(name='rec', llm=recorder).ask('hello')
    document = recorder.as_dict()
    assert document['version'] == 1
    replay = ReplayLLM(document['calls'])
    assert Agent(name='rec', llm=replay).ask('hello') == 'recorded answer'


def test_recording_delegates_provider_attributes():
    '''Model and other provider attributes stay reachable through the wrapper.'''
    inner = FakeLLM([[_final_with_usage()]])
    inner.model = 'fake-model'
    recorder = RecordLLM(inner)
    assert recorder.model == 'fake-model'
    with pytest.raises(AttributeError):
        recorder.no_such_attribute


def _drop(directory):
    '''Remove a scratch directory, ignoring a locked leftover.'''
    import shutil
    shutil.rmtree(directory, ignore_errors=True)
