"""Tool results: truncation and offloading of oversized outputs."""

import json
import os
import shutil
import uuid

from orchify import Agent, tool
from orchify.testing import FakeLLM, final, tool_call


def _directory():
    '''A writable scratch directory inside the test tree.'''
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data',
                        'results-%s' % uuid.uuid4().hex[:8])
    os.makedirs(path, exist_ok=True)
    return path


def _drop(directory):
    '''Remove a scratch directory, ignoring a locked leftover.'''
    shutil.rmtree(directory, ignore_errors=True)


_LONG = 'x' * 500 + 'THE MIDDLE' + 'y' * 500


def _plan():
    '''One tool call followed by a final answer.'''
    first = {'content': '',
             'final': {'content': 'working', 'reasoning': '',
                       'tool_calls': [tool_call('read_file', '{"path": "big.txt"}')]}}
    second = {'final': {'content': 'done', 'reasoning': '', 'tool_calls': []}}
    return [first, second]


def _agent(**kwargs):
    '''Agent with a tool that returns a long document.'''

    @tool()
    def read_file(path: str) -> str:
        '''Read a file.

        Args:
            path (str): Path to read.

        Returns:
            str: File contents.
        '''
        return _LONG

    return Agent(name='tools', llm=FakeLLM([[_plan()[0]], [_plan()[1]]]),
                 tools=[read_file], **kwargs)


def test_long_tool_results_are_shortened_for_the_model():
    '''The tool message keeps a head, a tail and a size marker.'''
    agent = _agent(tool_result_limit=120)
    assert agent.ask('read it', max_steps=2) == 'done'

    messages = agent.messages
    tool_message = [m for m in messages if m.get('role') == 'tool'][-1]
    content = tool_message['content']
    assert len(content) < 400
    assert 'result truncated' in content
    assert '%d characters in total' % len(_LONG) in content
    assert content.startswith('[read_file result truncated')
    assert 'y' * 20 in content


def test_truncation_offloads_the_full_text_to_disk():
    '''``tool_result_dir`` keeps the complete output and names it for the model.'''
    directory = _directory()
    try:
        agent = _agent(tool_result_limit=80, tool_result_dir=directory)
        assert agent.ask('read it', max_steps=2) == 'done'
        tool_message = [m for m in agent.messages if m.get('role') == 'tool'][-1]
        saved = os.path.join(directory, os.listdir(directory)[0])
        assert saved in tool_message['content']
        with open(saved, 'r', encoding='utf-8') as handle:
            assert handle.read() == _LONG
    finally:
        _drop(directory)


def test_truncation_is_reported_in_the_event_payload():
    '''The tool event says how much was dropped and where the rest went.'''
    directory = _directory()
    try:
        events = []

        def keep(event):
            if getattr(event, 'event_type', '') == 'tool:call:finish':
                events.append(event)

        from orchify import orchify_broker
        binding = orchify_broker.register_hook('tool:call:finish', keep)
        try:
            agent = _agent(tool_result_limit=50, tool_result_dir=directory)
            assert agent.ask('read it', max_steps=2) == 'done'
        finally:
            orchify_broker.remove_binding(binding)

        truncated = [event for event in events if event.payload.get('truncated')]
        assert truncated
        payload = truncated[0].payload
        assert payload['original_chars'] == len(_LONG)
        assert os.path.exists(payload['path'])
        assert 'truncated' in truncated[0].payload['result']
    finally:
        _drop(directory)


def test_results_are_untouched_by_default():
    '''Without a limit the tool message is the exact tool output.'''
    agent = _agent()
    assert agent.ask('read it', max_steps=2) == 'done'
    tool_message = [m for m in agent.messages if m.get('role') == 'tool'][-1]
    assert tool_message['content'] == _LONG
