"""Summarizing context plugin: recap old turns instead of dropping them."""

import asyncio

from orchify import Agent, orchify_plugins
from orchify.plugins.summarize import Summarizer, SummarizingContextPlugin
from orchify.testing import FakeLLM, final


def _conversation(turns=8, size=400):
    '''A long conversation with a system prompt and tool traffic.'''
    messages = [{'role': 'system', 'content': 'You are a careful assistant.'}]
    for index in range(turns):
        messages.append({'role': 'user', 'content': ('question %d ' % index) + 'q' * size})
        messages.append({
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call-%d' % index,
                'type': 'function',
                'function': {'name': 'lookup', 'arguments': '{"q": "x"}'},
            }],
        })
        messages.append({'role': 'tool', 'tool_call_id': 'call-%d' % index,
                         'content': ('result %d ' % index) + 'r' * size})
        messages.append({'role': 'assistant', 'content': ('answer %d ' % index) + 'a' * size})
    return messages


def _plugin(**kwargs):
    '''Plugin whose summarizer is a FakeLLM returning a fixed recap.'''
    summarizer = kwargs.pop('summarizer', None)
    if summarizer is None:
        summarizer = Summarizer(llm=FakeLLM([[final('RECAP: earlier work was done.')]]))
    options = {'max_prompt_tokens': 1200, 'reserve_tokens': 0, 'min_savings': 10}
    options.update(kwargs)
    return SummarizingContextPlugin(summarizer=summarizer, **options)


def test_compact_replaces_old_turns_with_a_recap():
    '''Old turns become one recap message and recent turns stay intact.'''
    plugin = _plugin()
    messages = _conversation()
    result = plugin.compact(messages)

    assert result.changed is True
    assert result.dropped > 0
    assert result.after < result.before
    recap = result.messages[1]
    assert recap['role'] == 'user'
    assert 'RECAP: earlier work was done.' in recap['content']
    assert 'summary of %d earlier message(s)' % result.dropped in recap['content']
    assert result.messages[0]['role'] == 'system'
    # The newest turn is still there, verbatim.
    assert result.messages[-1]['content'].startswith('answer 7')
    assert plugin.stats['summarized'] == 1
    assert plugin.stats['summarized_messages'] == result.dropped


def test_compact_keeps_a_tool_call_with_its_result():
    '''The recap never separates an assistant tool call from its tool result.'''
    plugin = _plugin()
    result = plugin.compact(_conversation())
    kept = result.messages

    for position, message in enumerate(kept):
        if message.get('role') == 'assistant' and message.get('tool_calls'):
            following = kept[position + 1]
            assert following.get('role') == 'tool'
            assert following['tool_call_id'] == message['tool_calls'][0]['id']


def test_compact_is_a_no_op_within_budget():
    '''A conversation that fits is returned untouched.'''
    plugin = _plugin(max_prompt_tokens=100000)
    messages = _conversation(turns=2)
    result = plugin.compact(messages)

    assert result.changed is False
    assert result.messages == messages
    assert plugin.stats['summarized'] == 0


def test_a_failing_summarizer_falls_back_to_the_marker():
    '''A provider error degrades to a marker recap instead of breaking the run.'''

    class Boom(FakeLLM):
        '''Provider that always fails.'''

        async def _core_request(self, **kwargs):
            raise RuntimeError('summary provider down')
            yield  # pragma: no cover - makes this an async generator

    plugin = _plugin(summarizer=Summarizer(llm=Boom([[final('unused')]])))
    result = plugin.compact(_conversation())

    assert result.changed is True
    assert result.messages[1]['content'].startswith('[summary of ')
    assert 'summary provider down' not in result.messages[1]['content']
    assert plugin.stats['fallbacks'] == 1


def test_summarizer_renders_tool_calls_for_the_prompt():
    '''Tool calls and results reach the summarizer as readable text.'''
    summarizer = Summarizer(llm=None)
    text = summarizer.render([
        {'role': 'user', 'content': 'find it'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'c1', 'type': 'function',
             'function': {'name': 'search', 'arguments': '{"q": "cats"}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'nine lives'},
    ])
    assert 'user: find it' in text
    assert 'calls: search({"q": "cats"})' in text
    assert 'tool: nine lives' in text


def test_plugin_runs_as_a_middleware_on_an_agent():
    '''Agent(plugins=[...]) installs the middleware and the prompt is condensed.'''
    summarizer_llm = FakeLLM([[final('RECAP from the middleware.')]])
    plugin = SummarizingContextPlugin(summarizer=Summarizer(llm=summarizer_llm),
                                      max_prompt_tokens=1200, reserve_tokens=0,
                                      min_savings=10)
    orchify_plugins.register(plugin)
    try:
        agent_llm = FakeLLM([[final('done')]])
        agent = Agent(name='summary-run', llm=agent_llm, plugins=['summarizing_context'])
        for index in range(6):
            agent.messages.append({'role': 'user', 'content': 'q%d ' % index + 'x' * 400})
            agent.messages.append({'role': 'assistant', 'content': 'a%d ' % index + 'y' * 400})
        assert agent.ask('next') == 'done'

        sent = agent_llm.calls[-1]['messages']
        assert any('RECAP from the middleware.' in str(m.get('content')) for m in sent)
        assert plugin.stats['summarized'] >= 1
    finally:
        orchify_plugins.unregister('summarizing_context')


def test_atrim_reports_and_updates_the_request():
    '''The async path replaces messages in place and counts the request.'''
    plugin = _plugin()
    request = {'messages': _conversation(), 'tools': None, 'run_id': ''}
    result = asyncio.run(plugin.atrim(request))

    assert result.changed is True
    assert request['messages'] is result.messages
    assert plugin.stats['requests'] == 1
    assert plugin.stats['last']['changed'] is True
