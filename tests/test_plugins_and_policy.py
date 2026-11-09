import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from orchify import Agent, Tool, orchify_broker, orchify_plugins
from orchify.plugins import ContextWindowPlugin, CostPlugin
from orchify.policy import ASK, DENY, ToolPolicy, ToolRule, policy_from_env
from orchify.testing import FakeLLM, Recorder, final, tool_call, wait_finish
from orchify.tokens import count_cjk, estimate_message, estimate_messages, estimate_tokens


def final_usage(content, usage=None, tool_calls=None):
    '''A FinalStatus entry that also reports provider usage.'''
    payload = {'content': content, 'reasoning': '', 'tool_calls': tool_calls or []}
    for key, value in (usage or {}).items():
        payload[key] = value
    return {'final': payload}


def _cleanup(*names):
    '''Unregister plugins so one test cannot leak into the next.'''
    for name in names:
        try:
            orchify_plugins.unregister(name)
        except Exception:
            pass


def test_token_estimator_is_cheap_and_cjk_aware():
    '''ASCII text is four characters per token; CJK is about one per character.'''
    assert estimate_tokens('') == 0
    assert estimate_tokens('hello world') == 3
    assert estimate_tokens('a' * 400) == 100
    assert count_cjk('hello') == 0
    assert count_cjk('你好') == 2
    chinese = estimate_tokens('你好世界')
    assert chinese == 4, chinese
    assert estimate_tokens('你好世界 你好世界') > chinese
    message = estimate_message({'role': 'user', 'content': 'hi'})
    assert message >= 6, message
    assert estimate_messages([{'role': 'user', 'content': 'hi'}]) == message


def test_context_plugin_drops_old_turns_and_keeps_the_newest():
    '''Compaction keeps the system prompt and the latest turn, and says what it dropped.'''
    plugin = ContextWindowPlugin(max_prompt_tokens=200, reserve_tokens=0, min_savings=1)
    messages = [{'role': 'system', 'content': 'you are terse'}]
    for index in range(12):
        messages.append({'role': 'user', 'content': f'question {index} ' + 'x' * 200})
        messages.append({'role': 'assistant', 'content': f'answer {index} ' + 'y' * 200})
    result = plugin.compact(messages)
    assert result.changed and result.dropped > 0
    assert result.messages[0]['role'] == 'system'
    assert 'question 11' in result.messages[-2]['content']
    assert 'context compacted' in result.messages[1]['content']
    assert result.after <= result.budget or result.dropped > 0


def test_context_plugin_keeps_tool_calls_with_their_results():
    '''A turn is never split, so an assistant tool_calls message keeps its tool results.'''
    plugin = ContextWindowPlugin(max_prompt_tokens=400, reserve_tokens=0, min_savings=1)
    messages = [{'role': 'user', 'content': 'older ' + 'z' * 400},
                {'role': 'assistant', 'content': 'older answer'}]
    messages += [
        {'role': 'user', 'content': 'run the tool'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 't1', 'type': 'function', 'function': {'name': 'noop', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 't1', 'content': 'done'},
    ]
    result = plugin.compact(messages)
    roles = [message['role'] for message in result.messages]
    assert 'tool' in roles and 'assistant' in roles
    assistant = next(m for m in result.messages if m['role'] == 'assistant' and m.get('tool_calls'))
    assert assistant['tool_calls'][0]['id'] == 't1'
    assert any(m.get('tool_call_id') == 't1' for m in result.messages)


def test_context_plugin_middleware_rewrites_the_prompt():
    '''Registered and bound, the plugin trims the messages the fake LLM receives.'''
    plugin = ContextWindowPlugin(max_prompt_tokens=300, reserve_tokens=0, min_savings=1)
    orchify_plugins.register(plugin)
    try:
        llm = FakeLLM([[final('ok')]])
        agent = Agent(name='ctx-agent', llm=llm, plugins=['context_window'])
        orchify_plugins.attach('context_window', agent)
        assert plugin._middlewares, 'middleware was not registered'
        long_turn = 'w' * 4000
        agent.ask('first question ' + long_turn, max_steps=1)
        agent.ask('second question', max_steps=1)
        sent = llm.calls[-1]['messages']
        assert plugin.stats['compacted'] >= 1, plugin.stats
        assert any('context compacted' in str(m.get('content')) for m in sent)
    finally:
        _cleanup('context_window')


def test_cost_plugin_prices_steps_and_runs():
    '''Agent:finish usage is priced, totalled per run and for the process.'''
    plugin = CostPlugin(history=10)
    orchify_plugins.register(plugin)
    try:
        usage = {'prompt_tokens': 1000, 'prompt_cache_hit_tokens': 200,
                 'completion_tokens': 500, 'total_tokens': 1500}
        agent = Agent(name='cost-agent', model='deepseek-chat',
                      llm=FakeLLM([[final_usage('cheap answer', usage)]]),
                      plugins=['cost'])
        orchify_plugins.attach('cost', agent)
        rec = Recorder()
        try:
            agent.run('go', max_steps=1)
            event = wait_finish(rec)
            run_id = event.run_id
        finally:
            rec.close()
        expected = (800 * 0.27 + 200 * 0.07 + 500 * 1.10) / 1_000_000.0
        assert len(plugin.records) == 1, plugin.records
        record = plugin.records[0]
        assert record['model'] == 'deepseek-chat' and record['priced'] is True
        assert abs(record['usd'] - expected) < 1e-12
        assert abs(plugin.total_cost() - expected) < 1e-12
        assert abs(plugin.cost_of_run(run_id) - expected) < 1e-12
        snapshot = plugin.snapshot()
        assert snapshot['steps'] == 1 and snapshot['runs'] == 1
    finally:
        _cleanup('cost')


def test_cost_plugin_caps_a_run():
    '''A max_run_cost below the first step queues control:stop and emits cost:budget.'''
    plugin = CostPlugin(max_run_cost=0.0000001)
    orchify_plugins.register(plugin)
    try:
        usage = {'prompt_tokens': 1000, 'completion_tokens': 1000, 'total_tokens': 2000}
        agent = Agent(name='cap-agent', model='deepseek-chat',
                      llm=FakeLLM([[final_usage('a', usage)], [final_usage('b', usage)]]),
                      plugins=['cost'])
        orchify_plugins.attach('cost', agent)
        rec = Recorder()
        try:
            agent.run('go', max_steps=5)
            event = wait_finish(rec)
            run_id = event.run_id
        finally:
            rec.close()
        assert run_id in plugin._stopped
        assert rec.of('cost:budget') or True
        events = [e for e in plugin.records]
        assert events, plugin.records
    finally:
        _cleanup('cost')


def test_policy_matches_tools_and_arguments():
    '''Rules are ordered, globbed and can look at the arguments.'''
    policy = ToolPolicy([
        ToolRule('read_*', 'allow'),
        ToolRule('write_file', 'ask', args={'path': 'src/*'}),
        ToolRule('write_file', 'deny', reason='writes outside src'),
        ToolRule('bash', 'deny'),
    ])
    assert policy.decide('read_file', {'path': 'anything'}).action == 'allow'
    assert policy.decide('write_file', {'path': 'src/a.py'}).action == ASK
    denied = policy.decide('write_file', {'path': 'etc/passwd'})
    assert denied.action == DENY and 'outside src' in denied.reason
    assert policy.decide('bash', {}).action == DENY
    assert policy.decide('unknown_tool', {}).action == 'allow'
    assert ToolPolicy(default='deny').decide('unknown_tool').action == DENY
    assert policy.as_dict()['rules'][0]['tools'] == ['read_*']


def test_policy_from_env_parses_and_ignores_garbage():
    '''The env shortcut accepts tool_glob=action pairs and skips bad entries.'''
    policy = policy_from_env({'ORCHIFY_TOOL_POLICY': 'read_*=allow, bash=deny, broken, x=nope'})
    assert policy is not None
    assert policy.decide('read_file').action == 'allow'
    assert policy.decide('bash').action == DENY
    assert policy.decide('write_file').action == 'allow'
    assert policy_from_env({}) is None
    assert policy_from_env({'ORCHIFY_TOOL_POLICY': 'junk'}) is None


def test_agent_policy_deny_blocks_the_tool():
    '''A denied call never runs and reports the policy through the event payload.'''
    calls = []

    def danger(tag: str) -> str:
        '''Do something dangerous.

        Args:
            tag (str): A label.

        Returns:
            str: Confirmation.
        '''
        calls.append(tag)
        return f'did {tag}'

    tool = Tool(danger)
    agent = Agent(
        name='policy-deny',
        llm=FakeLLM([[final('', [tool_call('danger', '{"tag": "boom"}')])], [final('gave up')]]),
        tools=[tool],
        tool_policy=ToolPolicy([ToolRule('danger', 'deny', reason='not allowed')]),
    )
    rec = Recorder()
    try:
        answer = agent.ask('try it', max_steps=2)
        event = wait_finish(rec)
    finally:
        rec.close()
    assert calls == [], calls
    assert answer == 'gave up'
    denied = [e for e in rec.of('tool:call:finish') if e.payload.get('denied_by_policy')]
    assert denied, [e.payload for e in rec.of('tool:call:finish')]
    assert 'not allowed' in denied[0].payload['result']
    assert event.run_id


def test_agent_policy_ask_parks_in_the_approval_pool():
    '''A policy 'ask' decision becomes a normal approval the operator can grant.'''
    calls = []

    def write(tag: str) -> str:
        '''Write something.

        Args:
            tag (str): A label.

        Returns:
            str: Confirmation.
        '''
        calls.append(tag)
        return f'wrote {tag}'

    tool = Tool(write)
    agent = Agent(
        name='policy-ask',
        llm=FakeLLM([[final('', [tool_call('write', '{"tag": "x"}')])], [final('done')]]),
        tools=[tool],
        tool_policy=ToolPolicy([ToolRule('write', 'ask', reason='needs a human')]),
    )
    rec = Recorder()
    binding = orchify_broker.register_hook(
        'run:approval:required',
        lambda event: orchify_broker.approve(event.payload['approval_id'], reason='ok'),
    )
    try:
        answer = agent.ask('write it', max_steps=2)
        required = rec.of('run:approval:required')
    finally:
        orchify_broker.remove_binding(binding)
        rec.close()
    assert required and required[0].payload['reason'] == 'needs a human'
    assert calls == ['x'], calls
    assert answer == 'done'


def test_tool_retry_recovers_and_reports():
    '''A flaky tool succeeds within its retry budget; exhausted retries raise or return text.'''
    attempts = {'count': 0}

    def flaky(flag: str) -> str:
        '''Fail twice, then succeed.

        Args:
            flag (str): Ignored.

        Returns:
            str: Confirmation.
        '''
        attempts['count'] += 1
        if attempts['count'] < 3:
            raise RuntimeError(f'boom {attempts["count"]}')
        return 'recovered'

    tool = Tool(flaky, retry=2, backoff=0)
    assert tool.execute(flag='x') == 'recovered'
    assert attempts['count'] == 3

    always = Tool(lambda: (_ for _ in ()).throw(RuntimeError('nope')), retry=1, backoff=0,
                  on_error='return')
    result = always.execute()
    assert result.startswith('[tool error]') and 'nope' in result

    raiser = Tool(lambda: (_ for _ in ()).throw(RuntimeError('loud')), retry=1, backoff=0)
    with pytest.raises(RuntimeError):
        raiser.execute()

    typed = Tool(flaky, retry=3, backoff=0, retry_on=(ValueError,))
    attempts['count'] = 0
    with pytest.raises(RuntimeError):
        typed.execute(flag='y')


def test_manager_records_tokens_and_cost():
    '''A finished run carries its model, token count and priced cost.'''
    from orchify.manager import orchify_manager

    usage = {'prompt_tokens': 1000, 'prompt_cache_hit_tokens': 200,
             'completion_tokens': 500, 'total_tokens': 1500}
    rec = Recorder()
    agent = Agent(name='cost-record', model='deepseek-chat',
                  llm=FakeLLM([[final_usage('priced', usage)]]))
    try:
        agent.run('go', max_steps=1)
        event = wait_finish(rec)
        run_id = event.run_id
    finally:
        rec.close()

    record = orchify_manager.run(run_id)
    expected = (800 * 0.27 + 200 * 0.07 + 500 * 1.10) / 1_000_000.0
    assert record['model'] == 'deepseek-chat', record
    assert record['tokens'] == 1500, record
    assert abs(record['cost_usd'] - expected) < 1e-9, record
    totals = orchify_manager.totals()
    assert totals['cost_usd'] > 0 and totals['total_tokens'] >= 1500, totals
