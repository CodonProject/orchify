import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify import Agent, Group, Tool
from orchify.broker import orchify_broker
from orchify.errors import UnknownRun
from orchify.llm.openai import OpenAICompat
from orchify.testing import (FakeLLM, Recorder, final, finish_payload,
                             tool_call, wait_finish, wait_until)


def _hook(event_type, fn):
    '''Register a test hook and return its binding.'''
    return orchify_broker.register_hook(event_type, fn, owner='')


def _close(*bindings):
    '''Remove test hooks.'''
    for binding in bindings:
        if binding is not None:
            orchify_broker.remove_binding(binding)


def _echo_tool():
    '''A trivial tool the scripted model can call.'''
    return Tool(func=lambda q: f'echo:{q}', name='echo', description='Echo the input')


def test_run_dict_overload_and_validation():
    '''run() accepts a dict of options and rejects a bad step budget.'''
    llm = FakeLLM([[final('ok')]])
    agent = Agent(name='dict', llm=llm, system_prompt='sys')
    rec = Recorder()
    try:
        run_id = agent.run('hi', {'max_steps': 2, 'run_id': 'dict-run', 'temperature': 0.2})
        assert run_id == 'dict-run'
        assert finish_payload(rec)['answer'] == 'ok'
        assert agent.wait('dict-run', timeout=5) is True
        try:
            agent.run('hi', 0)
        except ValueError as exc:
            assert 'max_steps' in str(exc)
        else:
            raise AssertionError('max_steps=0 must be rejected')
        legacy = agent.run('hi', turn_id='legacy-run')
        assert legacy == 'legacy-run'
        assert agent.wait('legacy-run', timeout=5) is True
    finally:
        rec.close()
    print('ok  run() dict overload, max_steps validation and the turn_id alias')


def test_wait_and_cancel_without_a_run():
    '''wait() explains the missing run; cancel() just reports False.'''
    agent = Agent(name='idle', llm=FakeLLM(), system_prompt='sys')
    try:
        agent.wait()
    except UnknownRun as exc:
        assert 'has not run yet' in str(exc)
    else:
        raise AssertionError('wait() must raise UnknownRun before any run')
    assert agent.wait('nope', timeout=0) is False
    assert agent.cancel() is False
    assert agent.cancel('nope') is False
    print('ok  wait() raises UnknownRun and cancel() returns False without a run')


def test_from_env_builds_an_openai_client():
    '''from_env() builds the provider client when none is given.'''
    agent = Agent.from_env('env-built', system_prompt='sys')
    assert isinstance(agent.llm, OpenAICompat)
    assert agent.name == 'env-built'
    print('ok  from_env() builds an OpenAICompat client')


def test_reset_replaces_the_history():
    '''reset() restores just the system prompt, or the messages it is given.'''
    agent = Agent(name='resettable', llm=FakeLLM([[final('a')], [final('b')]]),
                  system_prompt='sys')
    agent.ask('first', timeout=8)
    assert agent.last_answer == 'a'
    agent.reset()
    assert agent.messages == [{'role': 'system', 'content': 'sys'}]
    assert agent.last_answer == '' and agent.last_run_id == ''
    agent.reset(messages=[{'role': 'user', 'content': 'only'}])
    assert agent.messages == [{'role': 'user', 'content': 'only'}]
    print('ok  reset() restores the system prompt or the given history')


def test_registration_and_plugin_surface():
    '''Tool registration, plugin toggles and prompt updates behave and fail loudly.'''
    agent = Agent(name='surface', llm=FakeLLM(), system_prompt='old prompt')
    try:
        agent.add_tool('nope')
    except TypeError as exc:
        assert 'requires a Tool' in str(exc)
    else:
        raise AssertionError('add_tool() must reject non-Tool values')
    tool = _echo_tool()
    agent.add_tool(tool)
    try:
        agent.add_tool(tool, replace=False)
    except ValueError as exc:
        assert 'already registered' in str(exc)
    else:
        raise AssertionError('a duplicate tool without replace must be rejected')
    assert agent.remove_tool('echo') is True
    assert agent.remove_tool('echo') is False

    agent.enable_plugin('p1')
    assert 'p1' in agent.enabled_plugins
    agent.disable_plugin('p1')
    assert 'p1' not in agent.enabled_plugins
    agent.set_plugins(['p2', 'p3'])
    assert agent.enabled_plugins == {'p2', 'p3'}
    agent.set_plugins(None)
    assert agent.enabled_plugins == set()

    agent.set_system_prompt('new prompt')
    assert agent.messages[0] == {'role': 'system', 'content': 'new prompt'}
    assert agent.system_prompt == 'new prompt'
    print('ok  tool/plugin registration and set_system_prompt write-back')


def test_set_system_prompt_in_run_mode_leaves_history_alone():
    '''In 'run' mode the shared messages are owned by the caller.'''
    agent = Agent(name='runmode', llm=FakeLLM(), system_prompt='old', messages_mode='run')
    before = [dict(m) for m in agent.messages]
    agent.set_system_prompt('changed')
    assert agent.messages == before
    assert agent.system_prompt == 'changed'
    print('ok  set_system_prompt() does not touch the history in run mode')


def test_description_defaults_and_setter():
    '''The description falls back to the first prompt line, and can be overridden.'''
    agent = Agent(name='desc', llm=FakeLLM(), system_prompt='First line here\nsecond')
    assert agent.description == 'First line here'
    agent.description = 'custom description'
    assert agent.description == 'custom description'
    plain = Agent(name='plain', llm=FakeLLM())
    assert plain.description == 'Agent: plain'
    print('ok  description default and setter')


def test_run_messages_drop_the_caller_system_prompt():
    '''A per-run history keeps the agent's prompt and drops a supplied system turn.'''
    llm = FakeLLM([[final('ok')]])
    agent = Agent(name='runs', llm=llm, system_prompt='sys', messages_mode='run')
    rec = Recorder()
    try:
        agent.run('hi', messages=[{'role': 'system', 'content': 'ignored'},
                                  {'role': 'user', 'content': 'provided context'}])
        finish_payload(rec)
    finally:
        rec.close()
    sent = llm.calls[0]['messages']
    assert sent[0] == {'role': 'system', 'content': 'sys'}
    contents = [m.get('content') for m in sent]
    assert 'provided context' in contents, contents
    assert 'ignored' not in contents, 'the supplied system turn must be dropped'
    print('ok  per-run messages keep the agent prompt and drop a supplied system turn')


def test_control_override_answer():
    '''control:override_answer answers without calling the model.'''
    llm = FakeLLM([[final('model answer')]])
    agent = Agent(name='override', llm=llm, system_prompt='sys')

    def override(event):
        orchify_broker.feedback('control:override_answer', run_id=event.run_id,
                                payload={'content': 'forced answer'})

    binding = _hook('agent:start', override)
    rec = Recorder()
    try:
        agent.run('hi')
        payload = finish_payload(rec)
    finally:
        rec.close()
        _close(binding)
    assert payload['answer'] == 'forced answer'
    assert payload['overridden'] is True
    assert llm.calls == [], 'the override must skip the model call'
    assert [event.content for event in rec.of('agent:answer')] == ['forced answer']
    print('ok  control:override_answer answers without a model call')


def test_control_switch_model():
    '''control:switch_model changes the model of the next request.'''
    llm = FakeLLM([[final('ok')]])
    agent = Agent(name='switch', llm=llm, system_prompt='sys')

    def switch(event):
        orchify_broker.feedback('control:switch_model', run_id=event.run_id,
                                payload={'model': 'switched-model'})

    binding = _hook('agent:start', switch)
    rec = Recorder()
    try:
        agent.run('hi')
        finish_payload(rec)
    finally:
        rec.close()
        _close(binding)
    assert llm.calls[0]['model'] == 'switched-model'
    print('ok  control:switch_model rewrites the request model')


def test_control_update_messages_append():
    '''control:update_messages append extends the history before the request.'''
    llm = FakeLLM([[final('ok')]])
    agent = Agent(name='append', llm=llm, system_prompt='sys')

    def append(event):
        orchify_broker.feedback('control:update_messages', run_id=event.run_id,
                                payload={'op': 'append',
                                         'messages': [{'role': 'user', 'content': 'extra ctx'}]})

    binding = _hook('agent:start', append)
    rec = Recorder()
    try:
        agent.run('hi')
        finish_payload(rec)
    finally:
        rec.close()
        _close(binding)
    assert [m.get('content') for m in llm.calls[0]['messages']] == ['sys', 'hi', 'extra ctx']
    print('ok  control:update_messages append extends the history')


def test_control_update_messages_replace_and_truncate():
    '''replace keeps the system prompt and truncate keeps only the first messages.'''
    tool_step = {'final': {'content': 'need tool', 'reasoning': '',
                           'tool_calls': [tool_call('echo', '{"q": "1"}')]}}

    def run_with(payload):
        llm = FakeLLM([[tool_step], [final('done')]])
        agent = Agent(name=f'update-{payload["op"]}', llm=llm, system_prompt='sys')
        agent.add_tool(_echo_tool())
        fired = {'done': False}

        def update(event):
            if fired['done']:
                return
            fired['done'] = True
            orchify_broker.feedback('control:update_messages', run_id=event.run_id,
                                    payload=payload)

        binding = _hook('agent:finish', update)
        rec = Recorder()
        try:
            agent.run('hi')
            finish_payload(rec)
        finally:
            rec.close()
            _close(binding)
        return llm

    llm = run_with({'op': 'replace',
                    'messages': [{'role': 'system', 'content': 'bogus'},
                                 {'role': 'user', 'content': 'replaced'}]})
    sent = llm.calls[1]['messages']
    contents = [m.get('content') for m in sent]
    assert sent[0] == {'role': 'system', 'content': 'sys'}, sent
    assert sent[1] == {'role': 'user', 'content': 'replaced'}, sent
    assert 'hi' not in contents and 'bogus' not in contents, sent

    llm = run_with({'op': 'truncate', 'keep': 0})
    sent = llm.calls[1]['messages']
    assert sent[0] == {'role': 'system', 'content': 'sys'}, sent
    assert 'hi' not in [m.get('content') for m in sent], sent
    print('ok  control:update_messages replace/truncate reshape the next request')


def test_pause_then_stop_ends_the_run():
    '''A pause can be cancelled with a stop instead of a resume.'''
    llm = FakeLLM([[final('never reached')]])
    agent = Agent(name='pausestop', llm=llm, system_prompt='sys')

    def pause(event):
        orchify_broker.feedback('control:pause', run_id=event.run_id)

    def stop(event):
        orchify_broker.feedback('control:stop', run_id=event.run_id)

    pause_binding = _hook('agent:start', pause)
    stop_binding = _hook('run:paused', stop)
    rec = Recorder()
    try:
        agent.run('hi')
        payload = finish_payload(rec)
    finally:
        rec.close()
        _close(pause_binding, stop_binding)
    assert payload['stopped'] is True
    assert rec.of('run:paused'), 'the run must have parked first'
    assert not rec.of('agent:answer')
    assert llm.calls == []
    print('ok  control:stop while paused ends the run without calling the model')


def test_as_tool_with_caller_records_the_chain():
    '''Exposing an agent as a tool records the call on the caller's chain.'''
    callee = Agent(name='chain-callee', llm=FakeLLM([[final('inner')]]), system_prompt='sys')
    caller = Agent(name='chain-caller', llm=FakeLLM(), system_prompt='sys')
    rec = Recorder()
    try:
        tool = callee.as_tool(caller_agent=caller)
        tool.description is not None
        run_id = tool.execute(input='do the work')
        assert run_id == callee.last_run_id
        assert callee.wait(timeout=5) is True
        wait_until(lambda: len(rec.of('agent:call')) == 1)
    finally:
        rec.close()
    payload = rec.of('agent:call')[0].payload
    assert payload['caller'] == 'chain-caller' and payload['callee'] == 'chain-callee'
    assert payload['call_type'] == 'agent'
    assert callee.get_call_chain() is caller.get_call_chain()
    assert caller.get_call_chain_str() == 'Agent:chain-caller -> Agent:chain-callee'
    assert callee.get_call_chain_str() == 'Agent:chain-caller -> Agent:chain-callee'
    print('ok  as_tool(caller_agent=...) records the call chain')


def test_call_group_delegates_and_emits():
    '''call_group() forwards to the group and reports the group call.'''
    member = Agent(name='call-member', llm=FakeLLM([[final('member answer')]]), system_prompt='sys')
    group = Group(name='call-group', agents=[member])
    caller = Agent(name='call-owner', llm=FakeLLM(), system_prompt='sys')
    rec = Recorder()
    try:
        group_run_id = caller.call_group(group, 'nested work')
        assert group_run_id.startswith('call-group#')
        wait_until(lambda: bool(rec.of('agent:call')))
        wait_until(lambda: orchify_broker.wait_run(group_run_id, timeout=5))
    finally:
        rec.close()
    payload = rec.of('agent:call')[0].payload
    assert payload['caller'] == 'call-owner' and payload['callee'] == 'call-group'
    assert payload['call_type'] == 'group'
    assert group._call_path[0] == 'Agent:call-owner'
    assert caller.get_call_chain_str() == 'Agent:call-owner -> Group:call-group'
    print('ok  call_group() delegates and records a group call')


def test_inject_and_deny_tool_results():
    '''Queued tool results replace execution; denials are fed back to the model.'''
    executed = {'count': 0}

    def echo(q):
        executed['count'] += 1
        return f'echo:{q}'

    llm = FakeLLM([[
        {'final': {'content': '', 'reasoning': '', 'tool_calls': [
            tool_call('echo', '{"q": "one"}', call_id='c1'),
            tool_call('echo', '{"q": "two"}', call_id='c2'),
        ]}},
    ], [final('done')]])
    agent = Agent(name='inject', llm=llm, system_prompt='sys')
    agent.add_tool(Tool(func=echo, name='echo', description='Echo'))

    def inject(event):
        orchify_broker.feedback('control:inject_tool_result', run_id=event.run_id,
                                payload={'tool_id': 'c1', 'result': 'injected result'})
        orchify_broker.feedback('control:deny_tool', run_id=event.run_id,
                                payload={'tool_id': 'c2', 'reason': 'not allowed'})

    binding = _hook('agent:finish', inject)
    rec = Recorder()
    try:
        agent.run('hi')
        finish_payload(rec)
    finally:
        rec.close()
        _close(binding)
    assert executed['count'] == 0, 'injected and denied calls must not execute the tool'
    tool_msgs = [m for m in llm.calls[1]['messages'] if m.get('role') == 'tool']
    assert [m['content'] for m in tool_msgs] == ['injected result', '[denied] not allowed']
    assert [m['tool_call_id'] for m in tool_msgs] == ['c1', 'c2']
    print('ok  injected and denied tool results skip execution')


def test_malformed_arguments_and_unknown_tools():
    '''Bad argument JSON degrades to no args and an unknown tool reports itself.'''
    llm = FakeLLM([[
        {'final': {'content': '', 'reasoning': '', 'tool_calls': [
            tool_call('nargs', '{"q": ', call_id='c1'),
            tool_call('nargs', '["a"]', call_id='c2'),
            tool_call('ghost', '{}', call_id='c3'),
        ]}},
    ], [final('done')]])
    agent = Agent(name='edges', llm=llm, system_prompt='sys')
    agent.add_tool(Tool(func=lambda: 'noargs ok', name='nargs', description='No args'))
    rec = Recorder()
    try:
        agent.run('hi')
        finish_payload(rec)
    finally:
        rec.close()
    tool_msgs = [m for m in llm.calls[1]['messages'] if m.get('role') == 'tool']
    assert [m['content'] for m in tool_msgs] == ['noargs ok', 'noargs ok',
                                                 "Tool 'ghost' not found."]
    print('ok  malformed arguments and unknown tools are handled')


def test_pause_and_stop_on_the_last_step():
    '''The checkpoint after the final step can still pause or stop the run.'''
    def run_to_the_end(control):
        llm = FakeLLM([[{'final': {'content': '', 'reasoning': '',
                                  'tool_calls': [tool_call('echo', '{"q": "1"}')]}}]])
        agent = Agent(name=f'last-{control}', llm=llm, system_prompt='sys')
        agent.add_tool(_echo_tool())

        def send(event):
            orchify_broker.feedback(f'control:{control}', run_id=event.run_id)

        binding = _hook('tool:call:finish', send)
        resume_binding = None
        if control == 'pause':
            resume_binding = _hook('run:paused', lambda event: orchify_broker.feedback(
                'control:resume', run_id=event.run_id))
        rec = Recorder()
        try:
            agent.run('hi', max_steps=1)
            payload = finish_payload(rec)
        finally:
            rec.close()
            _close(binding, resume_binding)
        return payload, rec

    payload, rec = run_to_the_end('pause')
    assert payload['total_steps'] == 1
    assert rec.of('run:paused') and rec.of('run:resumed')
    assert payload['stopped'] is False

    payload, rec = run_to_the_end('stop')
    assert payload['total_steps'] == 1
    assert payload['stopped'] is True
    print('ok  the final checkpoint honours pause and stop')


def test_call_chain_accessors_without_a_chain():
    '''An idle agent reports no chain instead of inventing one.'''
    agent = Agent(name='nochain', llm=FakeLLM(), system_prompt='sys')
    assert agent.get_call_chain() is None
    assert agent.get_call_chain_str() == 'Agent:nochain (no call chain)'
    print('ok  call chain accessors without a chain')


def test_call_group_accepts_any_runner():
    '''call_group() only needs something with name and run().'''

    class StubGroup:
        def __init__(self):
            self.name = 'stub-group'
            self.calls = []

        def run(self, user_input, **kwargs):
            self.calls.append(user_input)
            return 'stub-run'

    stub = StubGroup()
    caller = Agent(name='stub-caller', llm=FakeLLM(), system_prompt='sys')
    rec = Recorder()
    try:
        run_id = caller.call_group(stub, 'go')
        wait_until(lambda: bool(rec.of('agent:call')))
    finally:
        rec.close()
    assert run_id == 'stub-run'
    assert stub.calls == ['go']
    assert rec.of('agent:call')[0].payload['callee'] == 'stub-group'
    print('ok  call_group() works with any runnable object')


def main():
    '''Run every test in this module and report a summary.'''
    tests = [
        test_run_dict_overload_and_validation,
        test_wait_and_cancel_without_a_run,
        test_from_env_builds_an_openai_client,
        test_reset_replaces_the_history,
        test_registration_and_plugin_surface,
        test_set_system_prompt_in_run_mode_leaves_history_alone,
        test_description_defaults_and_setter,
        test_run_messages_drop_the_caller_system_prompt,
        test_control_override_answer,
        test_control_switch_model,
        test_control_update_messages_append,
        test_control_update_messages_replace_and_truncate,
        test_pause_then_stop_ends_the_run,
        test_as_tool_with_caller_records_the_chain,
        test_call_group_delegates_and_emits,
        test_inject_and_deny_tool_results,
        test_malformed_arguments_and_unknown_tools,
        test_pause_and_stop_on_the_last_step,
        test_call_chain_accessors_without_a_chain,
        test_call_group_accepts_any_runner,
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
