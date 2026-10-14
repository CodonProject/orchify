import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify import Agent, CallChain, Group
from orchify.call_chain import CallBlocked
from orchify.event import GroupEvent
from orchify.group import GroupEventBroker
from orchify.testing import FakeLLM, Recorder, final, tool_call, wait_until


def _agent(name, reply, **kwargs):
    return Agent(name=name, llm=FakeLLM([[final(reply)]]), system_prompt='sys', **kwargs)


def _group(name, *members, mode='broadcast', **kwargs):
    return Group(name=name, agents=list(members), mode=mode, **kwargs)


def _event(event_type, payload=None):
    return GroupEvent.build_group_event('grp', 'code', 't1', event_type, payload or {})


def test_group_broker_callbacks_are_removable():
    '''A registered group callback can be removed again through its handle.'''
    broker = GroupEventBroker()
    seen = []
    handle = broker.on_event(lambda event: seen.append(event.event_type))
    assert handle.active is True
    assert handle.broker is broker

    broker.emit(_event('group:start'))
    broker.emit(_event('group:end', {'answer': 'ok'}))
    assert seen == ['group:start', 'group:end']
    assert [event.event_type for event in broker.get_events_by_type('group:end')] == ['group:end']
    assert broker.get_events_by_type('group:missing') == []

    assert handle.close() is True
    assert handle.active is False
    assert handle.close() is False

    broker.emit(_event('group:vote'))
    assert seen == ['group:start', 'group:end']
    assert len(broker.events) == 3
    assert broker.remove_event(lambda event: None) is False


def test_group_callback_context_manager():
    '''The handle unregisters its callback when the with block ends.'''
    broker = GroupEventBroker()
    seen = []
    with broker.on_event(seen.append) as handle:
        broker.emit(_event('group:start'))
        assert handle.active is True
    assert handle.active is False
    broker.emit(_event('group:end'))
    assert len(seen) == 1


def test_group_emit_records_before_calling_a_failing_callback():
    '''A raising callback propagates but the event is still recorded.'''
    broker = GroupEventBroker()

    def boom(event):
        raise RuntimeError('callback exploded')

    broker.on_event(boom)
    try:
        broker.emit(_event('group:start'))
    except RuntimeError as exc:
        assert 'callback exploded' in str(exc)
    else:
        raise AssertionError('emit should propagate the callback error')
    assert len(broker.events) == 1


def test_as_tool_needs_a_manager_agent():
    '''A group without a manager agent refuses to be exposed as a tool.'''
    group = _group('solo', _agent('member', 'hi'))
    group.manager_agent = None
    try:
        group.as_tool()
    except ValueError as exc:
        assert 'manager agent' in str(exc)
    else:
        raise AssertionError('as_tool should refuse a group without a manager agent')


def test_group_as_tool_records_the_callers_chain():
    '''Wrapping a group in a tool runs it and records the caller's call chain.'''
    member = _agent('member', 'panel answer')
    group = _group('panel', member)
    caller = _agent('caller', 'unused')
    rec = Recorder()
    try:
        tool = group.as_tool(caller)
        assert tool.name == 'panel'
        assert 'panel' in tool.description

        run_id = tool.execute(input='do it')
        assert run_id.startswith('panel#')
        assert group.last_run_ids and group.last_run_ids[0] == member.last_run_id

        assert caller._call_chain is not None
        frame = caller._call_chain.frames[0]
        assert frame.caller == 'Agent:caller' and frame.callee == 'Group:panel'
        assert frame.call_type == 'group' and frame.input_data == 'do it'
        assert 'Group:panel' in caller.get_call_chain_str()
        assert group._call_path[-1] == 'Group:panel'

        wait_until(lambda: len(rec.of('agent:call')) == 1)
        call = rec.of('agent:call')[0]
        assert call.payload['caller'] == 'caller'
        assert call.payload['callee'] == 'panel'
        assert call.payload['call_type'] == 'group'
        assert call.payload['call_depth'] == 1
    finally:
        rec.close()


def test_group_call_chain_helpers_include_nested_groups():
    '''Flat and full call chains describe members, nested groups and sub-frames.'''
    inner = _group('inner', _agent('inner-one', 'inner answer'))
    outer_member = _agent('outer-one', 'outer answer')
    outer = _group('outer', outer_member, inner)

    flat = outer.get_call_chain()
    assert flat[0] == 'Group:outer'
    assert any('Agent:outer-one' in line for line in flat)
    assert any('Group:inner' in line for line in flat)

    full = outer.get_full_call_chain()
    assert full.root_caller == 'Group:outer'
    labels = [(frame.caller, frame.callee, frame.call_type) for frame in full.frames]
    assert ('Group:outer', 'Agent:outer-one', 'agent') in labels
    assert ('Group:outer', 'Group:inner', 'group') in labels
    assert ('Group:inner', 'Agent:inner-one', 'agent') in labels

    outer_member._call_chain = CallChain(root_caller='Agent:outer-one')
    outer_member._call_chain.add_frame('Agent:outer-one', 'Group:inner', 'group', 'task')
    assert any('    -> Group:inner' in line for line in outer.get_call_chain())
    assert any(frame.caller == 'Agent:outer-one' for frame in outer.get_full_call_chain().frames)


def test_call_agent_and_call_group_emit_and_return_run_ids():
    '''Direct calls return the callee run id and publish the group events.'''
    member = _agent('callee', 'callee answer')
    group = _group('caller-group', _agent('starter', 'starter answer'))
    other = _group('other-group', member)
    rec = Recorder()
    try:
        run_id = group.call_agent(member, 'hello')
        assert run_id == member.last_run_id and run_id.startswith('callee#')
        assert member._call_depth == 1
        assert member._call_path[-1] == 'Agent:callee'
        assert [event.event_type for event in group.broker.get_events_by_type('agent_call')] == ['agent_call']

        wait_until(lambda: not member.is_running())
        other_run = group.call_group(other, 'nested')
        assert other_run == other.last_run_id and other_run.startswith('other-group#')
        assert other._call_path[-1] == 'Group:other-group'
        assert [event.event_type for event in group.broker.get_events_by_type('group_call')] == ['group_call']

        wait_until(lambda: len(rec.of('group:call_agent')) == 1 and len(rec.of('group:call_group')) == 1)
        assert rec.of('group:call_agent')[0].payload['agent_name'] == 'callee'
        assert rec.of('group:call_group')[0].payload['callee_group'] == 'other-group'
    finally:
        rec.close()


def test_outgoing_calls_block_at_the_depth_cap_and_on_cycles():
    '''A blocked outgoing call emits run:call:blocked and re-raises CallBlocked.'''
    target = _agent('target', 'ok')
    group = _group('guard', _agent('other', 'ok'))
    rec = Recorder()
    try:
        group.max_call_depth = 1
        group._call_path = ['Agent:far']
        try:
            group.call_agent(target, 'hello')
        except CallBlocked as exc:
            assert exc.reason == 'depth' and exc.callee == 'Agent:target'
        else:
            raise AssertionError('the depth cap should block the call')

        group._call_path = ['Agent:target']
        group.max_call_depth = 0
        try:
            group.call_agent(target, 'hello')
        except CallBlocked as exc:
            assert exc.reason == 'cycle' and exc.caller == 'Group:guard'
        else:
            raise AssertionError('a repeated callee should be blocked')

        wait_until(lambda: len(rec.of('run:call:blocked')) == 2)
        blocked = rec.of('run:call:blocked')
        assert blocked[0].payload['reason'] == 'depth'
        assert blocked[0].payload['max_call_depth'] == 1
        assert blocked[0].payload['callee'] == 'Agent:target'
        assert blocked[1].payload['reason'] == 'cycle'
        assert blocked[1].payload['call_depth'] == 1
    finally:
        rec.close()


def test_competition_group_dispatches_every_member():
    '''A competition run starts all members and returns the first run id.'''
    first = _agent('c1', 'one')
    second = _agent('c2', 'two')
    group = _group('race', first, second, mode='competition')
    run_id = group.run('solve it')
    assert run_id.startswith('race#')
    assert len(group.last_run_ids) == 2
    assert group.broker.get_events_by_type('start')
    assert group.broker.get_events_by_type('step')
    assert [event.event_type for event in group.broker.get_events_by_type('end')] == ['end']
    wait_until(lambda: not first.is_running() and not second.is_running())


def test_member_agents_are_callable_as_tools():
    '''A member agent can call another member as a tool (group wiring).'''
    expert = _agent('expert', 'expert answer')
    manager = Agent(
        name='manager',
        llm=FakeLLM([[final('', tool_calls=[tool_call('expert', '{"input": "sub-task"}')])],
                     [final('combined answer')]]),
        system_prompt='sys',
    )
    group = _group('team', manager, expert)
    rec = Recorder()
    try:
        run_id = group.run('plan the work')
        assert run_id.startswith('team#')

        wait_until(lambda: manager.last_answer == 'combined answer')
        assert expert.last_answer == 'expert answer'
        assert group.last_run_ids[0] == manager.last_run_id

        wait_until(lambda: len(rec.of('group:agent_call')) == 1)
        call = rec.of('group:agent_call')[0]
        assert call.payload['group_name'] == 'team'
        assert call.payload['agent_name'] == 'expert'
        assert call.payload['input'] == 'sub-task'
    finally:
        rec.close()


def test_group_validation_edges_and_guarded_registration():
    '''Invalid group definitions are rejected and manager wiring may fail safely.'''
    member = _agent('member', 'hi')

    try:
        _group('bad-mode', member, mode='nope')
    except ValueError as exc:
        assert 'Unsupported mode' in str(exc)
    else:
        raise AssertionError('an unsupported mode should raise')

    try:
        _group('empty')
    except ValueError as exc:
        assert 'at least one agent' in str(exc)
    else:
        raise AssertionError('an empty group should raise')

    try:
        _group('ghost-manager', member, manager_agent_name='nobody')
    except ValueError as exc:
        assert 'not found in the group' in str(exc)
    else:
        raise AssertionError('an unknown manager agent should raise')

    expert = _agent('expert-2', 'expert answer')
    manager = Agent(name='boss', llm=FakeLLM([[final('done')]]), system_prompt='sys')
    team = _group('delegating', manager, expert, mode='manager_delegation')
    assert team.manager_agent is manager
    assert 'expert-2' in manager.tools
    assert 'boss' not in expert.tools

    from orchify.manager import orchify_manager

    original = orchify_manager.register

    def _boom(obj):
        raise RuntimeError('no manager available')

    orchify_manager.register = _boom
    try:
        safe = Group(name='still-works', agents=[_agent('solo-2', 'hi')], mode='broadcast')
    finally:
        orchify_manager.register = original
    assert safe.code


def main():
    '''Run every test in this module and report a summary.'''
    tests = [
        test_group_broker_callbacks_are_removable,
        test_group_callback_context_manager,
        test_group_emit_records_before_calling_a_failing_callback,
        test_as_tool_needs_a_manager_agent,
        test_group_as_tool_records_the_callers_chain,
        test_group_call_chain_helpers_include_nested_groups,
        test_call_agent_and_call_group_emit_and_return_run_ids,
        test_outgoing_calls_block_at_the_depth_cap_and_on_cycles,
        test_competition_group_dispatches_every_member,
        test_member_agents_are_callable_as_tools,
        test_group_validation_edges_and_guarded_registration,
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
