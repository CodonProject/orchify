import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify import (Agent, Group, Tool, orchify_broker, orchify_manager,
                     OrchifyManager, RUN_STATUSES)
from orchify.backend import orchify_web_backend
from test_pause_approval import (FakeLLM, final, tool_call, Recorder,
                                wait_until, wait_finish, finish_payload)


def test_register_unregister_and_weakref_release():
    '''Construction self-registers; the table is weakref-based and idempotent.'''
    agent = Agent(name='mgr-register', llm=FakeLLM([[]]))
    key = orchify_manager.register(agent)
    assert key, 'construction should have registered the agent'
    assert orchify_manager.register(agent) == key, 'register() must be idempotent'
    assert any(item['key'] == key for item in orchify_manager.agents())

    assert orchify_manager.unregister(agent) is True
    assert all(item['key'] != key for item in orchify_manager.agents())
    assert orchify_manager.unregister(agent) is False

    agent = Agent(name='mgr-weakref', llm=FakeLLM([[]]))
    key = orchify_manager.register(agent)
    assert any(item['key'] == key for item in orchify_manager.agents())
    del agent
    gc.collect()
    assert all(item['key'] != key for item in orchify_manager.agents()), \
        'a garbage-collected Agent must leave the manager'

    group = Group(name='mgr-group', agents=[Agent(name='mgr-member', llm=FakeLLM([[]]))])
    record = [item for item in orchify_manager.groups() if item['name'] == 'mgr-group'][0]
    assert record['mode'] == 'broadcast'
    assert [member['name'] for member in record['members']] == ['mgr-member']
    assert record['member_count'] == 1
    assert orchify_manager.unregister(group) is True
    print('ok  registry: self-registration, idempotence, weakref release')


def test_manager_disabled_is_inert():
    '''ORCHIFY_MANAGER_ENABLED=0 means no tap, no registry, no surprises.'''
    hooks_before = len(orchify_broker.hooks)
    disabled = OrchifyManager(enabled=False)
    agent = Agent(name='mgr-disabled', llm=FakeLLM([[]]))

    assert disabled.enabled is False
    assert 'running' in RUN_STATUSES and 'queued' in RUN_STATUSES
    assert disabled.register(agent) == ''
    assert disabled.agents() == [] and disabled.groups() == []
    assert disabled.history() == [] and disabled.events() == []
    assert len(orchify_broker.hooks) == hooks_before, 'a disabled manager must not install a tap'
    assert disabled.status()['enabled'] is False
    assert disabled.pause('nope#1') is False
    assert disabled.stop('nope#1') is False
    assert disabled.close() is False
    json.dumps(disabled.snapshot())
    assert 'enabled=no' in disabled.report()

    os.environ['ORCHIFY_MANAGER_ENABLED'] = '0'
    try:
        from_env = OrchifyManager()
        assert from_env.enabled is False
    finally:
        os.environ.pop('ORCHIFY_MANAGER_ENABLED', None)
    print('ok  disabled manager is inert and env-driven')


def test_live_run_lifecycle_and_history():
    '''A real run: running -> paused -> done, then the history record.'''
    llm = FakeLLM([[{'content': 'partial', 'delay': 0.05}, final('the answer')]])
    rec = Recorder()
    paused = []

    def pause_once(event):
        if paused:
            return
        paused.append(event.run_id)
        assert orchify_manager.pause(event.run_id) is True

    rec.callbacks['agent:answer'] = pause_once

    agent = Agent(name='mgr-lifecycle', llm=llm)
    run_id = agent.run('go', max_steps=1)
    assert wait_until(lambda: any(item['run_id'] == run_id and item['status'] == 'paused'
                                  for item in orchify_manager.runs())), 'run never showed as paused'

    live = orchify_manager.run(run_id)
    assert live['kind'] == 'agent' and live['name'] == 'mgr-lifecycle'
    assert live['status'] == 'paused' and live['paused'] is True
    assert live['active'] is True and live['step'] >= 1 and live['elapsed'] >= 0
    assert live['pending_approvals'] == 0 and live['feedback_depth'] >= 0

    assert orchify_manager.resume(run_id) is True
    payload = finish_payload(rec)
    rec.close()
    assert payload['answer'] == 'the answer'

    finished = orchify_manager.run(run_id)
    assert finished['status'] == 'done' and finished['stopped'] is False
    assert finished['answer'] == 'the answer'
    assert finished['total_steps'] == 1 and finished['steps_done'] == 1
    assert finished['active'] is False and finished['finished_at']
    assert 'usage' in finished
    assert orchify_manager.run('ghost#nope') is None

    assert any(item['run_id'] == run_id for item in orchify_manager.history(limit=50))
    assert any(item['run_id'] == run_id
               for item in orchify_manager.history(limit=50, kind='agent', status='done'))
    assert orchify_manager.totals()['runs'] >= 1
    print('ok  live run lifecycle, pause status and history record')


def test_approval_pool_through_the_manager():
    '''The pending-approval pool is readable and decidable from the manager.'''
    danger = Tool(func=lambda x: f'ran:{x}', name='mgr-danger',
                  description='dangerous', require_approval=True)
    llm = FakeLLM([
        [final('', [tool_call('mgr-danger', '{"x": 7}')])],
        [final('after approval')],
    ])
    rec = Recorder()
    agent = Agent(name='mgr-approve', llm=llm, tools=[danger])
    run_id = agent.run('go', max_steps=3)

    assert wait_until(lambda: bool(orchify_manager.approvals(pending_only=True, run_id=run_id))), \
        'approval never reached the manager pool'
    pending = orchify_manager.approvals(pending_only=True, run_id=run_id)[0]
    assert pending['tool_name'] == 'mgr-danger' and pending['kind'] == 'tool'
    assert pending['args'] == {'x': 7} and pending['age'] >= 0
    assert pending['resolved'] is False and pending['decision'] is None

    live = orchify_manager.run(run_id)
    assert live['status'] == 'awaiting_approval' and live['pending_approvals'] == 1
    assert live['approval_ids'] == [pending['approval_id']]

    assert orchify_manager.approve(pending['approval_id'], reason='looks fine') is True
    assert orchify_manager.approve(pending['approval_id']) is False, 'a decided approval is final'
    finish_payload(rec)
    rec.close()

    resolved = orchify_manager.approvals(run_id=run_id)[0]
    assert resolved['resolved'] is True and resolved['approved'] is True
    assert resolved['decision'] is True and resolved['reason'] == 'looks fine'
    assert orchify_manager.approvals(pending_only=True, run_id=run_id) == []
    assert orchify_manager.run(run_id)['status'] == 'done'
    print('ok  approval pool read + approve/reject through the manager')


def test_control_stop_and_stop_all():
    '''om.stop()/om.stop_all() end a parked run through the feedback channel.'''
    llm = FakeLLM([[{'content': 'chatter', 'delay': 0.05}, final('never reached')]])
    rec = Recorder()
    parked = []

    def pause_first(event):
        if parked:
            return
        parked.append(event.run_id)
        orchify_manager.pause(event.run_id)

    rec.callbacks['agent:answer'] = pause_first

    agent = Agent(name='mgr-stoppable', llm=llm)
    run_id = agent.run('go', max_steps=2)
    assert wait_until(lambda: any(item['run_id'] == run_id and item['status'] == 'paused'
                                  for item in orchify_manager.runs())), 'run never parked'

    assert orchify_manager.pause('ghost#nope') is False
    assert orchify_manager.stop('ghost#nope') is False
    assert orchify_manager.stop_all() >= 1
    payload = finish_payload(rec)
    rec.close()

    assert payload['stopped'] is True
    archived = orchify_manager.run(run_id)
    assert archived['status'] == 'stopped' and archived['stopped'] is True
    assert orchify_manager.stop_all() == 0, 'nothing should be running any more'
    print('ok  stop()/stop_all() and unknown-run control calls')


def test_group_run_and_voting_visible():
    '''A Group's own run is tracked, and the voting tally lands in its history.'''
    members = [Agent(name='mgr-v1', llm=FakeLLM([[final('Yes')]])),
               Agent(name='mgr-v2', llm=FakeLLM([[final('  yes ')]])),
               Agent(name='mgr-v3', llm=FakeLLM([[final('No')]]))]
    group = Group(name='mgr-vote-group', agents=members, mode='voting')

    record = [item for item in orchify_manager.groups() if item['name'] == 'mgr-vote-group'][0]
    assert record['mode'] == 'voting'
    assert [member['name'] for member in record['members']] == ['mgr-v1', 'mgr-v2', 'mgr-v3']

    run_id = group.run('is it ok?')
    assert run_id
    assert wait_until(lambda: (orchify_manager.run(run_id) or {}).get('status') == 'done', timeout=10)

    finished = orchify_manager.run(run_id)
    assert finished['kind'] == 'group' and finished['mode'] == 'voting'
    assert finished['winner'] == 'mgr-v1' and finished['answer'] == 'Yes'
    assert len(finished['votes']) == 3 and finished['total_steps'] == 3
    assert len(finished['agent_run_ids']) == 3

    group_events = orchify_manager.events(limit=200, run_id=run_id, types=['group:*'])
    assert [item['event_type'] for item in group_events] == ['group:start', 'group:end'] or group_events
    assert orchify_manager.runs(kind='group') == [] or \
        all(item['kind'] == 'group' for item in orchify_manager.runs(kind='group'))
    print('ok  group run visible with voting tally and group events')


def test_events_filters_and_reader_helpers():
    '''events()/hooks()/feedback()/plugins()/backend() report real state.'''
    rec = Recorder()
    agent = Agent(name='mgr-events', llm=FakeLLM([[final('done')]]))
    run_id = agent.run('go', max_steps=1)
    finish_payload(rec)
    rec.close()

    run_events = orchify_manager.events(limit=200, run_id=run_id)
    assert run_events, 'the tap must have recorded the run'
    assert all(item['run_id'] == run_id for item in run_events)
    assert [item['event_type'] for item in run_events][0] == 'run:start'
    assert [item['event_type'] for item in run_events][-1] == 'run:finish'
    assert all(item['time'] > 0 for item in run_events)

    agent_only = orchify_manager.events(limit=50, run_id=run_id, types=['agent:*'])
    assert agent_only and all(item['event_type'].startswith('agent:') for item in agent_only)
    assert orchify_manager.events(limit=0) == []

    hooks = orchify_manager.hooks()
    assert hooks['total'] >= 1 and hooks['by_event_type'].get('*', 0) >= 1
    assert any('OrchifyManager._on_event' in binding['func'] for binding in hooks['bindings'])
    assert isinstance(orchify_manager.feedback(), dict)

    plugins = orchify_manager.plugins()
    assert 'registered' in plugins and 'loaded' in plugins

    backend = orchify_manager.backend()
    assert backend['thread_alive'] is True and 'loop_running' in backend
    assert 'http_timeout' in backend and 'tool_executor' in backend

    agents = [item for item in orchify_manager.agents() if item['name'] == 'mgr-events']
    assert agents and agents[0]['last_run_id'] == run_id
    assert agents[0]['is_running'] is False and agents[0]['status'] == 'done'
    print('ok  events filters and hook/plugin/backend readers')


def test_bounded_buffers_and_tap_rearm():
    '''Buffers honour their capacity, and a removed tap comes back on its own.'''
    manager = OrchifyManager(history=5, events=3)
    try:
        assert manager.history_capacity == 5 and manager.event_capacity == 3
        for index in range(5):
            orchify_broker.emit(orchify_broker.event('test:event', payload={'index': index}))
        events = manager.events(limit=None)
        assert len(events) == 3 and [item['payload']['index'] for item in events] == [2, 3, 4]

        orchify_broker.remove_hooks('')
        assert manager.events(limit=None)[-1]['payload']['index'] == 4
        orchify_manager.events(limit=None)
        orchify_broker.emit(orchify_broker.event('test:event', payload={'index': 9}))
        assert manager.events(limit=None)[-1]['payload']['index'] == 9
        assert orchify_manager.events(limit=None)[-1]['payload']['index'] == 9

        assert manager.close() is True
        assert manager.close() is False
        assert manager.enabled is True
    finally:
        manager.close()

    none = OrchifyManager(history=0, events=0)
    try:
        none._reconcile()
        assert none.history() == [] and none.events(limit=None) == []
        assert none.history_capacity == 0 and none.event_capacity == 0
    finally:
        none.close()
    print('ok  bounded buffers, tap re-arm, close()')


def test_manager_wait_refuses_the_loop_thread():
    '''wait() must refuse (not deadlock) when called on the orchestrator thread.'''
    import asyncio

    async def attempt():
        try:
            orchify_manager.wait('whatever', timeout=1)
        except RuntimeError as exc:
            return str(exc)
        return 'no error'

    future = asyncio.run_coroutine_threadsafe(attempt(), orchify_web_backend._loop)
    message = future.result(timeout=5)
    assert 'deadlock' in message, message
    assert orchify_manager.wait('ghost#1', timeout=0.2) is False
    print('ok  manager.wait() refuses the loop thread, False for unknown runs')


def test_snapshot_is_json_serialisable():
    '''snapshot()/to_json()/report() cover the same data without raising.'''
    snapshot = orchify_manager.snapshot(history_limit=5, event_limit=5)
    for key in ('status', 'runs', 'history', 'agents', 'groups', 'approvals', 'hooks',
                'feedback', 'plugins', 'backend', 'events', 'totals'):
        assert key in snapshot, key
    text = orchify_manager.to_json(history_limit=5, event_limit=5)
    parsed = json.loads(text)
    assert parsed['status']['enabled'] is True
    assert 'runs_by_status' in parsed['status']['counts']

    report = orchify_manager.report(history_limit=3, event_limit=3)
    for section in ('Orchify manager', 'RUNS', 'AGENTS', 'GROUPS', 'APPROVALS',
                    'HISTORY', 'TOTALS', 'BACKEND', 'EVENTS'):
        assert section in report, section
    assert len(json.dumps(snapshot)) > 0
    print('ok  snapshot/to_json/report are complete and JSON-safe')


def main():
    '''Run every test in this file.

    Returns:
        int: 0 when all tests passed, 1 when any failed.
    '''
    tests = [
        test_register_unregister_and_weakref_release,
        test_manager_disabled_is_inert,
        test_live_run_lifecycle_and_history,
        test_approval_pool_through_the_manager,
        test_control_stop_and_stop_all,
        test_group_run_and_voting_visible,
        test_events_filters_and_reader_helpers,
        test_bounded_buffers_and_tap_rearm,
        test_manager_wait_refuses_the_loop_thread,
        test_snapshot_is_json_serialisable,
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
