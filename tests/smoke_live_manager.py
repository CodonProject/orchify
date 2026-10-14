import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, OpenAICompat, Tool, orchify_broker, orchify_manager as om
from orchify.event import BaseEvent


def wait_run(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline and orchify_broker.runs:
        time.sleep(0.2)
    return not orchify_broker.runs


def test_manager_live_run():
    print('--- A. manager sees a live run (pause/resume/history) ---')
    events = []
    parked = []
    bindings = [orchify_broker.register_hook('*', lambda e: events.append(e.event_type), owner='')]

    def pause_from_manager(event: BaseEvent):
        if parked:
            return
        parked.append(True)
        assert om.pause(event.run_id) is True, 'om.pause() must accept the run id'
        live = om.run(event.run_id)
        assert live is not None and live['status'] in ('running', 'paused'), live

    bindings.append(orchify_broker.register_hook('agent:reason:step', pause_from_manager, owner=''))

    def resume_from_manager(event: BaseEvent):
        def later():
            time.sleep(0.4)
            assert om.resume(event.run_id) is True
        threading.Thread(target=later, daemon=True).start()

    bindings.append(orchify_broker.register_hook('run:paused', resume_from_manager, owner=''))

    agent = Agent(name='LiveManager', llm=OpenAICompat(),
                  system_prompt='You are terse. Answer with a single word.')
    run_id = agent.run('Say only: ok', max_steps=1)
    finished = wait_run()
    for binding in bindings:
        orchify_broker.remove_binding(binding)

    assert finished, f'run never finished; events={events}'
    assert 'run:paused' in events and 'run:resumed' in events, events

    record = om.run(run_id)
    assert record is not None, f'manager lost run {run_id}'
    assert record['status'] == 'done' and record['kind'] == 'agent', record
    assert record['answer'], record
    assert record['total_steps'] == 1 and record['active'] is False
    history = [item for item in om.history(limit=50) if item['run_id'] == run_id]
    assert history, 'the finished run must be in the history'

    status = om.status()
    assert status['enabled'] is True and status['counts']['runs'] == 0
    assert any(item['name'] == 'LiveManager' for item in om.agents())
    assert om.totals()['runs'] >= 1 and om.hooks()['total'] >= 1
    assert om.backend()['thread_alive'] is True
    assert len(om.events(limit=200, run_id=run_id)) >= 3
    import json
    json.loads(om.to_json())
    report = om.report()
    for section in ('Orchify manager', 'RUNS', 'HISTORY', 'TOTALS'):
        assert section in report, section
    print(f'ok  live run {run_id} finished, answer={record["answer"]!r}, '
          f'history/totals/events/report all agree')


def test_manager_live_approval():
    print('--- B. approval pool through the manager ---')
    ran = []

    def read_secret() -> str:
        '''Read the secret value.

        Returns:
            str: The secret, proving the tool actually ran.
        '''
        ran.append(True)
        return 'SECRET-42'

    secret = Tool(func=read_secret, name='read_secret',
                  description='Read the secret value.', require_approval=True)

    events = []
    approved_ids = []

    def on_required(event: BaseEvent):
        approval_id = event.payload['approval_id']
        pending = [item for item in om.approvals(pending_only=True)
                   if item['approval_id'] == approval_id]
        if not pending:
            return
        live = om.run(event.run_id)
        print(f'  om.approvals(): {pending[0]["tool_name"]} args={pending[0]["args"]} '
              f'run status={live["status"] if live else None}')
        approved_ids.append(approval_id)
        assert om.approve(approval_id, reason='smoke test') is True

    bindings = [orchify_broker.register_hook('*', lambda e: events.append(e.event_type), owner=''),
                orchify_broker.register_hook('run:approval:required', on_required, owner='')]

    agent = Agent(name='LiveManagerApproval', llm=OpenAICompat(), tools=[secret],
                  system_prompt='You must use the provided tool, then report its result verbatim.')
    run_id = agent.run('Call read_secret and tell me the value.', max_steps=3)
    finished = wait_run()
    for binding in bindings:
        orchify_broker.remove_binding(binding)

    assert finished, f'run never finished; events={events}'
    if not approved_ids:
        print('SKIP: the model chose not to call the tool this time')
        return
    assert 'run:approval:resolved' in events, events
    assert ran == [True], ran

    stored = [item for item in om.approvals(run_id=run_id)
              if item['approval_id'] == approved_ids[0]]
    assert stored, 'the manager must retain the decided approval after the run ended'
    assert stored[0]['resolved'] is True and stored[0]['decision'] is True
    assert stored[0]['reason'] == 'smoke test'
    assert om.approvals(pending_only=True, run_id=run_id) == []
    assert om.run(run_id)['status'] == 'done'
    print('ok  live approval was decided through the manager and is still readable')


if __name__ == '__main__':
    test_manager_live_run()
    test_manager_live_approval()
    print('\nall live manager tests passed')
