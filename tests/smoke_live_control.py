'''Live smoke test for the control channel and the approval pool.

Runs two short real conversations through OpenAICompat:
  A. pause/resume mid-stream  (control:pause -> run:paused -> control:resume)
  B. manual approval pool     (tool with require_approval=True, approved by a hook)
'''
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, OpenAICompat, Tool, orchify_broker
from orchify.event import BaseEvent


def watch(events):
    '''Record every event type; returns a dismissable binding.'''
    return orchify_broker.register_hook('*', lambda e: events.append(e.event_type), owner='')


def wait_run(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline and orchify_broker.runs:
        time.sleep(0.2)
    return not orchify_broker.runs


def test_pause_live():
    print('--- A. pause/resume mid-stream ---')
    events = []
    records = []
    bindings = [watch(events),
                orchify_broker.register_hook('*', lambda e: records.append(e), owner='')]
    paused = []

    def pause_on_first_reason(event: BaseEvent):
        if paused:
            return
        paused.append(True)
        orchify_broker.feedback('control:pause', run_id=event.run_id)

    bindings.append(orchify_broker.register_hook('agent:reason:step', pause_on_first_reason, owner=''))

    def resume_later(event: BaseEvent):
        def later():
            time.sleep(0.4)
            orchify_broker.feedback('control:resume', run_id=event.run_id)

        threading.Thread(target=later, daemon=True).start()

    bindings.append(orchify_broker.register_hook('run:paused', resume_later, owner=''))

    agent = Agent(name='LivePause', llm=OpenAICompat(),
                  system_prompt='You are terse. Answer with a single word.')
    run_id = agent.run('Say only: ok', max_steps=1)
    finished = wait_run()
    for b in bindings:
        b.disposed = True
    orchify_broker.remove_hooks('')

    assert finished, f'run never finished; events={events}'
    print('events:', [e for e in events if e in ('run:paused', 'run:resumed', 'run:finish')])
    assert 'run:paused' in events, events
    assert 'run:resumed' in events, events
    assert events.index('run:paused') < events.index('run:resumed') < events.index('run:finish'), events
    # scheme B: one run_id for the whole execution, turn_id is '<run_id>#<turn>'
    assert run_id and agent.last_run_id == run_id, (run_id, agent.last_run_id)
    assert records and all(r.run_id == run_id for r in records), \
        [(r.event_type, r.run_id) for r in records]
    assert all(r.turn_id == f'{run_id}#{r.turn}' for r in records), \
        [(r.event_type, r.turn_id, r.turn) for r in records]
    print(f'ok  live pause parked and resumed run {run_id} '
          f'({len(records)} events, all turn_id == run_id#turn)')


def test_approval_live():
    print('--- B. approval pool ---')
    ran = []

    def read_secret() -> str:
        """Read the secret value."""
        ran.append(True)
        return 'SECRET-42'

    secret = Tool(func=read_secret, name='read_secret',
                  description='Read the secret value.', require_approval=True)

    events = []
    seen = []
    bindings = [watch(events)]

    def on_required(event: BaseEvent):
        pending = orchify_broker.get_approval(event.payload['approval_id'])
        seen.append(pending)
        print(f'  approval requested: {event.payload["tool_name"]} args={event.payload["args"]}')
        orchify_broker.approve(event.payload['approval_id'], reason='smoke test')

    bindings.append(orchify_broker.register_hook('run:approval:required', on_required, owner=''))

    agent = Agent(name='LiveApproval', llm=OpenAICompat(), tools=[secret],
                  system_prompt='You must use the provided tool, then report its result verbatim.')
    agent.run('Call read_secret and tell me the value.', max_steps=3)
    finished = wait_run()
    for b in bindings:
        b.disposed = True
    orchify_broker.remove_hooks('')

    assert finished, f'run never finished; events={events}'
    print('events:', [e for e in events if e.startswith('run:approval') or e == 'tool:call:finish'])
    if 'run:approval:required' not in events:
        print('SKIP: the model chose not to call the tool this time')
        return
    assert 'run:approval:resolved' in events, events
    assert seen and seen[0].tool_name == 'read_secret', seen
    assert 'tool:call:finish' in events, events
    assert ran == [True], ran
    print('ok  live approval released the tool call and it executed')


if __name__ == '__main__':
    test_pause_live()
    test_approval_live()
    print('\nall live control tests passed')
