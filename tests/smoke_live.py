import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, OpenAICompat, orchify_broker
from orchify.event import BaseEvent

events = []


@orchify_broker.hook('*')
def on_event(event: BaseEvent):
    events.append((event.event_type, event.turn))
    if event.event_type in ('agent:answer', 'agent:reason:step', 'agent:finish'):
        piece = getattr(event, 'content', '') or ''
        if piece:
            print(f'  [{event.event_type}] {piece[:60]!r}')
    else:
        print(f'  [{event.event_type}]')


llm = OpenAICompat()
agent = Agent(
    name='Smoke',
    llm=llm,
    system_prompt='You are terse. Answer with a single word.',
)

print('run:start ->')
run_id = agent.run('Say only: ok', max_steps=1)
print('run_id =', run_id)

deadline = time.time() + 120
while time.time() < deadline and orchify_broker.runs:
    time.sleep(0.2)

finish = [e for e in events if e[0] == 'run:finish']
print('\nevent types:', [e[0] for e in events])
if not finish:
    print('FAIL: no run:finish (task still active?)')
    sys.exit(1)
print('finish observed; run_contexts empty:', not orchify_broker.run_contexts)
print('OK')
