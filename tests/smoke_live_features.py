import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINK = os.path.join(ROOT, '_smoke_features_sink.jsonl')

os.environ['ORCHIFY_MANAGER_SINK'] = SINK
os.environ['ORCHIFY_MAX_CONCURRENT_LLM_CALLS'] = '1'

from orchify import (Agent, OpenAICompat, orchify_broker, orchify_manager as om,
                     tool)
from orchify.backend import orchify_web_backend
from orchify.env import req_model
from orchify.gate import gate_stats
from orchify.manager import read_sink

MODEL = req_model('deepseek-chat')
os.environ['ORCHIFY_PRICE_TABLE'] = json.dumps(
    {MODEL: {'prompt': 0.5, 'completion': 1.0, 'cache_hit': 0.05}})

CALLS = []


@tool(cache=True)
async def async_lookup(key: str) -> str:
    '''Look up the value of a key (slow, cached).'''
    CALLS.append(key)
    if key == 'beta':
        await asyncio.sleep(0.05)
    return f'value-for-{key}'


def wait_runs(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline and orchify_broker.runs:
        time.sleep(0.2)
    return not orchify_broker.runs


def test_feature_run():
    print('--- A. budget + async tool + ask_user + sink ---')
    if os.path.exists(SINK):
        os.remove(SINK)

    events = []
    questions = []
    bindings = [orchify_broker.register_hook('*', lambda e: events.append(e.event_type),
                                             owner='')]

    def answer_question(event):
        if event.payload.get('kind') != 'question':
            return
        questions.append(event.payload.get('question', ''))
        assert orchify_broker.answer_approval(event.payload['approval_id'],
                                             'prod-env'), 'answering must succeed'

    bindings.append(orchify_broker.register_hook('run:approval:required',
                                                 answer_question, owner=''))

    agent = Agent(
        name='LiveFeatures',
        llm=OpenAICompat(),
        model=MODEL,
        tools=[async_lookup],
        enable_ask_user=True,
        budget_tokens=60000,
        budget_seconds=180,
        system_prompt=(
            'You are concise and follow instructions exactly. First call the tool '
            "async_lookup with key 'alpha'. Then call ask_user exactly once and ask "
            'which environment to use. Then answer with one line that contains both '
            'the looked-up value and the answer you were given.'
        ),
    )
    run_id = agent.run('Do the three steps now.', max_steps=4)
    finished = wait_runs()
    for binding in bindings:
        orchify_broker.remove_binding(binding)
    assert finished, f'run never finished; events={events[-10:]}'

    assert 'alpha' in CALLS, f'async_lookup never ran: {CALLS}'
    assert questions, 'the model never asked a question'
    assert 'ask_user' in events or 'run:approval:required' in events, events
    assert 'run:approval:resolved' in events, events

    record = om.run(run_id)
    assert record is not None and record['status'] == 'done', record
    assert record['answer'], record
    print(f'    answer: {record["answer"][:80]!r}')
    assert 'prod-env' in record['answer'], record['answer']
    assert 'value-for-alpha' in record['answer'], record['answer']

    budget = record['budget']
    assert budget['priced'] is True, budget
    assert budget['total_tokens'] > 0 and budget['cost_usd'] > 0, budget
    assert budget['limits']['tokens'] == 60000 and budget['limits']['seconds'] == 180, budget
    print(f"    budget: tokens={budget['total_tokens']} usd={budget['cost_usd']:.6f}")

    totals = om.totals()
    assert totals['cost_usd'] > 0 and totals['total_tokens'] > 0, totals

    decided = [item for item in om.questions() if item['run_id'] == run_id]
    assert decided and decided[0]['answer'] == 'prod-env', decided
    assert om.status()['counts']['pending_questions'] == 0

    assert om.sink_path == SINK, om.sink_path
    run_records = read_sink(SINK, kinds=['run'])
    assert len(run_records) == 1 and run_records[0]['run_id'] == run_id, run_records
    assert run_records[0]['status'] == 'done' and run_records[0]['budget']['cost_usd'] > 0
    assert read_sink(SINK, kinds=['event'], run_id=run_id, types=['run:*'])
    print(f"    sink: {om.status()['sink_records']} records in {os.path.basename(SINK)}")


def test_tool_cache_and_gates():
    print('--- B. tool result cache + gate wiring ---')
    gates = gate_stats()
    assert gates['llm']['enabled'] is True and gates['llm']['limit'] == 1, gates

    before = len(CALLS)
    first = async_lookup.execute(key='beta')
    assert first == 'value-for-beta' and async_lookup.last_cached is False
    second = async_lookup.execute(key='beta')
    assert second == 'value-for-beta' and async_lookup.last_cached is True
    assert len(CALLS) == before + 1, f'the cache missed: {CALLS}'
    info = async_lookup.cache_info()
    assert info['hits'] >= 1 and info['size'] >= 1, info
    print(f"    cache: {info}")

    async_lookup.clear_cache()
    assert async_lookup.cache_info()['size'] == 0
    assert orchify_web_backend._thread.is_alive()
    print('ok  cache miss/hit, gate wiring and the backend thread')


def main():
    '''Run the live feature checks and remove the sink file afterwards.

    Returns:
        int: 0 when every check passed.
    '''
    try:
        test_feature_run()
        test_tool_cache_and_gates()
    finally:
        om.close()
        try:
            if os.path.exists(SINK):
                os.remove(SINK)
        except OSError:
            pass
    print()
    print('all live feature tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
