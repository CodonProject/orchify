import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify import Agent, OrchifyManager, orchify_broker
from orchify.event import RuntimeEvent
from orchify.manager import read_sink
from test_pause_approval import (FakeLLM, Recorder, final, finish_payload,
                                wait_until)

ENV_SINK = 'ORCHIFY_MANAGER_SINK'


_COUNTER = [0]


def _new_path():
    _COUNTER[0] += 1
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, f'_sink_{os.getpid()}_{_COUNTER[0]}.jsonl')


def _cleanup(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _lines(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return [line for line in handle.read().splitlines() if line.strip()]


def _emit(event_type, run_id, payload=None):
    orchify_broker.emit(RuntimeEvent(agent_name='sink-helper', agent_code='sh01',
                                     turn_id='', event_type=event_type,
                                     run_id=run_id, payload=payload or {}))


def test_sink_records_events_and_runs():
    '''A configured sink writes every event plus the archived run record.'''
    path = _new_path()
    om = OrchifyManager(history=5, events=5, sink=path)
    try:
        rec = Recorder()
        agent = Agent(name='sink-agent', llm=FakeLLM([[final('the answer')]]))
        run_id = agent.run('go', max_steps=1)
        finish_payload(rec)
        rec.close()

        assert wait_until(lambda: len(read_sink(path, kinds=['run'])) == 1), \
            'the run record never reached the sink'
        records = [json.loads(line) for line in _lines(path)]
        assert all('record' in item for item in records)

        assert records[0]['record'] == 'event' and records[0]['event_type'] == 'run:start'

        run_records = [item for item in records if item['record'] == 'run']
        assert len(run_records) == 1
        run_record = run_records[0]
        assert run_record['run_id'] == run_id
        assert run_record['status'] == 'done' and run_record['answer'] == 'the answer'
        assert run_record['kind'] == 'agent', 'the run kind must survive the sink marker'

        assert [item['record'] for item in read_sink(path, kinds=['run'])] == ['run']
        assert len(read_sink(path, kinds=['event'])) == len(records) - 1
        assert all(item['run_id'] == run_id for item in read_sink(path, run_id=run_id))
        assert read_sink(path, run_id='ghost#1') == []
        assert read_sink(path, types=['run:*'])
        agent_events = [item for item in records
                        if item['record'] == 'event' and item['event_type'].startswith('agent:')]
        assert read_sink(path, types=['agent:*']) == agent_events and agent_events
        assert read_sink(path, limit=2) == records[-2:]
        assert read_sink(path, limit=0) == []
        assert om.replay() == records

        status = om.status()
        assert status['sink'] == path and status['sink_records'] == len(records)
        print('ok  sink writes event + run records; read_sink/replay filter them')
    finally:
        om.close()
        _cleanup(path)


def test_sink_is_opt_in_and_lazy():
    '''Without a path there is no file IO at all; the file opens on first write.'''
    path = _new_path()
    saved = os.environ.pop(ENV_SINK, None)
    om = OrchifyManager(history=2, events=2)
    try:
        assert om.sink_path == ''
        _emit('test:sink', 'lazy#1')
        assert not os.path.exists(path)
        assert read_sink('') == []
        assert read_sink(path + '.missing') == []
        assert read_sink(path, kinds=['run']) == []
        print('ok  the sink is opt-in, lazy, and read_sink tolerates missing files')
    finally:
        om.close()
        if saved is not None:
            os.environ[ENV_SINK] = saved
        _cleanup(path)


def test_sink_environment_variable_selects_the_path():
    '''ORCHIFY_MANAGER_SINK is honoured when no explicit path is given.'''
    path = _new_path()
    saved = os.environ.get(ENV_SINK)
    os.environ[ENV_SINK] = path
    om = OrchifyManager(history=2, events=2)
    try:
        assert om.sink_path == path
        _emit('test:sink', 'env#1')
        assert wait_until(lambda: os.path.exists(path)), 'the env sink was never opened'
        assert [item['event_type'] for item in read_sink(path)] == ['test:sink']
        print('ok  ORCHIFY_MANAGER_SINK selects the sink path')
    finally:
        om.close()
        if saved is None:
            os.environ.pop(ENV_SINK, None)
        else:
            os.environ[ENV_SINK] = saved
        _cleanup(path)


def test_sink_writes_when_the_event_buffer_is_zero():
    '''events=0 keeps nothing in memory but must still feed a configured sink.'''
    path = _new_path()
    om = OrchifyManager(history=1, events=0, sink=path)
    try:
        _emit('run:start', 'zero#1')
        _emit('test:sink', 'zero#1')
        _emit('run:finish', 'zero#1', {'stopped': False, 'answer': 'x'})
        assert wait_until(lambda: len(read_sink(path, kinds=['run'])) == 1)
        event_types = [item['event_type'] for item in read_sink(path, kinds=['event'])]
        assert event_types == ['run:start', 'test:sink', 'run:finish'], event_types
        assert list(om.events(limit=None)) == [], 'events=0 must retain nothing'
        print('ok  events=0 still writes every event to the sink')
    finally:
        om.close()
        _cleanup(path)


def test_read_sink_skips_malformed_lines():
    '''Blank, non-JSON and non-object lines are skipped; limit keeps the tail.'''
    path = _new_path()
    try:
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps({'record': 'event', 'run_id': 'm#1',
                                     'event_type': 'agent:answer'}) + '\n')
            handle.write('not json at all\n')
            handle.write('\n')
            handle.write('[1, 2, 3]\n')
            handle.write(json.dumps({'record': 'run', 'run_id': 'm#1',
                                     'status': 'done'}) + '\n')
            handle.write('{"record": "event", "run_id": "m#2"}\n')
        records = read_sink(path)
        assert [item.get('run_id') for item in records] == ['m#1', 'm#1', 'm#2']
        assert len(read_sink(path, kinds=['run'])) == 1
        assert read_sink(path, types=['agent:*']) == [records[0]]
        assert read_sink(path, limit=1) == [records[-1]]
        print('ok  read_sink skips malformed lines and honours limit')
    finally:
        _cleanup(path)


def main():
    '''Run every test in this file.

    Returns:
        int: 0 when all tests passed, 1 when any failed.
    '''
    tests = [
        test_sink_records_events_and_runs,
        test_sink_is_opt_in_and_lazy,
        test_sink_environment_variable_selects_the_path,
        test_sink_writes_when_the_event_buffer_is_zero,
        test_read_sink_skips_malformed_lines,
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
