import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, orchify_broker
from orchify.persistence import CheckpointStore, EventStore, load_agent, save_agent
from orchify.testing import FakeLLM, Recorder, final, tool_call, wait_finish, wait_until

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')


def _directory():
    '''Create a fresh directory under the ignored tests/.data tree.'''
    path = os.path.join(_DATA, f'persistence-{uuid.uuid4().hex[:8]}')
    os.makedirs(path, exist_ok=True)
    return path


def _drop(path):
    '''Remove a test directory.'''
    shutil.rmtree(path, ignore_errors=True)


def _usage_step(content, usage):
    '''A final step that reports usage.'''
    payload = {'content': content, 'reasoning': '', 'tool_calls': []}
    payload.update(usage)
    return {'final': payload}


def test_checkpoints_capture_each_step_and_resume_rewinds():
    '''Every finished step is checkpointed and can be restored into a new agent.'''
    directory = _directory()
    store = CheckpointStore(directory)
    try:
        calls = []

        def remember(text: str) -> str:
            '''Remember a value.

            Args:
                text (str): The value to remember.

            Returns:
                str: Confirmation text.
            '''
            calls.append(text)
            return f'stored {text}'

        agent = Agent(
            name='checkpointed',
            llm=FakeLLM([
                [final('', [tool_call('remember', '{"text": "alpha"}')])],
                [final('all done')],
            ]),
            tools=[__import__('orchify').Tool(remember)],
            checkpoint=store.callback,
        )
        rec = Recorder()
        try:
            assert agent.ask('remember alpha', max_steps=2) == 'all done'
            run_id = wait_finish(rec).run_id
        finally:
            rec.close()

        assert calls == ['alpha']
        history = store.history(run_id)
        assert len(history) >= 2, history
        assert history[-1]['step'] >= 2
        assert any(message.get('role') == 'tool' for message in history[-1]['messages'])
        assert run_id in store.runs()

        restored = Agent(name='restored', llm=FakeLLM([[final('continued')]]))
        record = store.resume(restored, run_id)
        assert record is not None
        assert any('alpha' in str(message.get('content')) for message in restored.messages)
        assert restored.ask('what did we do?', max_steps=1) == 'continued'
    finally:
        _drop(directory)


def test_resume_of_an_unknown_run_is_a_no_op():
    '''A run without checkpoints leaves the agent untouched and returns None.'''
    directory = _directory()
    try:
        store = CheckpointStore(directory)
        agent = Agent(name='fresh', llm=FakeLLM([[final('x')]]))
        before = list(agent.messages)
        assert store.resume(agent, 'nope#missing') is None
        assert agent.messages == before
    finally:
        _drop(directory)


def test_event_store_journals_a_run_and_can_detach():
    '''Attached to the broker, the journal records a run and stops when detached.'''
    directory = _directory()
    store = EventStore(directory, flush_every=0)
    try:
        rec = Recorder()
        agent = Agent(name='journalled', llm=FakeLLM([[_usage_step('answer', {'total_tokens': 12})]]))
        store.attach(orchify_broker)
        try:
            assert agent.ask('go', max_steps=1) == 'answer'
            run_id = wait_finish(rec).run_id
            wait_until(lambda: bool(list(store.events(run_id, event_type='run:start'))), timeout=5.0)
        finally:
            rec.close()
            store.flush()
        records = list(store.events(run_id))
        assert any(record['type'] == 'run:start' for record in records), records
        assert any(record['type'] == 'run:finish' for record in records), records
        finish = [record for record in records if record['type'] == 'run:finish'][-1]
        assert finish['payload']['usage']['total_tokens'] == 12
        safe = store.path_for(run_id)
        assert os.path.isfile(safe)

        store.detach()
        before = len(list(store.events(run_id)))
        agent.ask('again', max_steps=1)
        # The journal is a background hook: let the dispatcher finish the events
        # that were already queued before asserting that nothing new landed.
        orchify_broker.drain_hooks(timeout=5.0)
        store.flush()
        assert len(list(store.events(run_id))) == before
        assert run_id in store.runs()
    finally:
        _drop(directory)


def test_agent_snapshot_round_trip():
    '''save_agent/load_agent persist the conversation and model.'''
    directory = _directory()
    try:
        agent = Agent(name='snapshot', model='deepseek-chat',
                      llm=FakeLLM([[final('kept answer')]]))
        agent.ask('remember this', max_steps=1)
        path = os.path.join(directory, 'agent.json')
        assert save_agent(agent, path) == path

        other = Agent(name='snapshot', llm=FakeLLM([[final('hi')]]))
        payload = load_agent(other, path)
        assert payload['model'] == 'deepseek-chat'
        assert other.model == 'deepseek-chat'
        assert any('kept answer' in str(message.get('content')) for message in other.messages)
    finally:
        _drop(directory)
