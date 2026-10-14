import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify import Agent, Group, config, configure
from orchify.broker import Broker, orchify_broker
from orchify.errors import RunFailed
from orchify.gate import reset_gates
from orchify.llm.base import LLMInterface
from orchify.testing import FakeLLM, final, wait_until


class _Env:
    '''Set environment variables and restore them afterwards.'''

    def __init__(self):
        self.saved = {}

    def set(self, name, value):
        '''Set one variable, remembering its previous value.'''
        if name not in self.saved:
            self.saved[name] = os.environ.get(name)
        os.environ[name] = str(value)
        return self

    def restore(self):
        '''Put every remembered variable back.'''
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.saved.clear()


class BoomLLM(LLMInterface):
    '''Provider stub whose first request always fails.'''

    async def _core_request(self, messages=None, model=None, tools=None, **kwargs):
        '''Raise the provider error the agent reports.'''
        if False:
            yield None
        raise RuntimeError('provider exploded')


def _agent(name, reply, **kwargs):
    return Agent(name=name, llm=FakeLLM([[final(reply)]]), system_prompt='sys', **kwargs)


def test_terminal_outcome_is_recorded_for_a_finished_run():
    '''A finished run leaves its terminal payload in the broker registry.'''
    agent = _agent('done', 'the answer')
    run_id = agent.run('go')
    assert agent.wait(run_id, timeout=8) is True

    record = orchify_broker.run_outcome(run_id)
    assert record is not None
    assert record['run_id'] == run_id
    assert record['event_type'] == 'run:finish'
    assert record['payload']['answer'] == 'the answer'
    assert record['payload']['total_steps'] == 1
    assert record['payload']['stopped'] is False
    assert record['created_at'] > 0

    assert any(item['run_id'] == run_id for item in orchify_broker.run_outcomes())

    assert orchify_broker.clear_run_outcome(run_id) is True
    assert orchify_broker.run_outcome(run_id) is None
    assert orchify_broker.clear_run_outcome(run_id) is False


def test_ask_reports_an_abort_even_when_the_run_already_ended():
    '''ask() still sees the failure when the run ends before it waits.

    This is the regression test for the old hook-based capture: the hook was
    registered after ``run()`` returned, so a run that aborted in between left
    the capture empty, ``ask()`` found a finished run and quietly returned
    ``last_answer`` ('') instead of reporting the error.
    '''
    agent = Agent(name='boom', llm=BoomLLM(), system_prompt='sys')
    original = agent.run

    def run_then_finish(*args, **kwargs):
        run_id = original(*args, **kwargs)
        agent.wait(run_id, timeout=8)
        return run_id

    agent.run = run_then_finish
    try:
        agent.ask('go', timeout=8)
    except RunFailed as exc:
        assert 'provider exploded' in str(exc)
        assert 'provider exploded' in exc.error
        assert exc.run_id
    else:
        raise AssertionError('ask() did not raise RunFailed for an aborted run')

    record = orchify_broker.run_outcome(agent.last_run_id)
    assert record is not None
    assert record['event_type'] == 'agent:abort'
    assert 'provider exploded' in record['payload']['error']


def test_ask_returns_the_answer_when_the_run_ends_before_waiting():
    '''A run that finishes first still yields its answer through ask().'''
    agent = _agent('fast', 'raced answer')
    original = agent.run

    def run_then_finish(*args, **kwargs):
        run_id = original(*args, **kwargs)
        agent.wait(run_id, timeout=8)
        return run_id

    agent.run = run_then_finish
    assert agent.ask('go', timeout=8) == 'raced answer'


def test_member_runs_record_their_own_outcomes():
    '''Every member run of a group records a terminal outcome.'''
    first = _agent('m1', 'one')
    second = _agent('m2', 'two')
    group = Group(name='pair', agents=[first, second], mode='competition')
    group.run('solve')

    assert len(group.last_run_ids) == 2
    for run_id in group.last_run_ids:
        wait_until(lambda rid=run_id: orchify_broker.run_outcome(rid) is not None)
        record = orchify_broker.run_outcome(run_id)
        assert record['event_type'] == 'run:finish'
        assert record['payload']['answer'] in ('one', 'two')


def test_registry_is_bounded_and_can_be_disabled():
    '''The registry keeps the newest outcomes and honours a zero limit.'''
    env = _Env()
    try:
        env.set('ORCHIFY_RUN_OUTCOME_LIMIT', '2')
        broker = Broker()
        assert broker.run_outcome_limit == 2
        for index in range(4):
            broker.record_run_outcome(f'unit#{index}', 'run:finish', {'answer': str(index)})
        assert [item['run_id'] for item in broker.run_outcomes()] == ['unit#2', 'unit#3']
        assert broker.run_outcome('unit#0') is None
        assert broker.run_outcome('unit#3')['payload']['answer'] == '3'

        broker.record_run_outcome('unit#2', 'agent:abort', {'error': 'later failure'})
        record = broker.run_outcome('unit#2')
        assert record['event_type'] == 'agent:abort'
        assert len(broker.run_outcomes()) == 2

        env.set('ORCHIFY_RUN_OUTCOME_LIMIT', '0')
        disabled = Broker()
        assert disabled.run_outcome_limit == 0
        assert disabled.record_run_outcome('unit#x', 'run:finish', {}) is None
        assert disabled.run_outcome('unit#x') is None

        env.set('ORCHIFY_RUN_OUTCOME_LIMIT', 'junk')
        fallback = Broker()
        assert fallback.run_outcome_limit == 256
    finally:
        env.restore()


def test_run_outcome_limit_is_configurable():
    '''configure() reaches the live registry and config() reports it.'''
    env = _Env()
    try:
        configure(run_outcome_limit=8)
        assert orchify_broker.run_outcome_limit == 8
        assert config().run_outcome_limit == 8
        assert config().resolved()['run_outcome_limit'] == {
            'value': 8,
            'env': 'ORCHIFY_RUN_OUTCOME_LIMIT',
        }
    finally:
        env.restore()
        configure(run_outcome_limit=None)
        reset_gates()


def main():
    '''Run every test in this module and report a summary.'''
    tests = [
        test_terminal_outcome_is_recorded_for_a_finished_run,
        test_ask_reports_an_abort_even_when_the_run_already_ended,
        test_ask_returns_the_answer_when_the_run_ends_before_waiting,
        test_member_runs_record_their_own_outcomes,
        test_registry_is_bounded_and_can_be_disabled,
        test_run_outcome_limit_is_configurable,
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
