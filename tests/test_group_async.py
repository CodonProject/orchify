import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify import Agent, Group
from orchify.backend import orchify_web_backend
from orchify.broker import orchify_broker
from orchify.errors import RunFailed, RunTimeout
from orchify.llm.base import LLMInterface
from orchify.testing import FakeLLM, final, wait_until


class BoomLLM(LLMInterface):
    '''LLM whose request always fails, used to test member failures.'''

    async def _core_request(self, **kwargs):
        '''Raise instead of streaming.

        Args:
            **kwargs: Ignored request parameters.

        Yields:
            None: Never yields; the request fails immediately.
        '''
        if False:
            yield None
        raise RuntimeError('member exploded')


def _agent(name, reply, delay=0.0, llm=None):
    '''Build a member agent with a scripted answer.

    Args:
        name (str): Agent name.
        reply (str): Answer the fake model returns.
        delay (float): Seconds the fake model waits before answering.
        llm (Optional[LLMInterface]): Backend to use instead of a FakeLLM.

    Returns:
        Agent: The member agent.
    '''
    if llm is None:
        plan = ([{'delay': delay}] if delay else []) + [final(reply)]
        llm = FakeLLM([plan])
    return Agent(name=name, llm=llm, system_prompt='sys')


def _group(name, *members, mode='broadcast', **kwargs):
    '''Build a group around the given members.'''
    return Group(name=name, agents=list(members), mode=mode, **kwargs)


def _events(group):
    '''Return the group event types recorded so far.'''
    return [event.event_type for event in group.broker.events]


def test_broker_await_run_does_not_block():
    '''await_run polls the completion flag of a run.'''

    run_id = 'unit#await'
    orchify_broker.start_task(run_id, None, context={'run_id': run_id, 'task_name': run_id})
    try:
        assert run_id in orchify_broker.active_runs()
        assert asyncio.run(orchify_broker.await_run(run_id, timeout=0.05)) is False
    finally:
        orchify_broker.finish_task(run_id)
    assert asyncio.run(orchify_broker.await_run(run_id, timeout=0.5)) is True
    assert asyncio.run(orchify_broker.await_run('unit#missing', timeout=0.05)) is False
    assert asyncio.run(orchify_broker.await_run(run_id, timeout=0.0)) is True


def test_await_run_wakes_up_when_the_run_finishes():
    '''await_run returns as soon as a running task completes.'''

    agent = _agent('awaiter', 'done', delay=0.2)
    run_id = agent.run('task')

    async def wait_while_running():
        assert await orchify_broker.await_run(run_id, timeout=0.05) is False
        return await orchify_broker.await_run(run_id, timeout=8)

    assert asyncio.run(wait_while_running()) is True
    assert agent.last_answer == 'done'


def test_group_arun_tallies_votes():
    '''arun() awaits every member and tallies a voting group.'''

    group = _group(
        'async-vote',
        _agent('av-1', 'Yes'),
        _agent('av-2', '  yes '),
        _agent('av-3', 'No'),
        mode='voting',
    )
    assert group.is_running() is False
    run_id = asyncio.run(group.arun('pick one', timeout=8))
    assert run_id.startswith('async-vote#')
    assert group.is_running() is False
    assert orchify_broker.is_run_active(run_id) is False
    assert len(group.last_run_ids) == 3
    assert [vote.agent_name for vote in group.last_votes] == ['av-1', 'av-2', 'av-3']
    assert group.last_winner == 'av-1'
    assert group.last_answer == 'Yes'
    assert _events(group) == ['start', 'step', 'step', 'step', 'step', 'vote', 'end']
    end = group.broker.get_events_by_type('end')[-1]
    assert end.payload['winner'] == 'av-1'
    assert end.payload['answer'] == 'Yes'
    assert len(end.payload['votes']) == 3
    assert end.payload['agent_run_ids'] == list(group.last_run_ids)


def test_group_arun_on_the_orchestrator_loop_thread():
    '''arun() completes on the loop thread itself, without consuming a worker.'''

    group = _group(
        'loop-vote',
        _agent('lv-1', 'alpha'),
        _agent('lv-2', 'beta'),
        mode='voting',
    )
    future = asyncio.run_coroutine_threadsafe(group.arun('pick one', timeout=8), orchify_web_backend._loop)
    run_id = future.result(timeout=20)
    assert run_id.startswith('loop-vote#')
    assert len(group.last_votes) == 2
    assert group.last_winner == 'lv-1'
    assert group.last_answer == 'alpha'


def test_group_aask_returns_the_answer():
    '''aask() awaits the run and returns its answer.'''

    voting = _group(
        'async-ask',
        _agent('aa-1', 'Yes'),
        _agent('aa-2', 'yes'),
        mode='voting',
    )
    assert asyncio.run(voting.aask('pick one', timeout=8)) == 'Yes'
    assert voting.last_winner == 'aa-1'

    broadcast = _group('async-plain', _agent('ap-1', 'plain answer'))
    assert asyncio.run(broadcast.aask('task', timeout=8)) == 'plain answer'


def test_group_arun_timeout_cancels_the_run():
    '''arun() raises RunTimeout and stops the members that overran.'''

    group = _group('async-slow', _agent('as-1', 'late', delay=1.5))
    try:
        asyncio.run(group.arun('task', timeout=0.05))
    except RunTimeout as exc:
        assert isinstance(exc, TimeoutError)
        assert exc.run_id == group.last_run_id
        assert exc.timeout == 0.05
    else:
        raise AssertionError('arun() did not report the timeout')

    assert group.is_running() is False
    assert orchify_broker.is_run_active(group.last_run_id) is False
    assert 'error' in _events(group)
    assert 'end' not in _events(group)
    assert wait_until(lambda: not group.agents['as-1'].is_running(), timeout=5)


def test_group_aask_reports_a_member_failure():
    '''aask() raises RunFailed when a member aborted without an answer.'''

    group = _group('async-boom', Agent(name='ab-1', llm=BoomLLM(), system_prompt='sys'))
    try:
        asyncio.run(group.aask('task', timeout=8))
    except RunFailed as exc:
        assert exc.run_id == group.last_run_id
        assert 'member exploded' in str(exc.error)
    else:
        raise AssertionError('aask() did not report the member failure')
    assert group.is_running() is False
    assert orchify_broker.is_run_active(group.last_run_id) is False


def test_group_arun_keeps_run_guard_and_events():
    '''arun() still refuses a second overlapping run and emits start/end.'''

    group = _group('async-guard', _agent('ag-1', 'first', delay=0.3), mode='voting')

    async def overlapping():
        first = asyncio.create_task(group.arun('one', timeout=8))
        await asyncio.sleep(0.05)
        try:
            await group.arun('two', timeout=8)
        except Exception as exc:
            assert type(exc).__name__ == 'RunAlreadyActive'
            assert 'already running' in str(exc)
        else:
            raise AssertionError('arun() allowed an overlapping run')
        return await first

    run_id = asyncio.run(overlapping())
    assert run_id.startswith('async-guard#')
    assert group.last_answer == 'first'
    assert group.is_running() is False
    assert orchify_broker.is_run_active(run_id) is False
    assert len(group.broker.get_events_by_type('end')) == 1


def main():
    '''Run every test in this module and report a summary.'''
    tests = [
        test_broker_await_run_does_not_block,
        test_await_run_wakes_up_when_the_run_finishes,
        test_group_arun_tallies_votes,
        test_group_arun_on_the_orchestrator_loop_thread,
        test_group_aask_returns_the_answer,
        test_group_arun_timeout_cancels_the_run,
        test_group_aask_reports_a_member_failure,
        test_group_arun_keeps_run_guard_and_events,
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
