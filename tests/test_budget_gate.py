import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, Group, Tool, tool, orchify_broker
from test_pause_approval import FakeLLM, Recorder, final, tool_call, wait_until, wait_finish, finish_payload


def final_usage(content, usage=None, tool_calls=None):
    '''A FinalStatus entry that also reports provider usage.'''
    payload = {'content': content, 'reasoning': '', 'tool_calls': tool_calls or []}
    for key, value in (usage or {}).items():
        payload[key] = value
    return {'final': payload}


def _set_env(**pairs):
    '''Set env vars, returning the previous values for _restore_env.'''
    saved = {}
    for name, value in pairs.items():
        saved[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    return saved


def _restore_env(saved):
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_budget_tokens_stops_the_run():
    '''An over-budget step stops the run and says why, instead of continuing.'''
    llm = FakeLLM([[
        {'content': 'chunky'},
        final_usage('over', {'total_tokens': 500, 'prompt_tokens': 400, 'completion_tokens': 100}),
    ]])
    rec = Recorder()
    agent = Agent(name='budgettok', llm=llm, budget_tokens=50)
    agent.run('go', max_steps=3)
    payload = finish_payload(rec)
    types = rec.types()
    exceeded = rec.of('run:budget:exceeded')[0].payload if rec.of('run:budget:exceeded') else {}
    rec.close()

    assert 'run:budget:exceeded' in types, types
    assert exceeded['reasons'] == ['tokens'], exceeded
    assert exceeded['violations'][0]['limit'] == 50
    assert exceeded['total_tokens'] == 500
    assert payload['stopped'] is True
    assert payload['budget']['exceeded'] is True
    assert payload['budget']['reasons'] == ['tokens']
    assert len(llm.calls) == 1, len(llm.calls)
    print('ok  token budget stops the run before the next request')


def test_budget_seconds_stops_the_run():
    '''The wall-clock cap is evaluated between steps, not mid-response.'''
    llm = FakeLLM([[{'content': 'slow', 'delay': 0.3}, final('slow done')]])
    rec = Recorder()
    agent = Agent(name='budgetsec', llm=llm, budget_seconds=0.05)
    agent.run('go', max_steps=1)
    payload = finish_payload(rec)
    exceeded = rec.of('run:budget:exceeded')[0].payload if rec.of('run:budget:exceeded') else {}
    rec.close()

    assert 'seconds' in exceeded.get('reasons', []), exceeded
    assert payload['stopped'] is True
    print('ok  seconds budget stops the run')


def test_budget_usd_aborts_without_a_price():
    '''A USD cap that cannot be priced refuses the run instead of guessing.'''
    rec = Recorder()
    agent = Agent(name='unpriced', llm=FakeLLM([[final('x')]]),
                  model='mystery-model-zzz', budget_usd=0.01)
    agent.run('go', max_steps=1)
    assert wait_until(lambda: bool(rec.of('agent:abort'))), rec.types()
    error = rec.of('agent:abort')[0].payload['error']
    types = rec.types()
    rec.close()

    assert 'ORCHIFY_PRICE_TABLE' in error, error
    assert 'mystery-model-zzz' in error, error
    assert 'run:start' not in types, types
    print('ok  an unpriced USD cap refuses the run with a clear error')


def test_budget_usage_reports_cost_and_manager_totals():
    '''Providers' usage is priced, reported on run:finish, and totalled.'''
    from orchify.manager import orchify_manager

    usage = {'prompt_tokens': 1000, 'prompt_cache_hit_tokens': 200,
             'completion_tokens': 500, 'total_tokens': 1500}
    rec = Recorder()
    agent = Agent(name='priced', llm=FakeLLM([[final_usage('priced answer', usage)]]),
                  model='deepseek-chat')
    agent.run('go', max_steps=1)
    event = wait_finish(rec)
    payload = event.payload
    run_id = event.run_id
    rec.close()

    expected = (800 * 0.27 + 200 * 0.07 + 500 * 1.10) / 1_000_000.0
    budget = payload['budget']
    assert abs(budget['cost_usd'] - round(expected, 6)) < 1e-9, budget
    assert budget['priced'] is True and budget['exceeded'] is False
    assert budget['limits'] == {'tokens': None, 'usd': None, 'seconds': None}

    totals = orchify_manager.totals()
    assert totals['cost_usd'] > 0, totals
    assert totals['total_tokens'] >= 1500, totals
    record = orchify_manager.run(run_id)
    assert record is not None and record['budget']['priced'] is True, record
    print('ok  usage is priced, reported and accumulated')


def test_price_table_helpers_and_env_override():
    '''Prefix matching, cost maths and the ORCHIFY_PRICE_TABLE override.'''
    from orchify.budget import (Budget, Price, _parse_price_table, cost_of,
                                load_price_table, price_for)

    mini = price_for('gpt-4o-mini-2024-07-18')
    assert mini is not None and mini.prompt == 0.15, mini
    assert price_for('gpt-4o-2024-08-06').prompt == 2.50
    assert price_for('mystery-model-zzz') is None
    assert _parse_price_table('not json') == {}
    assert _parse_price_table('') == {}
    big = Price(prompt=1.0, completion=2.0, cache_hit=0.5)
    assert cost_of({'prompt_tokens': 1_000_000, 'completion_tokens': 1_000_000}, big) == 3.0
    assert cost_of({'prompt_tokens': 1000, 'prompt_cache_hit_tokens': 400,
                    'completion_tokens': 0}, big) == 0.0008
    assert cost_of({}, big) == 0.0

    saved = _set_env(ORCHIFY_PRICE_TABLE='{"custom-x": {"prompt": 1.0, "completion": 2.0}}')
    try:
        table = load_price_table()
        assert table['custom-x']['prompt'] == 1.0
        assert 'gpt-4o' in table, 'built-ins must survive the merge'
        custom = price_for('custom-x-turbo')
        assert custom is not None and custom.completion == 2.0
        assert custom.cache_hit == 1.0, 'cache_hit falls back to the prompt price'

        budget = Budget(tokens=10, usd=0.5, seconds=1)
        assert budget.enabled() is True
        assert budget.limits() == {'tokens': 10, 'usd': 0.5, 'seconds': 1}
        assert budget.violations({'total_tokens': 11}, 0.1, 'custom-x')[0]['reason'] == 'tokens'
        assert budget.violations({'total_tokens': 1}, 2.0, 'custom-x')[0]['reason'] == 'seconds'
        assert budget.violations({'total_tokens': 1, 'prompt_tokens': 0}, 0.1, 'custom-x') == []
        assert Budget().enabled() is False
    finally:
        _restore_env(saved)
    print('ok  price table, cost maths and env override')


def test_run_gate_queues_the_second_run():
    '''With one run slot, a second run reports run:queued and waits its turn.'''
    from orchify.gate import reset_gates
    from orchify.manager import orchify_manager

    saved = _set_env(ORCHIFY_MAX_CONCURRENT_RUNS='1')
    reset_gates()
    rec = Recorder()
    rec.callbacks['agent:answer'] = lambda e: orchify_broker.feedback('control:pause', run_id=e.run_id)
    agent = Agent(name='gated', llm=FakeLLM([
        [{'content': 'first', 'delay': 0.1}, final('first done')],
        [final('second done')],
    ]), allow_concurrent_runs=True)
    try:
        agent.run('one', max_steps=1)
        assert wait_until(lambda: bool(rec.of('run:paused'))), rec.types()
        first_run = rec.of('run:paused')[0].run_id

        agent.run('two', max_steps=1)
        assert wait_until(lambda: bool(rec.of('run:queued'))), rec.types()
        queued = rec.of('run:queued')[0]
        assert queued.payload['limit'] == 1, queued.payload
        record = orchify_manager.run(queued.run_id)
        assert record is not None and record['status'] == 'queued', record

        orchify_broker.feedback('control:resume', run_id=first_run)
        assert wait_until(lambda: len(rec.of('run:finish')) >= 2, 12.0), rec.types()
        types = rec.types()
        assert types.index('run:queued') < types.index('run:dequeued'), types
        assert rec.of('run:dequeued')[0].payload['wait_seconds'] >= 0
    finally:
        rec.close()
        _restore_env(saved)
        reset_gates()
    print('ok  the run gate queues the second run and dequeues it later')


def test_run_gate_stop_while_queued_finishes_cleanly():
    '''A run stopped while queued still reports an (empty) run:finish.'''
    from orchify.gate import reset_gates

    saved = _set_env(ORCHIFY_MAX_CONCURRENT_RUNS='1')
    reset_gates()
    rec = Recorder()
    rec.callbacks['agent:answer'] = lambda e: orchify_broker.feedback('control:pause', run_id=e.run_id)
    agent = Agent(name='gated-stop', llm=FakeLLM([
        [{'content': 'first'}, final('first done')],
        [final('never')],
    ]), allow_concurrent_runs=True)
    try:
        agent.run('one', max_steps=1)
        assert wait_until(lambda: bool(rec.of('run:paused'))), rec.types()
        first_run = rec.of('run:paused')[0].run_id

        agent.run('two', max_steps=1)
        assert wait_until(lambda: bool(rec.of('run:queued'))), rec.types()
        second_run = rec.of('run:queued')[0].run_id
        assert orchify_broker.feedback('control:stop', run_id=second_run) is True

        assert wait_until(lambda: len(rec.of('run:finish')) >= 1, 8.0), rec.types()
        payload = rec.of('run:finish')[0].payload
        assert payload['stopped'] is True, payload
        assert payload['total_steps'] == 0, payload
        assert payload['budget'] is None, payload

        orchify_broker.feedback('control:resume', run_id=first_run)
        assert wait_until(lambda: len(rec.of('run:finish')) >= 2, 8.0), rec.types()
        assert 'run:dequeued' not in rec.types(), rec.types()
    finally:
        rec.close()
        _restore_env(saved)
        reset_gates()
    print('ok  a run stopped while queued finishes cleanly')


def test_gate_units_and_env_wiring():
    '''LimitGate / RateLimiter behaviour, plus the LLM gate and RPM env vars.'''
    from orchify.gate import LimitGate, RateLimiter, get_llm_gate, get_llm_rpm, reset_gates

    async def drive():
        gate = LimitGate(limit=1, name='unit')
        assert gate.enabled and gate.would_block() is False
        assert await gate.acquire() is True
        assert gate.in_use == 1 and gate.would_block() is True
        assert await LimitGate(limit=2).acquire(stop=lambda: True) is True
        assert await gate.acquire(stop=lambda: True) is False
        gate.release()
        assert gate.in_use == 0
        gate.release()
        assert gate.in_use == 0, 'release must never go negative'

        limiter = RateLimiter(per_minute=2, name='unit-rpm')
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        assert await limiter.acquire(stop=lambda: True) is False
        assert limiter.stats()['used_last_minute'] == 2
        assert await RateLimiter(0).acquire() is True
        assert LimitGate(0).enabled is False

    asyncio.run(drive())

    saved = _set_env(ORCHIFY_MAX_CONCURRENT_LLM_CALLS='2', ORCHIFY_LLM_RPM='7')
    try:
        reset_gates()
        assert get_llm_gate().enabled is True and get_llm_gate().limit == 2
        assert get_llm_rpm().enabled is True and get_llm_rpm().per_minute == 7
    finally:
        _restore_env(saved)
        reset_gates()
    print('ok  gate units and env wiring')


def test_plan_call_depth_and_cycle():
    '''plan_call() is the single place depth and cycles are decided.'''
    from orchify.call_chain import CallBlocked, plan_call

    path = plan_call('Agent:a', [], 'Agent:b', 3)
    assert path == ['Agent:a', 'Agent:b'], path
    deeper = plan_call('Agent:b', path, 'Agent:c', 3)
    assert deeper == ['Agent:a', 'Agent:b', 'Agent:c'], deeper
    assert plan_call('Agent:c', deeper, 'Agent:d', 3) == ['Agent:a', 'Agent:b', 'Agent:c', 'Agent:d']
    try:
        plan_call('Agent:b', path, 'Agent:a', 99)
    except CallBlocked as blocked:
        assert blocked.reason == 'cycle', blocked.reason
        assert blocked.callee == 'Agent:a' and blocked.path == path
    else:
        raise AssertionError('a cycle was not blocked')
    try:
        plan_call('Agent:c', deeper, 'Agent:d', 2)
    except CallBlocked as blocked:
        assert blocked.reason == 'depth' and blocked.limit == 2
        assert 'ORCHIFY_MAX_CALL_DEPTH' in str(blocked)
    else:
        raise AssertionError('the depth cap was not enforced')
    assert plan_call('Agent:a', ['Agent:a'] * 50, 'Agent:b', 0)[-1] == 'Agent:b'
    assert plan_call('Agent:solo', [], 'Agent:b', 5) == ['Agent:solo', 'Agent:b']
    print('ok  plan_call decides depth and cycles')


def test_agent_call_depth_guard_emits_event():
    '''A refused call raises CallBlocked and reports run:call:blocked.'''
    from orchify.call_chain import CallBlocked

    deep_a = Agent(name='depth-a', llm=FakeLLM([[final('a')]]), max_call_depth=1)
    deep_b = Agent(name='depth-b', llm=FakeLLM([[final('b')]]))
    rec = Recorder()
    try:
        deep_a._call_path = ['Agent:root', 'Agent:depth-a']
        try:
            deep_a.call_agent(deep_b, 'go')
        except CallBlocked as blocked:
            assert blocked.reason == 'depth', blocked.reason
        else:
            raise AssertionError('the agent depth cap was not enforced')

        blocked_events = rec.of('run:call:blocked')
        assert len(blocked_events) == 1, rec.types()
        payload = blocked_events[0].payload
        assert payload['reason'] == 'depth', payload
        assert payload['callee'] == 'Agent:depth-b', payload
        assert payload['call_path'] == ['Agent:root', 'Agent:depth-a'], payload
        assert payload['max_call_depth'] == 1, payload
    finally:
        rec.close()

    caller = Agent(name='depth-c', llm=FakeLLM([[final('c')]]), max_call_depth=3)
    callee = Agent(name='depth-d', llm=FakeLLM([[final('d')]]))
    rec = Recorder()
    try:
        caller.call_agent(callee, 'go')
        wait_finish(rec)
        assert callee._call_path == ['Agent:depth-c', 'Agent:depth-d'], callee._call_path
        assert callee._call_depth == 1, callee._call_depth
    finally:
        rec.close()
    print('ok  the agent call guard blocks, reports and hands down the path')


def test_group_call_guard_and_path_transfer():
    '''Group calls are guarded too, including the group-as-a-tool wrapper.'''
    from orchify.call_chain import CallBlocked

    member = Agent(name='gmember', llm=FakeLLM([[final('m')]]))
    other = Agent(name='gother', llm=FakeLLM([[final('o')]]))
    group = Group(name='gteam', agents=[member, other], mode='broadcast', max_call_depth=2)
    rec = Recorder()
    try:
        group.call_agent(member, 'hello')
        wait_finish(rec)
        assert member._call_path == ['Group:gteam', 'Agent:gmember'], member._call_path
        assert member._call_depth == 1, member._call_depth

        group._call_path = ['Group:root', 'Group:mid', 'Group:gteam']
        try:
            group.call_agent(other, 'nope')
        except CallBlocked as blocked:
            assert blocked.reason == 'depth', blocked.reason
        else:
            raise AssertionError('the group depth cap was not enforced')
        assert rec.of('run:call:blocked'), rec.types()

        wrapper = group.as_tool(caller_agent=member)
        member._call_path = ['Agent:outer', 'Agent:gmember']
        member.max_call_depth = 1
        try:
            wrapper.func(input='tool path')
        except CallBlocked as blocked:
            assert blocked.reason == 'depth', blocked.reason
        else:
            raise AssertionError('the group tool wrapper was not guarded')
    finally:
        rec.close()
    print('ok  the group call guard blocks and transfers the path')


def test_manager_surfaces_gates_and_queued_status():
    '''The manager reports the gates and the new queued run status.'''
    from orchify.manager import orchify_manager

    status = orchify_manager.status()
    assert set(status['gates']) == {'runs', 'llm', 'llm_rpm'}, status['gates']
    assert set(status['backend']['gates']) == {'runs', 'llm', 'llm_rpm'}
    assert 'queued' in status['counts']['runs_by_status'], status['counts']
    assert 'cost_usd' in orchify_manager.totals()
    print('ok  the manager surfaces gates, cost and the queued status')


def main():
    '''Run every test in this file; the first failure propagates.'''
    tests = [
        test_budget_tokens_stops_the_run,
        test_budget_seconds_stops_the_run,
        test_budget_usd_aborts_without_a_price,
        test_budget_usage_reports_cost_and_manager_totals,
        test_price_table_helpers_and_env_override,
        test_run_gate_queues_the_second_run,
        test_run_gate_stop_while_queued_finishes_cleanly,
        test_gate_units_and_env_wiring,
        test_plan_call_depth_and_cycle,
        test_agent_call_depth_guard_emits_event,
        test_group_call_guard_and_path_transfer,
        test_manager_surfaces_gates_and_queued_status,
    ]
    for test in tests:
        test()
    print(f'all {len(tests)} tests passed')


if __name__ == '__main__':
    main()
