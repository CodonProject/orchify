'''
Latency contracts: control waiters must be woken, not polled.

These tests pin the behaviour behind ``ORCHIFY_POLL_INTERVAL``: resume, approval and
run completion used to be discovered by a 50 ms poll, so a huge ``POLL_INTERVAL``
made a pause take seconds to unwind. Every waiter now parks on an ``asyncio.Event``
that the broker sets directly, so the same work completes immediately while the
poll interval is deliberately nonsensical.
'''

import asyncio
import contextlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, orchify_broker
from orchify import broker as broker_mod
from orchify.backend import orchify_web_backend
from orchify.testing import FakeLLM, final

#: A poll interval so large that any waiter still polling it would miss the
#: 3 s ceiling these tests use.
_SLOW_POLL = 5.0
_CEILING = 1.0


@contextlib.contextmanager
def patched_poll(seconds=_SLOW_POLL):
    '''Temporarily force the broker poll interval to ``seconds``.'''
    original = broker_mod.POLL_INTERVAL
    broker_mod.POLL_INTERVAL = seconds
    try:
        yield
    finally:
        broker_mod.POLL_INTERVAL = original


def submit(coro):
    '''Run ``coro`` on the shared backend loop and return its concurrent future.'''
    return asyncio.run_coroutine_threadsafe(coro, orchify_web_backend._loop)


def wait_until(predicate, timeout=5.0):
    '''Spin until ``predicate()`` is true or ``timeout`` elapses.'''
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_wait_resume_wakes_on_feedback_not_on_a_tick():
    '''A parked run resumes the moment control:resume arrives.'''
    run_id = 'latency-resume'

    async def park():
        orchify_broker.arm_resume(run_id)
        return await orchify_broker.wait_resume(run_id, timeout=30)

    # feedback() only routes to a registered run, so seed the context this test
    # stands in for; a real run registers itself through start_task().
    orchify_broker.run_contexts[run_id] = {'run_id': run_id}
    with patched_poll():
        future = submit(park())
        try:
            assert wait_until(lambda: run_id in orchify_broker._pause_events), \
                'the pause waiter was never armed'
            started = time.monotonic()
            assert orchify_broker.feedback('control:resume', run_id=run_id) is True
            assert future.result(10) == 'resumed'
            elapsed = time.monotonic() - started
        finally:
            future.cancel()
            orchify_broker.run_contexts.pop(run_id, None)

    assert elapsed < _CEILING, f'resume took {elapsed:.3f}s with a {_SLOW_POLL}s poll interval'
    assert run_id not in orchify_broker._pause_events, 'pause waiter leaked'
    print(f'ok  resume wakes immediately ({elapsed * 1000:.1f} ms, poll interval ignored)')


def test_wait_approval_wakes_on_the_decision():
    '''A parked approval returns as soon as the decision lands.'''
    run_id = 'latency-approval'
    approval = orchify_broker.request_approval(run_id, tool_name='write_file', args={'path': 'a.md'})
    approval_id = approval.approval_id

    async def park():
        return await orchify_broker.wait_approval(approval_id, timeout=30, run_id=run_id)

    with patched_poll():
        future = submit(park())
        try:
            assert wait_until(lambda: approval_id in orchify_broker._approval_events), \
                'the approval event was never registered'
            started = time.monotonic()
            assert orchify_broker.approve(approval_id, reason='ok') is True
            resolved = future.result(10)
            elapsed = time.monotonic() - started
        finally:
            future.cancel()
            orchify_broker.release_approval(approval_id)

    assert resolved.approved is True and resolved.reason == 'ok', resolved
    assert elapsed < _CEILING, f'approval took {elapsed:.3f}s with a {_SLOW_POLL}s poll interval'
    assert run_id not in orchify_broker._approvals_by_run, 'per-run approval index leaked'
    print(f'ok  approval wakes immediately ({elapsed * 1000:.1f} ms)')


def test_await_run_returns_when_the_run_finishes():
    '''await_run futures complete on run:finish, not on the next tick.'''
    llm = FakeLLM([[{'content': 'thinking', 'delay': 0.05}, final('done')]])

    async def await_finish(run_id):
        return await orchify_broker.await_run(run_id, timeout=30)

    with patched_poll():
        agent = Agent(name='latency-run', llm=llm)
        started = time.monotonic()
        run_id = agent.run('go', max_steps=1)
        future = submit(await_finish(run_id))
        try:
            assert future.result(10) is True
            elapsed = time.monotonic() - started
        finally:
            future.cancel()

    assert elapsed < _CEILING, f'completion took {elapsed:.3f}s with a {_SLOW_POLL}s poll interval'
    print(f'ok  run completion observed immediately ({elapsed * 1000:.1f} ms)')


def test_emit_only_visits_matching_bindings():
    '''
    Literal bindings are indexed, so emit touches the matching ones plus the
    glob/star hooks - it never walks the whole registry.
    '''
    seen = []
    bindings = []
    try:
        for index in range(20):
            bindings.append(orchify_broker.register_hook(
                f'probe:one:{index}', lambda e: seen.append(e.event_type)))
        glob_binding = orchify_broker.register_hook('probe:many:*',
                                                    lambda e: seen.append('glob'))
        star_binding = orchify_broker.register_hook('*', lambda e: seen.append('star'))
        bindings.extend([glob_binding, star_binding])

        registry = orchify_broker.hooks
        orchify_broker.emit(orchify_broker.event('probe:one:7'))
        assert seen == ['probe:one:7', 'star'], seen
        assert orchify_broker.hooks is registry, 'emit rebuilt the hook list for a plain event'

        seen.clear()
        orchify_broker.emit(orchify_broker.event('probe:many:1'))
        assert seen == ['glob', 'star'], seen

        seen.clear()
        removed = bindings.pop(0)
        assert orchify_broker.remove_binding(removed) is True
        orchify_broker.emit(orchify_broker.event('probe:one:0'))
        assert seen == ['star'], seen
    finally:
        for binding in bindings:
            orchify_broker.remove_binding(binding)
    print('ok  emit dispatches through the literal index')


def test_limit_gate_hands_over_every_freed_slot():
    '''
    Two slots freed in one go must wake two waiters: the first taker leaves the
    queue with room still left, and the queue has to drain without another
    release() ever arriving.
    '''
    from orchify.gate import LimitGate

    async def scenario():
        gate = LimitGate(2, 'latency')
        gate._in_use = 2          # pretend two holders are already inside
        order = []

        async def worker(name):
            await gate.acquire()
            order.append(name)

        tasks = [asyncio.ensure_future(worker(name)) for name in ('c', 'd')]
        await asyncio.sleep(0.05)
        assert len(gate._waiters) == 2, len(gate._waiters)
        gate.release()
        gate.release()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
        return gate, order

    started = time.monotonic()
    gate, order = submit(scenario()).result(10)
    elapsed = time.monotonic() - started

    assert order == ['c', 'd'], order
    assert gate.in_use == 2, gate.in_use
    assert not gate._waiters, list(gate._waiters)
    print(f'ok  limit gate drains its queue ({elapsed * 1000:.1f} ms)')


def main():
    '''Run every test in this module.'''
    test_wait_resume_wakes_on_feedback_not_on_a_tick()
    test_wait_approval_wakes_on_the_decision()
    test_await_run_returns_when_the_run_finishes()
    test_emit_only_visits_matching_bindings()
    test_limit_gate_hands_over_every_freed_slot()


if __name__ == '__main__':
    main()
