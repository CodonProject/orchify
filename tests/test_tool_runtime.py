import asyncio
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify import Agent, Tool, tool
from orchify.backend import orchify_web_backend

from test_pause_approval import (FakeLLM, Recorder, final, tool_call, wait_until,
                                 wait_finish)


class Counter:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def bump(self):
        with self.lock:
            self.calls += 1
        return self.calls


def make_cached(counter, **kwargs):
    def work(value: str = 'x') -> str:
        counter.bump()
        return f'work:{value}'

    return Tool(func=work, name='work', cache=True, **kwargs)


def _run_on_backend_loop(coro_fn, timeout=10.0):
    '''Run a coroutine on the shared backend loop and return its result.'''
    future = asyncio.run_coroutine_threadsafe(coro_fn(), orchify_web_backend._loop)
    return future.result(timeout=timeout)


def test_sync_cache_hit_miss_and_stats():
    counter = Counter()
    t = make_cached(counter)

    assert t.execute(value='a') == 'work:a'
    assert counter.calls == 1
    assert t.last_cached is False

    assert t.execute(value='a') == 'work:a'
    assert counter.calls == 1, 'second identical call must be served from cache'
    assert t.last_cached is True

    assert t.execute(value='b') == 'work:b'
    assert counter.calls == 2

    info = t.cache_info()
    assert info['enabled'] is True and info['maxsize'] == 256
    assert (info['hits'], info['misses']) == (1, 2), info
    assert info['size'] == 2 and info['inflight'] == 0

    t.clear_cache()
    assert t.cache_info()['size'] == 0
    assert t.execute(value='a') == 'work:a'
    assert counter.calls == 3, 'cleared cache must re-execute'

    t.reset_cache_stats()
    info = t.cache_info()
    assert (info['hits'], info['misses'], info['evictions']) == (0, 0, 0)
    print('ok  cache hit/miss/last_cached/stats/clear/reset')


def test_cache_ttl_expiry_and_disabled_path():
    counter = Counter()

    def slowish(value: str = 'x') -> str:
        counter.bump()
        return f'ttl:{value}'

    ttl_tool = Tool(func=slowish, name='ttl', cache=0.15)
    assert ttl_tool.execute(value='a') == 'ttl:a'
    assert ttl_tool.execute(value='a') == 'ttl:a'
    assert counter.calls == 1, 'within TTL the value is reused'
    time.sleep(0.25)
    assert ttl_tool.execute(value='a') == 'ttl:a'
    assert counter.calls == 2, 'expired entry must be recomputed'

    plain = Tool(func=slowish, name='plain')
    assert plain.cache is None and plain.cache_info()['enabled'] is False
    plain.execute(value='a')
    plain.execute(value='a')
    assert counter.calls == 4, 'cache=None must never cache'

    zero = Tool(func=slowish, name='zero', cache=0)
    assert zero.cache_info()['enabled'] is False
    print('ok  cache TTL expiry and disabled paths')


def test_cache_exceptions_are_not_cached():
    counter = Counter()

    def flaky(value: str = 'x') -> str:
        if counter.bump() == 1:
            raise ValueError('boom')
        return f'ok:{value}'

    t = Tool(func=flaky, name='flaky', cache=True)
    try:
        t.execute(value='a')
    except ValueError as exc:
        assert 'boom' in str(exc)
    else:
        raise AssertionError('expected the first call to raise')
    assert t.execute(value='a') == 'ok:a'
    assert t.cache_info()['size'] == 1
    print('ok  failed calls are never cached')


def test_cache_lru_bound_and_eviction():
    counter = Counter()
    t = make_cached(counter, cache_size=2)

    for value in ('a', 'b', 'c'):
        t.execute(value=value)

    info = t.cache_info()
    assert info['size'] == 2 and info['maxsize'] == 2
    assert info['evictions'] == 1, info

    t.execute(value='c')
    assert counter.calls == 3
    t.execute(value='a')
    assert counter.calls == 4
    print('ok  LRU bound evicts the least recently used entry')


def test_cache_single_flight_coalesces_concurrent_calls():
    counter = Counter()

    def slow(value: str = 'x') -> str:
        counter.bump()
        time.sleep(0.25)
        return f'one:{value}'

    t = Tool(func=slow, name='one-shot', cache=True)
    results = []
    results_lock = threading.Lock()

    def call():
        value = t.execute(value='same')
        with results_lock:
            results.append(value)

    threads = [threading.Thread(target=call) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert results == ['one:same'] * 6, results
    assert counter.calls == 1, f'expected one computation, got {counter.calls}'
    assert t.cache_info()['coalesced'] >= 1
    assert t.cache_info()['inflight'] == 0, 'in-flight slots must be released'
    print('ok  concurrent identical calls collapse to one computation')


def test_async_tool_runs_on_the_backend_loop():
    counter = Counter()
    loops = []

    async def add(a: int = 1, b: int = 2) -> int:
        loops.append(asyncio.get_running_loop())
        counter.bump()
        await asyncio.sleep(0.02)
        return a + b

    t = Tool(func=add, name='async-add', cache=True)
    assert t.is_async is True
    assert t.execute(a=1, b=2) == 3
    assert t.execute(a=1, b=2) == 3
    assert counter.calls == 1, 'async results are cached after resolution'
    assert t.last_cached is True
    assert loops and all(loop is orchify_web_backend._loop for loop in loops), \
        'the coroutine must run on the shared backend loop'
    print('ok  async tool resolves on the backend loop and caches its value')


def test_async_tool_timeout():
    counter = Counter()

    async def hang(seconds: float = 2.0) -> str:
        counter.bump()
        await asyncio.sleep(seconds)
        return 'too late'

    t = Tool(func=hang, name='hang', timeout=0.2)
    started = time.time()
    try:
        t.execute(seconds=2.0)
    except TimeoutError as exc:
        assert 'timed out after 0.2s' in str(exc), str(exc)
    else:
        raise AssertionError('expected a TimeoutError')
    assert time.time() - started < 1.5, 'the wait must be bounded by the timeout'
    assert counter.calls == 1
    print('ok  async tool timeout is enforced and reported')


def test_async_tool_refuses_the_loop_thread():
    async def echo(value: str = 'x') -> str:
        return value

    t = Tool(func=echo, name='echo-async')

    async def attempt():
        try:
            t.execute(value='x')
        except RuntimeError as exc:
            return str(exc)
        return 'no error'

    message = _run_on_backend_loop(attempt)
    assert 'backend event-loop thread' in message, message
    print('ok  async tool refuses to block the backend loop thread')


def test_async_tool_through_an_agent_run():
    counter = Counter()

    async def async_echo(value: str = '') -> str:
        counter.bump()
        await asyncio.sleep(0.02)
        return f'echo:{value}'

    echo = Tool(func=async_echo, name='async_echo')
    llm = FakeLLM([
        [final('calling', tool_calls=[tool_call('async_echo', '{"value": "hi"}')])],
        [final('done')],
    ])
    agent = Agent(name='async-agent', llm=llm, tools=[echo])
    rec = Recorder()
    try:
        agent.run('go', max_steps=3)
        payload = wait_finish(rec).payload
    finally:
        rec.close()

    finished = rec.of('tool:call:finish')
    assert finished, rec.types()
    result = finished[0].payload.get('result')
    assert result == 'echo:hi', result
    assert 'Error executing tool' not in str(result), result
    assert counter.calls == 1
    assert payload['answer'] == 'done'
    print('ok  an agent run executes an async tool and feeds back its real result')


def test_async_tool_decorator_and_sync_unchanged():
    @tool(name='deco-async', cache=True)
    async def deco_async(value: str = 'v') -> str:
        return f'deco:{value}'

    assert deco_async.is_async is True
    assert deco_async.execute(value='1') == 'deco:1'
    assert deco_async.execute(value='1') == 'deco:1'
    assert deco_async.cache_info()['hits'] == 1

    @tool(name='deco-sync')
    def deco_sync(value: str = 'v') -> str:
        return f'sync:{value}'

    assert deco_sync.is_async is False
    assert deco_sync.execute(value='1') == 'sync:1'
    assert deco_sync.execute() == 'sync:v'
    print('ok  decorator surfaces for async and sync tools')


def main():
    '''Run every test in this file.

    Returns:
        int: 0 when all tests passed, 1 when any failed.
    '''
    tests = [
        test_sync_cache_hit_miss_and_stats,
        test_cache_ttl_expiry_and_disabled_path,
        test_cache_exceptions_are_not_cached,
        test_cache_lru_bound_and_eviction,
        test_cache_single_flight_coalesces_concurrent_calls,
        test_async_tool_runs_on_the_backend_loop,
        test_async_tool_timeout,
        test_async_tool_refuses_the_loop_thread,
        test_async_tool_through_an_agent_run,
        test_async_tool_decorator_and_sync_unchanged,
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
