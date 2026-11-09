import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, orchify_broker, orchify_manager
from orchify.broker import HookBinding
from orchify.testing import FakeLLM, final, wait_finish, Recorder, wait_until


def _event(event_type='hook:probe', payload=None, run_id=''):
    return orchify_broker.event(event_type, payload=payload or {}, run_id=run_id)


def test_priority_orders_dispatch_and_keeps_registration_order():
    order = []
    bindings = []
    try:
        bindings.append(orchify_broker.register_hook('hook:order', lambda e: order.append('low')))
        bindings.append(orchify_broker.register_hook('hook:order', lambda e: order.append('high'), priority=5))
        bindings.append(orchify_broker.register_hook('hook:order', lambda e: order.append('first'), priority=5))
        bindings.append(orchify_broker.register_hook('hook:order', lambda e: order.append('negative'), priority=-2))
        orchify_broker.emit(_event('hook:order'))
        assert order == ['high', 'first', 'low', 'negative'], order
        assert bindings[1].priority == 5 and bindings[3].priority == -2
    finally:
        for binding in bindings:
            orchify_broker.remove_binding(binding)
    print('ok  priority orders the dispatch')


def test_background_hook_does_not_block_emit():
    seen = []
    gate = threading.Event()
    slow = orchify_broker.register_hook('hook:slow', lambda e: (gate.wait(3.0), seen.append('slow')),
                                        background=True)
    fast = orchify_broker.register_hook('hook:slow', lambda e: seen.append('fast'))
    try:
        started = time.monotonic()
        orchify_broker.emit(_event('hook:slow'))
        elapsed = time.monotonic() - started
        assert elapsed < 0.5, f'emit blocked for {elapsed:.3f}s on a background hook'
        assert seen == ['fast'], seen
        assert orchify_broker.pending_hooks() >= 1
        gate.set()
        assert orchify_broker.drain_hooks(timeout=3.0) is True
        assert 'slow' in seen
        assert orchify_broker.pending_hooks() == 0
    finally:
        gate.set()
        orchify_broker.remove_binding(fast)
        orchify_broker.remove_binding(slow)
    print('ok  background hooks do not delay emit')


def test_background_hook_errors_are_logged():
    calls = []

    def boom(event):
        calls.append('boom')
        raise RuntimeError('background hook exploded')

    binding = orchify_broker.register_hook('hook:boom', boom, background=True)
    try:
        orchify_broker.emit(_event('hook:boom'))
        assert orchify_broker.drain_hooks(timeout=3.0) is True
        assert calls == ['boom']
        orchify_broker.emit(_event('hook:boom'))
        assert orchify_broker.drain_hooks(timeout=3.0) is True
        assert calls == ['boom', 'boom'], 'the dispatcher must survive a raising hook'
    finally:
        orchify_broker.remove_binding(binding)
    print('ok  background hook errors are contained')


def test_disposed_background_hook_is_skipped():
    seen = []
    gate = threading.Event()
    binding = orchify_broker.register_hook('hook:dispose', lambda e: seen.append(e.event_type),
                                           background=True)
    first = orchify_broker.register_hook('hook:dispose', lambda e: gate.wait(3.0),
                                          background=True)
    try:
        orchify_broker.emit(_event('hook:dispose'))
        orchify_broker.remove_binding(binding)
        gate.set()
        assert orchify_broker.drain_hooks(timeout=3.0) is True
        assert seen == [], f'a removed background hook must not run: {seen}'
    finally:
        gate.set()
        orchify_broker.remove_binding(binding)
        orchify_broker.remove_binding(first)
    print('ok  a removed background hook is skipped')


def test_decorator_and_api_pass_the_options_through():
    seen = []

    @orchify_broker.hook('hook:decorated', priority=3, background=True)
    def decorated(event):
        seen.append(('decorated', event.event_type))

    import orchify

    via_api = orchify.on('hook:decorated', lambda e: seen.append(('api', e.event_type)),
                         owner='hooks-test', priority=1)
    try:
        binding = orchify_manager.hooks()['bindings']
        mine = [item for item in binding if item['event_type'] == ['hook:decorated']]
        assert any(item['priority'] == 3 and item['background'] is True for item in mine), mine
        assert any(item['priority'] == 1 and item['background'] is False for item in mine), mine
        assert orchify_manager.hooks()['pending_background'] == 0
        orchify_broker.emit(_event('hook:decorated'))
        assert ('api', 'hook:decorated') in seen
        assert orchify_broker.drain_hooks(timeout=3.0) is True
        assert ('decorated', 'hook:decorated') in seen
        assert isinstance(via_api, HookBinding) and via_api.owner == 'hooks-test'
    finally:
        orchify.off(via_api)
        orchify_broker.remove_hooks('')
    print('ok  the decorator and orchify.on() forward priority and background')


def test_a_whole_run_with_a_background_hook():
    seen = []
    slow = orchify_broker.register_hook('agent:answer', lambda e: time.sleep(0.15) or seen.append('slow'),
                                        background=True)
    rec = Recorder()
    llm = FakeLLM([[{'content': 'hi', 'delay': 0.01}, final('hi')]])
    agent = Agent(name='hook-runner', llm=llm, system_prompt='sys')
    try:
        agent.run('go')
        wait_finish(rec)
        assert any(event.event_type == 'run:finish' for event in rec.events)
        assert orchify_broker.drain_hooks(timeout=3.0) is True
        assert seen == ['slow'], seen
    finally:
        rec.close()
        orchify_broker.remove_binding(slow)
    print('ok  a background hook runs alongside a real run')


def test_drain_hooks_times_out():
    gate = threading.Event()
    binding = orchify_broker.register_hook('hook:timeout', lambda e: gate.wait(3.0), background=True)
    try:
        orchify_broker.emit(_event('hook:timeout'))
        assert orchify_broker.drain_hooks(timeout=0.1) is False
        gate.set()
        assert orchify_broker.drain_hooks(timeout=3.0) is True
    finally:
        gate.set()
        orchify_broker.remove_binding(binding)
        orchify_broker.drain_hooks(timeout=3.0)
    print('ok  drain_hooks() reports a timeout')


def test_hook_binding_repr_shows_the_new_fields():
    binding = orchify_broker.register_hook('hook:repr', lambda e: None, priority=2, background=True)
    try:
        text = repr(binding)
        assert 'priority=2' in text and 'background=True' in text, text
    finally:
        orchify_broker.remove_binding(binding)
    print('ok  HookBinding exposes priority and background')


def test_candidate_cache_sees_hooks_registered_later():
    seen = []
    orchify_broker.emit(_event('hook:cache'))
    binding = orchify_broker.register_hook('hook:cache', lambda e: seen.append('late'))
    try:
        orchify_broker.emit(_event('hook:cache'))
        assert seen == ['late'], seen
    finally:
        orchify_broker.remove_binding(binding)
    orchify_broker.emit(_event('hook:cache'))
    assert seen == ['late'], seen
    print('ok  the candidate cache follows hook registration')


def main():
    tests = [
        test_priority_orders_dispatch_and_keeps_registration_order,
        test_background_hook_does_not_block_emit,
        test_background_hook_errors_are_logged,
        test_disposed_background_hook_is_skipped,
        test_decorator_and_api_pass_the_options_through,
        test_a_whole_run_with_a_background_hook,
        test_drain_hooks_times_out,
        test_hook_binding_repr_shows_the_new_fields,
        test_candidate_cache_sees_hooks_registered_later,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except Exception as exc:
            import traceback
            failed.append(test.__name__)
            print(f'FAIL {test.__name__}: {exc}')
            traceback.print_exc()
    if failed:
        print(f'\n{len(failed)} failed: {failed}')
        sys.exit(1)
    print(f'\nall {len(tests)} tests passed')


if __name__ == '__main__':
    main()
