import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, Tool, tool, orchify_broker
from orchify.llm import LLMInterface
from orchify.llm.base import Chunk, FinalStatus, Response


class FakeLLM(LLMInterface):
    '''
    Scripted LLM. `script` is a list of plans, one per request(); each plan is a
    list of steps: {'content': str, 'delay': float} yields a stream chunk,
    {'final': {...}} yields the final status.
    The last plan repeats when more requests arrive than plans.
    '''

    def __init__(self, script):
        super().__init__()
        self.script = script
        self.calls = []

    async def _core_request(self, messages=None, model=None, tools=None, **kwargs):
        self.calls.append({'messages': [dict(m) for m in (messages or [])], 'model': model})
        plan = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        for entry in plan:
            if entry.get('delay'):
                await asyncio_sleep(entry['delay'])
            if entry.get('content') is not None:
                yield Response(current_chunk=Chunk(is_cot=False, content=entry['content']))
            if entry.get('final') is not None:
                yield Response(
                    current_chunk=None,
                    final_status=FinalStatus(**entry['final']),
                    is_final=True,
                )


def asyncio_sleep(seconds):
    import asyncio
    return asyncio.sleep(seconds)


def final(content, tool_calls=None):
    return {'final': {'content': content, 'reasoning': '', 'tool_calls': tool_calls or []}}


def tool_call(name, args, call_id='c1'):
    return {'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': args}}


class Recorder:
    '''Collects events and runs callbacks, with hooks registered owner-less so
    they fire regardless of the enabled-plugin filter.'''

    def __init__(self, callbacks=None):
        self.events = []
        self.lock = threading.Lock()
        self.callbacks = callbacks or {}
        self.binding = orchify_broker.register_hook('*', self._on_event, owner='')

    def _on_event(self, event):
        with self.lock:
            self.events.append(event)
        cb = self.callbacks.get(event.event_type)
        if cb is not None:
            cb(event)

    def types(self):
        with self.lock:
            return [e.event_type for e in self.events]

    def of(self, event_type):
        with self.lock:
            return [e for e in self.events if e.event_type == event_type]

    def close(self):
        binding, self.binding = self.binding, None
        if binding is not None:
            orchify_broker.remove_binding(binding)


def wait_until(predicate, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def wait_finish(rec, timeout=8.0):
    assert wait_until(lambda: bool(rec.of('run:finish')), timeout), \
        f'run never finished; events={rec.types()}'
    return rec.of('run:finish')[0]


def finish_payload(rec):
    return wait_finish(rec).payload


def test_configure_with_tools_is_idempotent_and_visible():
    '''Group context must not stack up, and must reach the LLM in agent mode.'''
    agent = Agent(name='ctx', llm=FakeLLM([[]]), system_prompt='BASE')
    agent._configure_with_tools([], {'mode_description': 'pipeline'})
    first = agent.system_prompt
    agent._configure_with_tools([], {'mode_description': 'pipeline'})
    second = agent.system_prompt

    assert first == second, f'context duplicated: {first!r} -> {second!r}'
    assert first.count('Group Collaboration Context') == 1
    assert agent.messages[0]['role'] == 'system'
    assert 'Group Collaboration Context' in agent.messages[0]['content']
    assert 'BASE' in agent.messages[0]['content']
    print('ok  configure_with_tools is idempotent and visible to the LLM')


def test_as_tool_and_description():
    agent = Agent(name='helper', llm=FakeLLM([[]]), system_prompt='Summarizes things well.')
    assert agent.description == 'Summarizes things well.'
    t = agent.as_tool()
    assert isinstance(t, Tool)
    assert 'Summarizes things well.' in t.description
    bare = Agent(name='bare', llm=FakeLLM([[]]))
    assert bare.description == 'Agent: bare'
    assert bare.as_tool().description
    print('ok  as_tool/description no longer raises AttributeError')


def test_tool_executor_env_and_laziness():
    from orchify import agent as agent_mod
    agent_mod.shutdown_tool_executor()
    assert agent_mod._tool_executor is None, 'executor must not exist before first use'
    os.environ['ORCHIFY_TOOL_WORKERS'] = '5'
    try:
        ex = agent_mod.get_tool_executor()
        assert ex._max_workers == 5, ex._max_workers
        assert ex is agent_mod.get_tool_executor(), 'executor must be shared'
    finally:
        os.environ.pop('ORCHIFY_TOOL_WORKERS', None)
        agent_mod.shutdown_tool_executor()
    print('ok  tool executor is lazy and honours ORCHIFY_TOOL_WORKERS')


def test_pause_mid_stream_and_lost_resume_race():
    '''
    A pause issued while the model is still streaming must park the run, and a
    resume sent in direct reaction to run:paused (before wait_resume is called)
    must not be lost.
    '''
    llm = FakeLLM([[
        {'content': 'one', 'delay': 0.08},
        {'content': 'two', 'delay': 0.08},
        {'content': 'three', 'delay': 0.08},
        final('mid-stream done'),
    ]])
    rec = Recorder()

    def pause_on_first_chunk(event):
        orchify_broker.feedback('control:pause', run_id=event.run_id)

    rec.callbacks['agent:answer'] = pause_on_first_chunk

    def resume_immediately(event):
        orchify_broker.feedback('control:resume', run_id=event.run_id)

    rec.callbacks['run:paused'] = resume_immediately

    agent = Agent(name='midstream', llm=llm)
    turn_id = agent.run('go', max_steps=1)
    payload = finish_payload(rec)
    rec.close()

    types = rec.types()
    assert 'run:paused' in types, types
    assert 'run:resumed' in types, types
    assert types.index('run:paused') < types.index('run:resumed') < types.index('run:finish'), types
    assert payload['stopped'] is False
    assert rec.of('run:resumed')[0].payload['status'] == 'resumed'
    assert llm.calls and len(llm.calls) == 1, len(llm.calls)
    print('ok  mid-stream pause honoured; resume cannot be lost')


def test_pause_on_last_step():
    '''A pause requested during the final step still parks before run:finish.'''
    llm = FakeLLM([[final('only answer')]])
    rec = Recorder()

    def pause_on_finish(event):
        orchify_broker.feedback('control:pause', run_id=event.run_id)

    rec.callbacks['agent:finish'] = pause_on_finish
    rec.callbacks['run:paused'] = lambda e: orchify_broker.feedback('control:resume', run_id=e.run_id)

    agent = Agent(name='laststep', llm=llm)
    agent.run('go', max_steps=1)
    payload = finish_payload(rec)
    types = rec.types()
    rec.close()

    assert 'run:paused' in types, types
    assert types.index('run:paused') < types.index('run:finish'), types
    assert payload['stopped'] is False
    print('ok  pause on the last step is honoured')


def test_final_checkpoint_after_last_tool_step():
    '''
    The very last window: a pause that arrives while the final step's tools are
    running must still park the run, because the loop exits right after them.
    '''
    noop = Tool(func=lambda: 'noop-ran', name='noop', description='noop')
    llm = FakeLLM([[final('', [tool_call('noop', '{}')])]])
    rec = Recorder()
    paused = []

    def pause_after_tool(event):
        if paused:
            return
        paused.append(True)
        orchify_broker.feedback('control:pause', run_id=event.run_id)

    rec.callbacks['tool:call:finish'] = pause_after_tool
    rec.callbacks['run:paused'] = lambda e: orchify_broker.feedback('control:resume', run_id=e.run_id)

    agent = Agent(name='finalckpt', llm=llm, tools=[noop])
    agent.run('go', max_steps=1)
    finish_payload(rec)
    types = rec.types()
    rec.close()

    assert 'tool:call:finish' in types, types
    assert 'run:paused' in types, types
    assert types.index('tool:call:finish') < types.index('run:paused') < types.index('run:finish'), types
    assert len(llm.calls) == 1, len(llm.calls)
    print('ok  final checkpoint catches a pause during the last tool step')


def test_stop_while_paused_terminates():
    '''control:stop must interrupt a parked run instead of hanging forever.'''
    llm = FakeLLM([[final('a')]])
    rec = Recorder()

    def pause_then_stop(event):
        orchify_broker.feedback('control:pause', run_id=event.run_id)
        threading.Thread(
            target=lambda: (time.sleep(0.2),
                            orchify_broker.feedback('control:stop', run_id=event.run_id)),
            daemon=True,
        ).start()

    rec.callbacks['agent:finish'] = pause_then_stop

    agent = Agent(name='stopwhilepaused', llm=llm)
    agent.run('go', max_steps=3)
    payload = finish_payload(rec)
    types = rec.types()
    rec.close()

    assert 'run:paused' in types, types
    assert payload['stopped'] is True, payload
    print('ok  control:stop interrupts a paused run')


def test_pause_timeout_self_resumes():
    '''A configured pause_timeout must not leave the run parked forever.'''
    llm = FakeLLM([[final('done')]])
    rec = Recorder()
    rec.callbacks['agent:finish'] = lambda e: orchify_broker.feedback('control:pause', run_id=e.run_id)

    agent = Agent(name='pausetimeout', llm=llm, pause_timeout=0.3)
    started = time.time()
    agent.run('go', max_steps=1)
    finish_payload(rec)
    elapsed = time.time() - started
    resumed = rec.of('run:resumed')
    rec.close()

    assert 0.25 <= elapsed < 5, elapsed
    assert resumed and resumed[0].payload['status'] == 'timeout', [e.payload for e in resumed]
    print('ok  pause_timeout self-resumes instead of hanging')


def test_approval_approved_executes_tool():
    ran = []
    danger = Tool(func=lambda x: ran.append(x) or f'ran:{x}', name='danger',
                  description='dangerous', require_approval=True)
    assert danger.needs_approval is True

    llm = FakeLLM([
        [final('', [tool_call('danger', '{"x": 7}')])],
        [final('tool finished')],
    ])
    rec = Recorder()
    decisions = []

    def approve(event):
        approval = orchify_broker.get_approval(event.payload['approval_id'])
        decisions.append(approval)
        orchify_broker.approve(event.payload['approval_id'], reason='looks fine')

    rec.callbacks['run:approval:required'] = approve

    agent = Agent(name='approve', llm=llm, tools=[danger])
    agent.run('go', max_steps=3)
    finish_payload(rec)
    types = rec.types()
    rec.close()

    assert 'run:approval:required' in types, types
    assert 'run:approval:resolved' in types, types
    assert decisions and decisions[0].tool_name == 'danger'
    assert decisions[0].args == {'x': 7}, decisions[0].args
    assert ran == [7], ran
    finishes = [e for e in rec.of('tool:call:finish') if e.tool_name == 'danger']
    assert finishes and finishes[0].payload['result'] == 'ran:7'
    resolved = rec.of('run:approval:resolved')[0]
    assert resolved.payload['approved'] is True and resolved.payload['timed_out'] is False
    print('ok  approved tool call executes')


def test_approval_rejected_blocks_tool():
    ran = []
    danger = Tool(func=lambda x: ran.append(x) or f'ran:{x}', name='danger',
                  description='dangerous', require_approval=True)

    llm = FakeLLM([
        [final('', [tool_call('danger', '{"x": 1}')])],
        [final('gave up')],
    ])
    rec = Recorder()
    rec.callbacks['run:approval:required'] = lambda e: orchify_broker.reject(
        e.payload['approval_id'], reason='too risky')

    agent = Agent(name='reject', llm=llm, tools=[danger])
    agent.run('go', max_steps=3)
    finish_payload(rec)
    rec.close()

    assert ran == [], f'tool must not run when rejected: {ran}'
    finishes = [e for e in rec.of('tool:call:finish') if e.tool_name == 'danger']
    assert finishes, 'a denied call must still report a tool result'
    assert '[approval denied]' in finishes[0].payload['result']
    assert 'too risky' in finishes[0].payload['result']
    resolved = rec.of('run:approval:resolved')[0]
    assert resolved.payload['approved'] is False
    print('ok  rejected tool call is blocked and fed back as a result')


def test_approval_timeout_fails_closed():
    ran = []
    danger = Tool(func=lambda x: ran.append(x) or 'ran', name='danger',
                  description='dangerous', require_approval=True)

    llm = FakeLLM([
        [final('', [tool_call('danger', '{"x": 1}')])],
        [final('gave up')],
    ])
    rec = Recorder()

    agent = Agent(name='timeout', llm=llm, tools=[danger], approval_timeout=0.3)
    agent.run('go', max_steps=3)
    finish_payload(rec)
    rec.close()

    assert ran == [], f'tool must not run on timeout: {ran}'
    resolved = rec.of('run:approval:resolved')[0]
    assert resolved.payload['approved'] is False
    assert resolved.payload['timed_out'] is True
    leftover = [a for a in orchify_broker.pending_approvals() if a.tool_name == 'danger']
    assert leftover == [], leftover
    print('ok  approval timeout fails closed (deny, tool never runs)')


def test_agent_level_approval_policy():
    '''require_approval=True on the Agent gates tools that did not opt in.'''
    ran = []
    plain = Tool(func=lambda: ran.append('x') or 'plain-ran', name='plain', description='plain')

    llm = FakeLLM([
        [final('', [tool_call('plain', '{}')])],
        [final('done')],
    ])
    rec = Recorder()
    rec.callbacks['run:approval:required'] = lambda e: orchify_broker.approve(e.payload['approval_id'])

    agent = Agent(name='policy', llm=llm, tools=[plain], require_approval=True)
    agent.run('go', max_steps=3)
    finish_payload(rec)
    rec.close()

    assert ran == ['x'], ran
    assert 'run:approval:required' in rec.types()
    print('ok  Agent-level require_approval policy gates every tool')


def test_callable_approval_policy():
    '''A callable policy can gate selectively on tool name/args.'''
    hits = []
    safe = Tool(func=lambda n: hits.append(n) or n, name='safe', description='safe')

    llm = FakeLLM([
        [final('', [tool_call('safe', '{"n": "a"}', 'c1'),
                    tool_call('safe', '{"n": "blocked"}', 'c2')])],
        [final('done')],
    ])
    rec = Recorder()
    requested = []

    def gate(tool, args):
        return args.get('n') == 'blocked'

    rec.callbacks['run:approval:required'] = lambda e: (
        requested.append(e.payload['tool_name']),
        orchify_broker.reject(e.payload['approval_id'], reason='not this one'),
    )

    agent = Agent(name='callable', llm=llm, tools=[safe], require_approval=gate)
    agent.run('go', max_steps=3)
    finish_payload(rec)
    rec.close()

    assert requested == ['safe'], requested
    assert hits == ['a'], hits
    print('ok  callable approval policy gates selectively')


def test_update_messages_tolerates_bad_payload():
    '''Malformed control:update_messages payloads must not kill the run.'''
    llm = FakeLLM([[final('one')], [final('two')]])
    rec = Recorder()

    def bad_update(event):
        orchify_broker.feedback('control:update_messages', run_id=event.run_id,
                                payload={'op': 'append', 'messages': ['not-a-dict', None, {'role': 'user', 'content': 'ok'}]})
        orchify_broker.feedback('control:update_messages', run_id=event.run_id,
                                payload={'op': 'replace', 'messages': [42, {'role': 'user', 'content': 'still ok'}]})

    rec.callbacks['agent:answer'] = bad_update
    agent = Agent(name='badupdate', llm=llm)
    agent.run('go', max_steps=2)
    payload = finish_payload(rec)
    rec.close()

    assert payload, 'run must finish despite the malformed payload'
    assert not rec.of('agent:abort'), rec.types()
    print('ok  malformed update_messages payloads are ignored, run survives')


def test_stop_while_awaiting_approval_ends_run():
    '''
    A control:stop that lands while the run is parked on an approval must both
    deny the call and end the run - the stop must not be swallowed.
    '''
    ran = []
    danger = Tool(func=lambda: ran.append(1) or 'ran', name='danger',
                  description='dangerous', require_approval=True)

    llm = FakeLLM([
        [final('', [tool_call('danger', '{}')])],
        [final('should never be reached')],
    ])
    rec = Recorder()
    stopped = []

    def request_then_stop(event):
        orchify_broker.feedback('control:stop', run_id=event.run_id)

    rec.callbacks['run:approval:required'] = request_then_stop

    agent = Agent(name='stoponapproval', llm=llm, tools=[danger])
    agent.run('go', max_steps=3)
    payload = finish_payload(rec)
    resolved = rec.of('run:approval:resolved')
    rec.close()

    assert ran == [], f'tool must not run: {ran}'
    assert payload['stopped'] is True, payload
    assert resolved and resolved[0].payload['approved'] is False
    assert resolved[0].payload['reason'] == 'run stopped', resolved[0].payload
    assert len(llm.calls) == 1, f'run must not continue past the stop: {len(llm.calls)} calls'
    assert orchify_broker.pending_approvals() == [] or all(
        a.tool_name != 'danger' for a in orchify_broker.pending_approvals())
    print('ok  control:stop while awaiting approval ends the run cleanly')


def test_hook_errors_are_isolated():
    '''
    A broken hook (no __name__, raising) must neither crash the run nor the
    broker's hook-dispatch bookkeeping.
    '''
    class Boom:
        def __call__(self, event):
            raise RuntimeError('boom')

    boom = orchify_broker.register_hook('*', Boom(), owner='')
    import functools
    partial_hook = functools.partial(lambda event, tag: None, tag='x')
    part = orchify_broker.register_hook('*', partial_hook, owner='')

    llm = FakeLLM([[final('survived')]])
    rec = Recorder()
    agent = Agent(name='hookfail', llm=llm)
    agent.run('go', max_steps=1)
    payload = finish_payload(rec)
    rec.close()
    boom.disposed = True
    part.disposed = True

    assert payload, 'run must finish despite the broken hook'
    assert not rec.of('agent:abort'), rec.types()
    print('ok  broken/unnamed hooks are isolated')


def test_timeout_env_defaults():
    '''pause/approval timeouts fall back to env, but explicit args win.'''
    os.environ['ORCHIFY_PAUSE_TIMEOUT'] = '0.25'
    os.environ['ORCHIFY_APPROVAL_TIMEOUT'] = '0.5'
    try:
        a = Agent(name='envtimeout', llm=FakeLLM([[]]))
        assert a.pause_timeout == 0.25, a.pause_timeout
        assert a.approval_timeout == 0.5, a.approval_timeout
        b = Agent(name='explicit', llm=FakeLLM([[]]), pause_timeout=9, approval_timeout=7)
        assert b.pause_timeout == 9 and b.approval_timeout == 7
        os.environ['ORCHIFY_PAUSE_TIMEOUT'] = 'nonsense'
        c = Agent(name='badenv', llm=FakeLLM([[]]))
        assert c.pause_timeout is None, c.pause_timeout
    finally:
        os.environ.pop('ORCHIFY_PAUSE_TIMEOUT', None)
        os.environ.pop('ORCHIFY_APPROVAL_TIMEOUT', None)
    print('ok  pause/approval timeouts honour env with explicit-arg precedence')


def test_run_id_is_the_execution_handle():
    '''
    run_id identifies one execution: run() returns it, every event carries it,
    and turn_id is the step-scoped label '<run_id>#<turn>'.
    '''
    llm = FakeLLM([
        [final('', [tool_call('noop', '{}')])],
        [final('done')],
    ])
    noop = Tool(func=lambda: 'noop-ran', name='noop', description='noop')
    rec = Recorder()
    agent = Agent(name='identity', llm=llm, tools=[noop])

    run_id = agent.run('go', max_steps=2)
    payload = finish_payload(rec)
    rec.close()

    assert run_id, 'run() must return the run handle'
    assert agent.last_run_id == run_id, agent.last_run_id
    assert payload['total_steps'] == 2, payload

    for event in rec.events:
        assert event.run_id == run_id, (event.event_type, event.run_id, run_id)
        assert event.turn_id == f'{run_id}#{event.turn}', \
            (event.event_type, event.turn_id, event.turn)

    steps = sorted({e.turn for e in rec.events})
    assert steps == [1, 2], steps
    assert rec.of('run:next')[0].turn_id == f'{run_id}#2'
    assert run_id not in orchify_broker.run_contexts, orchify_broker.run_contexts
    print('ok  run_id is the execution handle; turn_id is <run_id>#<turn>')


def test_feedback_addressing_and_run_id_aliases():
    '''
    Feedback addressed at a run_id works; an unknown run returns False; a
    turn_id resolves to its run; and a caller-supplied run_id (or the legacy
    turn_id alias) becomes the handle of the execution.
    '''
    llm = FakeLLM([[{'content': 'a', 'delay': 0.05}, final('done')]])
    rec = Recorder()
    resolved = []

    def pause_by_turn(event):
        resolved.append(orchify_broker.run_id_of_turn(event.turn_id))
        if rec.paused:
            return
        rec.paused = True
        assert orchify_broker.feedback('control:pause', turn_id=event.turn_id) is True
        assert orchify_broker.feedback('control:pause', run_id='nope#1') is False
        assert orchify_broker.feedback('control:pause', turn_id='ghost#9') is False

    rec.paused = False
    rec.callbacks['agent:answer'] = pause_by_turn
    rec.callbacks['run:paused'] = lambda e: orchify_broker.feedback('control:resume', run_id=e.run_id)

    agent = Agent(name='addressing', llm=llm)
    run_id = agent.run('go', max_steps=1)
    payload = finish_payload(rec)
    rec.close()

    assert resolved and all(r == run_id for r in resolved), (resolved, run_id)
    assert orchify_broker.run_id_of_turn('ghost#9') is None
    assert orchify_broker.run_id_of_turn(f'{run_id}#1') is None, 'turn index must not outlive the run'

    a = Agent(name='explicit', llm=FakeLLM([[final('x')]]))
    assert a.run('go', max_steps=1, run_id='my-run-1') == 'my-run-1'
    b = Agent(name='legacy', llm=FakeLLM([[final('y')]]))
    assert b.run('go', max_steps=1, turn_id='legacy-1') == 'legacy-1'
    assert payload['total_steps'] == 1
    print('ok  feedback addressing, unknown-run refusal, run_id/turn_id aliases')


def test_group_keeps_every_member_run_id():
    '''
    A group run has its own run_id while every member keeps its own, so a single
    member can still be targeted after a fan-out.
    '''
    from orchify import Group

    a1 = Agent(name='g1', llm=FakeLLM([[final('one')]]))
    a2 = Agent(name='g2', llm=FakeLLM([[final('two')]]))
    group = Group(name='panel', agents=[a1, a2], mode='voting')

    group_run_id = group.run('decide')
    assert group_run_id.startswith('panel#'), group_run_id
    assert group.last_run_id == group_run_id
    assert len(group.last_run_ids) == 2, group.last_run_ids
    assert len(set(group.last_run_ids)) == 2, 'members must not share a run_id'
    assert group_run_id not in group.last_run_ids

    end = [e for e in group.broker.events if e.event_type == 'end'][0]
    assert end.run_id == group_run_id, end.run_id
    assert end.payload['agent_run_ids'] == group.last_run_ids
    assert end.turn_id == f'{group_run_id}#1'

    for member_id in group.last_run_ids:
        assert wait_until(lambda mid=member_id: mid not in orchify_broker.run_contexts), \
            f'member run {member_id} never finished'
    print('ok  group run_id is separate from every member run_id')


def test_round_robin_mode_removed():
    '''The round_robin mode is gone: asking for it must fail loudly, not silently.'''
    from orchify import Group

    a = Agent(name='rr_a', llm=FakeLLM([[final('x')]]))
    try:
        Group(name='chain', agents=[a], mode='round_robin')
    except ValueError as exc:
        assert 'round_robin' in str(exc) and 'removed' in str(exc), exc
    else:
        raise AssertionError("mode='round_robin' must raise ValueError")

    for mode in ('broadcast', 'manager_delegation', 'voting', 'competition'):
        member = Agent(name=f'm_{mode}', llm=FakeLLM([[final('x')]]))
        group = Group(name=f'g_{mode}', agents=[member], mode=mode)
        assert group.mode == mode
    print('ok  round_robin mode removed; broadcast/manager_delegation/voting/competition intact')


def test_broker_pool_bookkeeping():
    '''The pool exposes pending/resolved entries and rejects double decisions.'''
    approval = orchify_broker.request_approval('task#pool_bookkeeping', kind='tool', tool_name='t', args={'a': 1})
    assert approval.approval_id in [a.approval_id for a in orchify_broker.pending_approvals()]
    assert orchify_broker.pending_approvals('task#pool_bookkeeping') == [approval]
    assert orchify_broker.pending_approvals('other') == []
    assert orchify_broker.approve(approval.approval_id, reason='ok') is True
    assert orchify_broker.approve(approval.approval_id) is False, 'double decision must be rejected'
    assert orchify_broker.reject('missing-id') is False
    assert orchify_broker.pending_approvals('task#pool_bookkeeping') == []
    assert orchify_broker.release_approval(approval.approval_id) is True
    assert orchify_broker.get_approval(approval.approval_id) is None
    print('ok  approval pool bookkeeping')


def test_agent_run_guard_cancel_and_wait():
    '''
    One run per Agent by default: a second run() is refused instead of silently
    sharing `messages`, and cancel()/wait() give the caller control of the run.
    '''
    rec = Recorder()
    llm = FakeLLM([[{'content': 'hi', 'delay': 0.05}, final('never reached')]])
    agent = Agent(name='guard', llm=llm)

    rec.callbacks['agent:answer'] = lambda e: orchify_broker.feedback('control:pause', run_id=e.run_id)
    rec.callbacks['run:paused'] = lambda e: None

    run_id = agent.run('go', max_steps=1)
    assert wait_until(lambda: bool(rec.of('run:paused'))), rec.types()
    assert agent.is_running() is True

    try:
        agent.run('again', max_steps=1)
    except RuntimeError as exc:
        assert 'already running' in str(exc), exc
    else:
        raise AssertionError('a second run() must be refused while one is in flight')

    assert agent.cancel(run_id) is True
    assert agent.wait(run_id, timeout=5) is True
    payload = finish_payload(rec)
    rec.close()

    assert payload['stopped'] is True, payload['stopped']
    assert agent.is_running() is False, 'the run slot must be released when the run ends'
    assert agent.current_run_id == '', agent.current_run_id
    print('ok  a second run() is refused; cancel()/wait() end the run cleanly')


def test_allow_concurrent_runs_opt_in():
    '''allow_concurrent_runs=True restores overlapping runs for callers who want them.'''
    rec = Recorder()
    rec.callbacks['agent:answer'] = lambda e: orchify_broker.feedback('control:pause', run_id=e.run_id)
    rec.callbacks['run:paused'] = lambda e: None

    agent = Agent(
        name='concurrent',
        llm=FakeLLM([
            [{'content': 'a', 'delay': 0.05}, final('one')],
            [{'content': 'b', 'delay': 0.05}, final('two')],
        ]),
        allow_concurrent_runs=True,
    )

    first = agent.run('go', max_steps=1)
    assert wait_until(lambda: len(rec.of('run:paused')) >= 1), rec.types()
    second = agent.run('go again', max_steps=1)
    assert second and second != first
    assert wait_until(lambda: len(rec.of('run:paused')) >= 2), rec.types()

    orchify_broker.feedback('control:resume', run_id=first)
    orchify_broker.feedback('control:resume', run_id=second)
    assert wait_until(lambda: len(rec.of('run:finish')) >= 2), rec.types()
    rec.close()
    print('ok  allow_concurrent_runs=True permits overlapping runs')


def test_agent_wait_refuses_the_loop_thread():
    '''wait() must refuse to block the orchestrator loop thread (it would deadlock).'''
    import asyncio
    from orchify.backend import orchify_web_backend

    agent = Agent(name='loopwait', llm=FakeLLM([[final('x')]]))

    async def attempt():
        try:
            agent.wait('whatever', timeout=1)
        except RuntimeError as exc:
            return str(exc)
        return ''

    future = asyncio.run_coroutine_threadsafe(attempt(), orchify_web_backend._loop)
    message = future.result(timeout=5)
    assert 'deadlock' in message, message
    print('ok  wait() refuses to block the orchestrator loop thread')


def test_group_voting_tally_and_run_guard():
    '''
    Voting waits for its members, tallies `Vote`s (majority, normalised) and
    reports the winner; a second overlapping group run is refused.
    '''
    from orchify import Group, Vote

    a1 = Agent(name='v1', llm=FakeLLM([[final('Yes')]]))
    a2 = Agent(name='v2', llm=FakeLLM([[final('  yes ')]]))
    a3 = Agent(name='v3', llm=FakeLLM([[final('No')]]))
    group = Group(name='majority', agents=[a1, a2, a3], mode='voting')

    group_run_id = group.run('decide')
    assert group_run_id.startswith('majority#'), group_run_id
    assert [v.agent_name for v in group.last_votes] == ['v1', 'v2', 'v3'], group.last_votes
    assert all(isinstance(v, Vote) for v in group.last_votes)
    assert group.last_winner == 'v1', group.last_winner
    assert group.last_answer == 'Yes', group.last_answer
    assert group.is_running() is False

    vote_event = [e for e in group.broker.events if e.event_type == 'vote'][0]
    assert vote_event.payload['winner'] == 'v1', vote_event.payload
    end = [e for e in group.broker.events if e.event_type == 'end'][0]
    assert end.payload['winner'] == 'v1', end.payload
    assert end.payload['answer'] == 'Yes', end.payload
    assert len(end.payload['votes']) == 3, end.payload

    slow1 = Agent(name='s1', llm=FakeLLM([[{'content': 'x', 'delay': 0.4}, final('x')]]))
    slow2 = Agent(name='s2', llm=FakeLLM([[{'content': 'y', 'delay': 0.4}, final('y')]]))
    busy = Group(name='busy', agents=[slow1, slow2], mode='voting')

    worker = threading.Thread(target=lambda: busy.run('go'), daemon=True)
    worker.start()
    assert wait_until(lambda: busy.is_running(), 5), 'the group run never started'
    try:
        busy.run('again')
    except RuntimeError as exc:
        assert 'already running' in str(exc), exc
    else:
        raise AssertionError('a second group run() must be refused while one is in flight')
    worker.join(10)
    assert not worker.is_alive(), 'the first group run never finished'
    assert busy.is_running() is False

    nowait = Group(name='nowait', agents=[Agent(name='n1', llm=FakeLLM([[final('a')]]))],
                   mode='voting', vote_timeout=0)
    nowait.run('go')
    assert nowait.last_votes == [], nowait.last_votes
    assert orchify_broker.wait_run(nowait.last_run_ids[0], timeout=5) is True
    print('ok  voting tallies a majority; overlapping group runs are refused')


def test_run_done_handles_survive_and_reset():
    '''
    Run completion handles survive the run (a caller can still wait for a
    finished run), and a reused run_id gets a fresh handle - never the stale,
    already-set one that used to hang `Agent.wait()`.
    '''
    orchify_broker.start_task('unit#reuse', None)
    first = orchify_broker.run_done_event('unit#reuse')
    assert first is not None and first.is_set() is False
    assert orchify_broker.wait_run('unit#reuse', timeout=0.05) is False

    orchify_broker.start_task('unit#reuse', None)
    assert orchify_broker.run_done_event('unit#reuse') is first, 'the handle must stay stable'

    orchify_broker.finish_task('unit#reuse')
    assert first.is_set() is True
    assert orchify_broker.wait_run('unit#reuse', timeout=0.05) is True
    assert orchify_broker.is_run_active('unit#reuse') is False

    orchify_broker.start_task('unit#reuse', None)
    second = orchify_broker.run_done_event('unit#reuse')
    assert second is not first and second.is_set() is False
    orchify_broker.finish_task('unit#reuse')
    assert orchify_broker.wait_run('unit#reuse', timeout=0.05) is True

    assert orchify_broker.run_done_event('unit#nope') is None
    assert orchify_broker.wait_run('unit#nope', timeout=0.01) is False
    print('ok  run completion handles survive and reset on reuse')


def test_hooks_run_outside_the_emit_lock():
    '''A blocked hook must not block emit() bookkeeping or other hooks.'''
    from orchify.event import RuntimeEvent

    entered = threading.Event()
    release = threading.Event()
    seen = []

    def blocking(event):
        entered.set()
        release.wait(5)
        seen.append('blocked')

    def other(event):
        seen.append('other')

    b1 = orchify_broker.register_hook('test:blocked', blocking, owner='locktest')
    b2 = orchify_broker.register_hook('test:other', other, owner='locktest')

    def event(event_type):
        return RuntimeEvent(agent_name='locktest', agent_code='locktest',
                            turn_id='', event_type=event_type, payload={})

    worker = threading.Thread(target=lambda: orchify_broker.emit(event('test:blocked')), daemon=True)
    worker.start()
    try:
        assert entered.wait(5), 'the blocking hook never ran'
        start = time.time()
        orchify_broker.emit(event('test:other'))
        elapsed = time.time() - start
        assert elapsed < 1.0, f'emit() waited {elapsed:.2f}s for a blocked hook: hooks still run under the lock'
        assert 'other' in seen, seen
    finally:
        release.set()
        worker.join(5)
        orchify_broker.remove_hooks('locktest')
    assert 'blocked' in seen, seen
    print('ok  hooks run outside the emit lock (a slow hook cannot stall emit)')


def test_http_timeout_default_and_env():
    '''The HTTP client is no longer timeout-less, and ORCHIFY_HTTP_TIMEOUT can override it.'''
    import httpx
    from orchify.backend import DEFAULT_HTTP_TIMEOUT, _read_http_timeout, orchify_web_backend

    def shape(timeout):
        return None if timeout is None else (timeout.connect, timeout.read, timeout.write, timeout.pool)

    os.environ.pop('ORCHIFY_HTTP_TIMEOUT', None)
    assert shape(_read_http_timeout()) == shape(DEFAULT_HTTP_TIMEOUT)
    assert isinstance(orchify_web_backend.timeout, httpx.Timeout), 'the shared client must have a timeout'

    os.environ['ORCHIFY_HTTP_TIMEOUT'] = '0'
    assert _read_http_timeout() is None, 'explicit 0 must mean "no timeout"'
    os.environ['ORCHIFY_HTTP_TIMEOUT'] = '42'
    assert shape(_read_http_timeout()) == (42.0, 42.0, 42.0, 42.0), _read_http_timeout()
    os.environ['ORCHIFY_HTTP_TIMEOUT'] = 'nonsense'
    assert shape(_read_http_timeout()) == shape(DEFAULT_HTTP_TIMEOUT)
    os.environ.pop('ORCHIFY_HTTP_TIMEOUT', None)
    print('ok  HTTP timeout has a default and honours ORCHIFY_HTTP_TIMEOUT')


def test_llm_retry_helpers():
    '''429/5xx/transport errors are retryable, 4xx are not, and backoff honours Retry-After.'''
    import httpx
    from orchify.llm.openai import RETRYABLE_STATUS, _is_retryable, _read_int_env, _retry_delay

    def status_error(code, headers=None):
        request = httpx.Request('POST', 'https://example.invalid/v1/chat/completions')
        response = httpx.Response(code, headers=headers or {}, request=request)
        return httpx.HTTPStatusError(f'HTTP {code}', request=request, response=response)

    assert 429 in RETRYABLE_STATUS and 503 in RETRYABLE_STATUS
    assert _is_retryable(status_error(429)) is True
    assert _is_retryable(status_error(503)) is True
    assert _is_retryable(status_error(400)) is False
    assert _is_retryable(status_error(401)) is False
    assert _is_retryable(httpx.ConnectError('boom')) is True
    assert _is_retryable(ValueError('nope')) is False

    os.environ.pop('ORCHIFY_LLM_RETRY_BACKOFF', None)
    assert _retry_delay(0, ValueError('x')) == 0.5
    assert _retry_delay(1, ValueError('x')) == 1.0
    assert _retry_delay(10, ValueError('x')) == 30.0, 'the backoff must be capped'
    assert _retry_delay(0, status_error(429, {'Retry-After': '2'})) == 2.0

    assert _read_int_env('ORCHIFY_LLM_MAX_RETRIES', 2) == 2
    os.environ['ORCHIFY_LLM_MAX_RETRIES'] = '5'
    assert _read_int_env('ORCHIFY_LLM_MAX_RETRIES', 2) == 5
    os.environ['ORCHIFY_LLM_MAX_RETRIES'] = 'junk'
    assert _read_int_env('ORCHIFY_LLM_MAX_RETRIES', 2) == 2
    os.environ.pop('ORCHIFY_LLM_MAX_RETRIES', None)
    print('ok  retry classification, Retry-After backoff and env knobs')


def main():
    '''Run every test in this file.

    Returns:
        int: 0 when all tests passed, 1 when any failed.
    '''
    here = os.path.dirname(os.path.abspath(__file__))
    this = os.path.abspath(__file__)
    tests = [
        test_configure_with_tools_is_idempotent_and_visible,
        test_as_tool_and_description,
        test_tool_executor_env_and_laziness,
        test_pause_mid_stream_and_lost_resume_race,
        test_pause_on_last_step,
        test_final_checkpoint_after_last_tool_step,
        test_stop_while_paused_terminates,
        test_pause_timeout_self_resumes,
        test_approval_approved_executes_tool,
        test_approval_rejected_blocks_tool,
        test_approval_timeout_fails_closed,
        test_stop_while_awaiting_approval_ends_run,
        test_agent_level_approval_policy,
        test_callable_approval_policy,
        test_update_messages_tolerates_bad_payload,
        test_hook_errors_are_isolated,
        test_timeout_env_defaults,
        test_broker_pool_bookkeeping,
        test_run_id_is_the_execution_handle,
        test_feedback_addressing_and_run_id_aliases,
        test_group_keeps_every_member_run_id,
        test_round_robin_mode_removed,
        test_agent_run_guard_cancel_and_wait,
        test_allow_concurrent_runs_opt_in,
        test_agent_wait_refuses_the_loop_thread,
        test_group_voting_tally_and_run_guard,
        test_run_done_handles_survive_and_reset,
        test_hooks_run_outside_the_emit_lock,
        test_http_timeout_default_and_env,
        test_llm_retry_helpers,
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
