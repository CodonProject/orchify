import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import orchify
from orchify import (Agent, Group, Tool, orchify_broker, orchify_manager,
                     configure, config, console, off, on, RunAlreadyActive,
                     RunFailed, RunTimeout, UnknownRun, WaitOnLoopThread,
                     OrchifyError)
from orchify.broker import HookBinding
from orchify.gate import gate_stats, reset_gates
from orchify.llm import LLMInterface
from orchify.testing import FakeLLM, Recorder, final, tool_call, wait_finish, wait_until


class BoomLLM(LLMInterface):
    '''An LLM whose provider call always fails.'''

    async def _core_request(self, messages=None, model=None, tools=None, **kwargs):
        '''Raise like a broken provider would.'''
        raise RuntimeError('provider exploded')
        yield None


def test_agent_ask_returns_the_answer_and_reset_forgets_it():
    '''ask() is run()+wait()+answer in one call, and reset() clears the history.'''
    llm = FakeLLM([[final('first answer')], [final('second answer')]])
    agent = Agent(name='erg-ask', llm=llm, system_prompt='You are terse.')

    assert agent.ask('hi') == 'first answer'
    assert agent.last_answer == 'first answer'
    assert agent.last_run_id and not agent.is_running()

    assert agent.ask('again') == 'second answer', 'a second ask() must work on the same agent'
    assert wait_until(lambda: not agent.is_running()), 'the run slot must be released once ask() returns'

    agent.messages.append({'role': 'user', 'content': 'stale'})
    agent.reset()
    assert len(agent.messages) == 1 and agent.messages[0]['role'] == 'system'
    assert agent.messages[0]['content'] == 'You are terse.'
    assert agent.last_answer == '' and agent.last_error == '' and agent.last_run_id == ''
    assert agent.ask('third') == 'second answer', 'the agent must keep working after reset()'
    print('ok  Agent.ask() answer round-trip + reset()')


def test_agent_from_env():
    '''from_env() builds a working agent from the environment.'''
    agent = Agent.from_env('erg-env', llm=FakeLLM([[final('env answer')]]))
    assert agent.name == 'erg-env'
    assert agent.model == (os.getenv('MODEL_ID') or 'gpt-4o')
    assert agent.ask('hi') == 'env answer'
    print('ok  Agent.from_env()')


def test_agent_ask_timeout_and_request_errors():
    '''ask() reports a missed deadline and a failed run with typed errors.'''
    slow = Tool(func=lambda: time.sleep(0.6) or 'slow done', name='erg-slow', description='slow')
    agent = Agent(
        name='erg-timeout',
        llm=FakeLLM([[final('', [tool_call('erg-slow', '{}')])], [final('never reached')]]),
        tools=[slow],
    )
    started = time.time()
    try:
        agent.ask('go', timeout=0.15, max_steps=3)
    except RunTimeout as exc:
        assert isinstance(exc, TimeoutError) and isinstance(exc, OrchifyError)
        assert exc.run_id and 'erg-timeout' in str(exc)
    else:
        raise AssertionError('ask() must raise RunTimeout when the deadline passes')
    assert time.time() - started < 3
    assert wait_until(lambda: not agent.is_running(), timeout=8), \
        'a timed-out ask() must ask the run to stop'

    failing = Agent(name='erg-boom', llm=BoomLLM())
    try:
        failing.ask('go')
    except RunFailed as exc:
        assert 'provider exploded' in exc.error and exc.run_id
        assert not isinstance(exc, (RunTimeout, RunAlreadyActive))
    else:
        raise AssertionError('ask() must raise RunFailed when the run aborts')
    assert 'provider exploded' in failing.last_error
    print('ok  ask() timeout -> RunTimeout, provider failure -> RunFailed')


def test_agent_run_guard_raises_run_already_active():
    '''A second overlapping run raises RunAlreadyActive, which is a RuntimeError.'''
    agent = Agent(
        name='erg-guard',
        llm=FakeLLM([[{'content': 'slow', 'delay': 0.6}, final('done')], [final('second')]]),
    )
    run_id = agent.run('go', max_steps=2)
    try:
        agent.run('again')
    except RunAlreadyActive as exc:
        assert isinstance(exc, RuntimeError) and isinstance(exc, OrchifyError)
        assert exc.run_id == run_id and exc.owner == 'erg-guard'
        assert 'already running' in str(exc)
    else:
        raise AssertionError('the run guard must raise RunAlreadyActive')
    assert wait_until(lambda: not agent.is_running(), timeout=8), \
        'the first run must end before the next ask()'
    assert agent.ask('waited') == 'second', 'ask() must work once the parked run ended'
    print('ok  RunAlreadyActive guard and error hierarchy')


def test_agent_async_api():
    '''aask() and arun() await the run from any thread.'''
    llm = FakeLLM([[final('async answer')], [final('second answer')]])
    agent = Agent(name='erg-async', llm=llm)

    assert asyncio.run(agent.aask('hi')) == 'async answer'
    run_id = asyncio.run(agent.arun('again'))
    assert run_id == agent.last_run_id and agent.last_answer == 'second answer'
    print('ok  Agent.aask()/Agent.arun()')


def test_group_ask_modes_and_reset():
    '''Group.ask() resolves the member answer for every mode.'''
    group = Group(
        name='erg-group',
        agents=[Agent(name='erg-boss', llm=FakeLLM([[final('boss answer')]]))],
        mode='broadcast',
    )
    assert group.ask('task', timeout=8) == 'boss answer'
    assert group.last_run_id and not group.is_running()

    voting = Group(
        name='erg-vote',
        agents=[
            Agent(name='erg-v1', llm=FakeLLM([[final('Yes')]])),
            Agent(name='erg-v2', llm=FakeLLM([[final('  yes ')]])),
            Agent(name='erg-v3', llm=FakeLLM([[final('No')]])),
        ],
        mode='voting',
    )
    assert voting.ask('task', timeout=8) == 'Yes'
    assert voting.last_winner == 'erg-v1' and len(voting.last_votes) == 3

    voting.reset()
    assert voting.last_run_ids == [] and voting.last_votes == []
    assert voting.last_winner is None and voting.last_answer == ''
    assert voting.last_run_id == '' and voting.current_run_id == ''

    async_group = Group(
        name='erg-gasync',
        agents=[Agent(name='erg-ga', llm=FakeLLM([[final('group async answer')]]))],
    )
    assert asyncio.run(async_group.aask('task', timeout=8)) == 'group async answer'
    run_id = asyncio.run(
        Group(
            name='erg-garun',
            agents=[Agent(name='erg-gb', llm=FakeLLM([[final('another')]]))],
        ).arun('task', timeout=8)
    )
    assert run_id.startswith('erg-garun#')
    print('ok  Group.ask()/aask()/arun()/reset()')


def test_group_wait_errors_are_typed():
    '''Group.wait() reports missing runs and loop-thread blocking with types.'''
    group = Group(name='erg-gwait', agents=[Agent(name='erg-gw', llm=FakeLLM([[]]))])
    try:
        group.wait()
    except UnknownRun as exc:
        assert isinstance(exc, ValueError) and exc.owner == 'erg-gwait'
    else:
        raise AssertionError('wait() must raise UnknownRun without a run')

    async def attempt():
        try:
            group.wait('erg-gwait#1', timeout=0.2)
        except WaitOnLoopThread as exc:
            return str(exc)
        return ''

    from orchify.backend import orchify_web_backend
    text = asyncio.run_coroutine_threadsafe(attempt(), orchify_web_backend._loop).result(timeout=5)
    assert 'deadlock' in text, 'waiting on the loop thread must be refused'
    print('ok  Group.wait() UnknownRun / WaitOnLoopThread')


def test_on_and_off_subscriptions():
    '''on()/off() register application hooks without the plugin owner dance.'''
    seen = []
    binding = on('agent:answer', lambda event: seen.append(event.event_type), owner='erg-hook')
    assert isinstance(binding, HookBinding)

    agent = Agent(name='erg-hook-agent',
                  llm=FakeLLM([[{'content': 'chatter'}, final('hooked answer')]]))
    assert agent.ask('hi') == 'hooked answer'
    assert seen == ['agent:answer'], \
        'a hook owned by the application must keep firing (no plugin filter)'

    assert off(binding) is True
    assert off(binding) is False
    assert agent.ask('hi') == 'hooked answer'
    assert seen == ['agent:answer'], 'a removed hook must not fire again'

    counts = []
    decorated = on('run:start')(lambda event: counts.append(event.run_id))
    assert isinstance(decorated, HookBinding)
    agent.ask('hi')
    assert len(counts) == 1
    assert off(decorated) is True

    on('run:start', lambda event: None, owner='erg-owner')
    assert off('erg-owner') is True
    assert off('erg-owner') is False

    def handler(event):
        return None

    on('run:start', handler)
    assert off(handler) is True, 'off() must accept the callback itself'
    try:
        off(1234)
    except TypeError as exc:
        assert 'HookBinding' in str(exc)
    else:
        raise AssertionError('off() must reject unknown targets')
    print('ok  orchify.on()/off()')


def test_configure_and_config():
    '''configure() writes the environment and refreshes the runtime.'''
    import orchify.broker as broker_module

    names = ('ORCHIFY_MAX_CALL_DEPTH', 'ORCHIFY_LLM_RPM', 'ORCHIFY_MANAGER_HISTORY',
             'ORCHIFY_LOG_ECHO', 'ORCHIFY_POLL_INTERVAL', 'ORCHIFY_TOOL_CACHE_SIZE')
    original = {name: os.environ.get(name) for name in names}
    try:
        settings = configure(
            max_call_depth=3,
            llm_rpm=60,
            manager_history=7,
            log_echo=False,
            poll_interval=0.012,
            tool_cache_size=8,
        )
        assert settings.max_call_depth == 3 and settings.tool_cache_size == 8
        assert os.environ['ORCHIFY_MAX_CALL_DEPTH'] == '3'
        assert os.environ['ORCHIFY_LOG_ECHO'] == '0'
        assert abs(broker_module.POLL_INTERVAL - 0.012) < 1e-9, 'POLL_INTERVAL must be refreshed'
        assert gate_stats()['llm_rpm']['per_minute'] == 60, 'gates must be rebuilt from the env'
        assert config().max_call_depth == 3
        assert config().resolved()['max_call_depth']['value'] == 3
        assert config().resolved()['max_call_depth']['env'] == 'ORCHIFY_MAX_CALL_DEPTH'

        try:
            configure(not_a_setting=1)
        except TypeError as exc:
            assert 'not_a_setting' in str(exc)
        else:
            raise AssertionError('configure() must reject unknown settings')
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        reset_gates()
        configure()
    print('ok  configure() applies, refreshes gates and rejects unknown settings')


def test_console_loop_handles_approvals_and_questions():
    '''console() answers tool approvals and open questions from a script.'''
    danger = Tool(func=lambda x: f'ran:{x}', name='erg-danger',
                  description='dangerous', require_approval=True)
    agent = Agent(
        name='erg-console',
        llm=FakeLLM([
            [final('', [tool_call('erg-danger', '{"x": 1}')])],
            [final('approved and done')],
        ]),
        tools=[danger],
    )
    run_id = agent.run('go', max_steps=3)
    assert wait_until(lambda: bool(orchify.pending_approvals(run_id))), 'no approval to decide'

    prompts = []
    outputs = []
    loop = console(inp=lambda prompt: prompts.append(prompt) or 'y',
                   out=outputs.append, poll_interval=0.02)
    assert loop.running is False and loop.run_once() is True
    assert wait_until(lambda: not agent.is_running(), timeout=8)
    assert agent.last_answer == 'approved and done'
    assert 'approve?' in prompts[0] and 'erg-danger' in prompts[0]
    assert any('approved' in line for line in outputs)
    record = orchify.approvals(run_id=run_id)[0]
    assert record['approved'] is True and record['decision'] is True

    asking = Agent(
        name='erg-console-q',
        llm=FakeLLM([
            [final('', [tool_call('ask_user', '{"question": "Which env?"}')])],
            [final('deployed to prod')],
        ]),
        enable_ask_user=True,
    )
    question_run = asking.run('go', max_steps=3)
    assert wait_until(lambda: bool(orchify.questions(pending_only=True, run_id=question_run))), \
        'no question to answer'
    assert console(inp=lambda prompt: 'prod', out=outputs.append).run_once() is True
    assert wait_until(lambda: not asking.is_running(), timeout=8)
    assert asking.last_answer == 'deployed to prod'
    assert orchify.questions(pending_only=True, run_id=question_run) == []

    idle = console(inp=lambda prompt: '', out=outputs.append, poll_interval=0.05)
    assert idle.start().running is True
    assert idle.stop() is True and idle.running is False
    print('ok  console() approval + question handling and thread lifecycle')


def test_api_readers_and_report():
    '''The top-level readers expose the live and finished runs.'''
    agent = Agent(name='erg-readers', llm=FakeLLM([[final('reader answer')]]))
    assert agent.ask('hi') == 'reader answer'

    live = orchify.runs()
    assert any(item['run_id'] == agent.last_run_id or item['status'] == 'done' for item in live) or live == []
    record = orchify.run(agent.last_run_id)
    assert record is not None and record['kind'] == 'agent'
    history = orchify.history(limit=5)
    assert any(item['run_id'] == agent.last_run_id for item in history)
    state = orchify.status()
    assert 'counts' in state and state['counts']['pending_approvals'] >= 0
    text = orchify.report(history_limit=3, event_limit=3)
    assert 'rg-readers' in text and 'RUNS' in text
    assert orchify.approvals(run_id=agent.last_run_id) == []
    print('ok  runs()/run()/history()/status()/report()')


def test_testing_module_helpers():
    '''orchify.testing ships the fakes the library tests use.'''
    rec = Recorder()
    agent = Agent(name='erg-testing', llm=FakeLLM([[final('tested', [])]]))
    run_id = agent.run('hi', max_steps=2)
    event = wait_finish(rec)
    assert event.run_id == run_id and event.payload['answer'] == 'tested'
    assert rec.types()[0] == 'run:start'
    assert rec.of('run:finish') and rec.of('nope') == []
    rec.close()
    assert rec.of('run:finish'), 'closing the recorder must not drop captured events'
    print('ok  orchify.testing FakeLLM/Recorder helpers')


def main():
    '''Run every ergonomics test in this file.'''
    tests = [
        test_agent_ask_returns_the_answer_and_reset_forgets_it,
        test_agent_from_env,
        test_agent_ask_timeout_and_request_errors,
        test_agent_run_guard_raises_run_already_active,
        test_agent_async_api,
        test_group_ask_modes_and_reset,
        test_group_wait_errors_are_typed,
        test_on_and_off_subscriptions,
        test_configure_and_config,
        test_console_loop_handles_approvals_and_questions,
        test_api_readers_and_report,
        test_testing_module_helpers,
    ]
    for test in tests:
        test()
    print(f'all {len(tests)} tests passed')


if __name__ == '__main__':
    main()
