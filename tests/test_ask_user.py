import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, Tool, tool, orchify_broker, orchify_manager
from test_pause_approval import FakeLLM, Recorder, final, tool_call, wait_until, wait_finish


def test_ask_user_tool_shape():
    '''enable_ask_user registers an ask_question tool with a required question.'''
    agent = Agent(name='qs', llm=FakeLLM([[]]), enable_ask_user=True)

    assert 'ask_user' in agent.tools, list(agent.tools)
    tool_obj = agent.tools['ask_user']
    assert tool_obj.ask_question is True
    assert tool_obj.require_approval is False
    info = tool_obj.build_info()['function']
    assert info['parameters']['required'] == ['question'], info['parameters']
    assert info['parameters']['properties']['question']['type'] == 'string'

    fresh = agent.ask_user_tool()
    assert fresh.ask_question is True and fresh.name == 'ask_user'

    @tool(name='ask_decorated', ask_question=True)
    def ask_decorated(question: str) -> str:
        '''Ask something.'''
        return question

    assert ask_decorated.ask_question is True
    plain = Agent(name='qs2', llm=FakeLLM([[]]))
    assert 'ask_user' not in plain.tools
    print('ok  ask_user tool shape and registration')


def test_ask_user_parks_and_answers():
    '''The run parks on the question and continues with the human answer.'''
    llm = FakeLLM([
        [final('', [tool_call('ask_user', '{"question": "Which env?"}')])],
        [final('deployed to prod')],
    ])
    rec = Recorder()
    seen = {}

    def answer(event):
        seen['payload'] = dict(event.payload)
        seen['kind'] = getattr(event, 'kind', None)
        assert orchify_broker.answer_approval(event.payload['approval_id'], 'prod') is True

    rec.callbacks['run:approval:required'] = answer

    agent = Agent(name='q1', llm=llm, enable_ask_user=True)
    agent.run('deploy it', max_steps=3)
    payload = wait_finish(rec).payload
    types = rec.types()
    rec.close()

    assert 'run:approval:required' in types and 'run:approval:resolved' in types, types
    assert seen['kind'] == 'question', seen
    assert seen['payload']['kind'] == 'question', seen['payload']
    assert seen['payload']['tool_name'] == 'ask_user'
    assert seen['payload']['args'] == {'question': 'Which env?'}, seen['payload']['args']

    resolved = rec.of('run:approval:resolved')[0].payload
    assert resolved['approved'] is True, resolved
    assert resolved['answer'] == 'prod', resolved
    assert resolved['reason'] == 'prod', resolved
    assert resolved['timed_out'] is False, resolved

    finished = [e for e in rec.of('tool:call:finish') if e.tool_name == 'ask_user']
    assert finished, [e.event_type for e in rec.events]
    assert finished[0].payload['result'] == 'prod', finished[0].payload
    assert finished[0].payload['answered'] is True
    assert finished[0].payload['timed_out'] is False
    assert finished[0].payload['question'] == 'Which env?', finished[0].payload

    assert payload['answer'] == 'deployed to prod', payload
    assert agent.last_answer == 'deployed to prod'
    second = llm.calls[1]['messages']
    assert any(m.get('role') == 'tool' and m.get('content') == 'prod' for m in second), second
    print('ok  ask_user parks the run and feeds the human answer back')


def test_ask_user_through_the_manager():
    '''Manager questions()/status() expose the question and answer() releases it.'''
    llm = FakeLLM([
        [final('', [tool_call('ask_user', '{"question": "How many?"}')])],
        [final('made 3')],
    ])
    rec = Recorder()
    holder = {}

    def hold(event):
        holder['approval_id'] = event.payload['approval_id']
        holder['run_id'] = event.run_id

    rec.callbacks['run:approval:required'] = hold

    agent = Agent(name='q2', llm=llm, enable_ask_user=True)
    run_id = agent.run('how many', max_steps=3)
    assert wait_until(lambda: bool(holder.get('approval_id'))), 'no question was parked'

    pending = orchify_manager.questions(pending_only=True, run_id=run_id)
    assert len(pending) == 1, pending
    assert pending[0]['kind'] == 'question', pending[0]
    assert pending[0]['args'] == {'question': 'How many?'}, pending[0]['args']
    assert orchify_manager.run(run_id)['status'] == 'awaiting_approval'
    assert orchify_manager.status()['counts']['pending_questions'] >= 1

    assert orchify_manager.answer(holder['approval_id'], '3') is True
    assert orchify_manager.answer(holder['approval_id'], 'later') is False, \
        'a decided question is final'

    payload = wait_finish(rec).payload
    rec.close()

    assert payload['answer'] == 'made 3', payload
    answered = orchify_manager.questions(run_id=run_id)
    assert answered, 'the decided question must survive the run'
    assert answered[0]['answer'] == '3', answered[0]
    assert answered[0]['resolved'] is True
    assert orchify_manager.questions(pending_only=True) == []
    print('ok  manager exposes and answers open questions')


def test_ask_user_without_an_answer():
    '''A question nobody answers degrades to "[no answer]", it does not abort.'''
    llm = FakeLLM([
        [final('', [tool_call('ask_user', '{"question": "Which env?"}')])],
        [final('moved on')],
    ])
    rec = Recorder()

    agent = Agent(name='q3', llm=llm, enable_ask_user=True, approval_timeout=0.2)
    agent.run('go', max_steps=3)
    payload = wait_finish(rec).payload
    rec.close()

    finished = [e for e in rec.of('tool:call:finish') if e.tool_name == 'ask_user']
    assert finished, 'the model must still get a tool result'
    assert finished[0].payload['answered'] is False, finished[0].payload
    assert finished[0].payload['timed_out'] is True, finished[0].payload
    assert 'no answer' in finished[0].payload['result'], finished[0].payload

    resolved = rec.of('run:approval:resolved')[0].payload
    assert resolved['approved'] is False and resolved['timed_out'] is True, resolved
    assert payload['stopped'] is False, payload
    assert payload['answer'] == 'moved on', payload
    assert orchify_broker.pending_questions() == []
    print('ok  an unanswered question degrades to "[no answer]"')


def test_ask_user_stop_while_parked():
    '''control:stop reaches a run parked on a question and ends it cleanly.'''
    llm = FakeLLM([
        [final('', [tool_call('ask_user', '{"question": "Which env?"}')])],
        [final('never reached')],
    ])
    rec = Recorder()
    rec.callbacks['run:approval:required'] = lambda e: orchify_broker.feedback(
        'control:stop', run_id=e.run_id
    )

    agent = Agent(name='q4', llm=llm, enable_ask_user=True)
    agent.run('go', max_steps=3)
    payload = wait_finish(rec).payload
    rec.close()

    assert payload['stopped'] is True, payload
    assert orchify_broker.pending_questions() == [], 'a stopped run must not leak a question'
    assert not [e for e in rec.of('tool:call:finish') if e.tool_name == 'ask_user'], \
        'the parked call must not report a result'
    print('ok  a stop interrupts a parked question')


def test_ask_user_tool_function_is_never_executed():
    '''ask_question=True intercepts the call; the tool function itself is not run.'''
    calls = []
    custom = Tool(func=lambda question: calls.append(question) or 'executed',
                  name='ask_user', description='ask the user',
                  parameters=[{
                      'name': 'question', 'kind': 'positional or keyword',
                      'default': 'N/A', 'annotation': 'str',
                      'description': 'the question', 'required': True,
                  }],
                  ask_question=True)

    llm = FakeLLM([
        [final('', [tool_call('ask_user', '{"question": "Why?"}')])],
        [final('because')],
    ])
    rec = Recorder()
    rec.callbacks['run:approval:required'] = lambda e: orchify_broker.answer_approval(
        e.payload['approval_id'], 'because I said so'
    )

    agent = Agent(name='q5', llm=llm, tools=[custom])
    agent.run('why', max_steps=3)
    wait_finish(rec)
    rec.close()

    assert calls == [], calls
    finished = rec.of('tool:call:finish')[0]
    assert finished.payload['result'] == 'because I said so', finished.payload
    print('ok  ask_question intercepts the call before execution')


def test_broker_question_helpers():
    '''ask_question/answer_approval/pending_questions behaviour at the broker level.'''
    assert orchify_broker.answer_approval('a-does-not-exist', 'x') is False

    approval = orchify_broker.ask_question('unit#q', 'why?', step=2)
    assert approval.kind == 'question'
    assert approval.args == {'question': 'why?'}, approval.args
    assert approval.answer == ''
    assert approval.decision is None

    pending = orchify_broker.pending_questions('unit#q')
    assert [item.approval_id for item in pending] == [approval.approval_id], pending
    assert orchify_broker.pending_questions('other#run') == []

    assert orchify_broker.answer_approval(approval.approval_id, 'because') is True
    assert orchify_broker.answer_approval(approval.approval_id, 'again') is False
    stored = orchify_broker.get_approval(approval.approval_id)
    assert stored.answer == 'because', stored
    assert stored.approved is True and stored.resolved is True
    assert stored.decision is True
    assert orchify_broker.pending_questions('unit#q') == []

    assert orchify_broker.release_approval(approval.approval_id) is True
    assert orchify_broker.get_approval(approval.approval_id) is None
    print('ok  broker question helpers')


def main():
    '''Run every test in this file, timing each one.

    Returns:
        int: 0 when all tests passed, 1 when any failed.
    '''
    tests = [
        test_ask_user_tool_shape,
        test_ask_user_parks_and_answers,
        test_ask_user_through_the_manager,
        test_ask_user_without_an_answer,
        test_ask_user_stop_while_parked,
        test_ask_user_tool_function_is_never_executed,
        test_broker_question_helpers,
    ]
    failed = 0
    for test in tests:
        started = time.time()
        try:
            test()
        except Exception as exc:
            failed += 1
            import traceback
            print(f'FAIL {test.__name__}: {exc}')
            traceback.print_exc()
        else:
            print(f'     {test.__name__} ({time.time() - started:.2f}s)')
    if failed:
        print(f'{failed} of {len(tests)} tests failed')
        return 1
    print(f'all {len(tests)} tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
