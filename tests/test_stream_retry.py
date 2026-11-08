import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify import Agent
from orchify.errors import RunFailed
from orchify.llm.base import Chunk, FinalStatus, LLMInterface, Response
from orchify.testing import Recorder


class FlakyStream(LLMInterface):
    '''A backend that replays scripted streams, raising where the script says so.'''

    def __init__(self, attempts):
        '''Store the attempt scripts.

        Args:
            attempts (list): One script per request; each script is a list of
                ``('answer'|'cot', text)`` items and optional ``('raise', message)``.
        '''
        super().__init__()
        self.attempts = [list(script) for script in attempts]
        self.calls = 0

    async def _core_request(self, **kwargs):
        '''Yield the scripted chunks for the current attempt.

        Args:
            **kwargs (Any): Request parameters (ignored).
        '''
        index = min(self.calls, len(self.attempts) - 1)
        script = self.attempts[index]
        self.calls += 1
        answer = ''
        cot = ''
        for item in script:
            if item[0] == 'raise':
                raise RuntimeError(item[1])
            is_cot = item[0] == 'cot'
            text = item[1]
            if is_cot:
                cot += text
            else:
                answer += text
            yield Response(current_chunk=Chunk(is_cot=is_cot, content=text,
                                               total_content=answer,
                                               total_cot_content=cot))
        yield Response(current_chunk=None,
                       final_status=FinalStatus(content=answer, reasoning=cot),
                       is_final=True)


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


def _agent(llm):
    '''An agent over the given backend.'''
    return Agent(name='streamer', llm=llm, system_prompt='sys')


def test_mid_stream_failure_is_replayed_without_duplicates():
    '''A stream that dies mid-answer is retried and the shown prefix is not repeated.'''
    llm = FlakyStream([
        [('answer', 'Hel'), ('answer', 'lo '), ('raise', 'connection reset')],
        [('answer', 'Hel'), ('answer', 'lo '), ('answer', 'world')],
    ])
    agent = _agent(llm)
    rec = Recorder()
    try:
        answer = agent.ask('say hello', timeout=8)
        assert answer == 'Hello world'
        assert llm.calls == 2, 'the failing attempt must be retried once'
    finally:
        rec.close()

    emitted = [event.content for event in rec.of('agent:answer')]
    assert emitted == ['Hel', 'lo ', 'world'], emitted
    print('ok  mid-stream retry replays the prefix without duplicating it')


def test_divergent_replay_fails_instead_of_splicing():
    '''A retry that is not a continuation of the shown text aborts the run.'''
    llm = FlakyStream([
        [('answer', 'Alpha'), ('raise', 'boom')],
        [('answer', 'Beta instead')],
    ])
    agent = _agent(llm)
    try:
        agent.ask('say alpha', timeout=8)
    except RunFailed as exc:
        assert 'diverged' in str(exc), str(exc)
        assert 'diverged' in exc.error, exc.error
    else:
        raise AssertionError('a divergent replay must not be spliced silently')
    assert llm.calls == 2
    print('ok  a divergent replay aborts with a clear error')


def test_stream_retries_can_be_disabled():
    '''ORCHIFY_STREAM_RETRIES=0 keeps the old behaviour: fail at the first error.'''
    env = _Env().set('ORCHIFY_STREAM_RETRIES', '0')
    try:
        llm = FlakyStream([
            [('answer', 'partial'), ('raise', 'connection reset')],
            [('answer', 'partial again')],
        ])
        agent = _agent(llm)
        try:
            agent.ask('say partial', timeout=8)
        except RunFailed as exc:
            assert 'connection reset' in exc.error
        else:
            raise AssertionError('retries are disabled, so the run must abort')
        assert llm.calls == 1
    finally:
        env.restore()
    print('ok  ORCHIFY_STREAM_RETRIES=0 disables the automatic replay')


def test_failure_before_any_content_is_not_agent_retried():
    '''Nothing was shown yet, so the agent leaves the retry to the provider layer.'''
    llm = FlakyStream([
        [('raise', 'connect failed')],
        [('answer', 'never used')],
    ])
    agent = _agent(llm)
    try:
        agent.ask('go', timeout=8)
    except RunFailed as exc:
        assert 'connect failed' in exc.error
    else:
        raise AssertionError('a pre-content failure must abort')
    assert llm.calls == 1, 'the agent only replays once content has been shown'
    print('ok  pre-content failures stay with the provider retry layer')


def test_reasoning_prefix_is_also_suppressed():
    '''Reasoning chunks already shown are suppressed on the replay too.'''
    llm = FlakyStream([
        [('cot', 'think '), ('cot', 'hard'), ('raise', 'reset')],
        [('cot', 'think '), ('cot', 'hard'), ('cot', 'er'), ('answer', 'done')],
    ])
    agent = _agent(llm)
    rec = Recorder()
    try:
        answer = agent.ask('reason', timeout=8)
        assert answer == 'done'
    finally:
        rec.close()

    reasons = [event.content for event in rec.of('agent:reason:step')]
    assert reasons == ['think ', 'hard', 'er'], reasons
    answers = [event.content for event in rec.of('agent:answer')]
    assert answers == ['done'], answers
    print('ok  reasoning text is suppressed separately from the answer')


def main():
    '''Run every test in this module and report a summary.'''
    tests = [
        test_mid_stream_failure_is_replayed_without_duplicates,
        test_divergent_replay_fails_instead_of_splicing,
        test_stream_retries_can_be_disabled,
        test_failure_before_any_content_is_not_agent_retried,
        test_reasoning_prefix_is_also_suppressed,
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
