import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify import Agent, MissingCredentials, OpenAICompat, OrchifyError
from orchify.broker import orchify_broker
from orchify.llm.base import LLMInterface
from orchify.testing import FakeLLM, final


def _offline_agent(**kwargs):
    '''An agent backed by a scripted offline LLM.'''
    return Agent(name='offline', llm=FakeLLM([[final('offline answer')]]),
                 system_prompt='sys', **kwargs)


def test_openai_backend_reports_a_missing_key():
    '''The live backend explains a missing key and stays silent with one set.'''
    bare = OpenAICompat(api_key='')
    message = bare.missing_credentials()
    assert message is not None
    assert 'API_KEY' in message
    assert 'FakeLLM' in message
    assert 'orchify doctor' in message

    configured = OpenAICompat(api_key='test-key', base_url='http://127.0.0.1:9/v1')
    assert configured.missing_credentials() is None

    assert LLMInterface().missing_credentials() is None
    print('ok  OpenAICompat reports a missing key, keyless backends do not')


def test_agent_run_fails_fast_without_a_key():
    '''Agent.run raises before registering a run when the provider needs a key.'''
    agent = Agent(name='needs-key', llm=OpenAICompat(api_key=''), system_prompt='sys')
    try:
        agent.run('hello')
    except MissingCredentials as exc:
        assert isinstance(exc, OrchifyError)
        assert isinstance(exc, RuntimeError)
        assert exc.owner == 'needs-key'
        assert 'API_KEY' in str(exc)
    else:
        raise AssertionError('run() should refuse to start without an API key')

    assert agent.is_running() is False
    assert agent.last_run_id == ''
    assert orchify_broker.run_contexts == {}
    print('ok  run() fails fast: no run id, no slot, no broker context')


def test_ask_and_arun_report_the_same_problem():
    '''ask() surfaces the configuration error instead of a provider 401.'''
    agent = Agent(name='asks-key', llm=OpenAICompat(api_key=''), system_prompt='sys')
    try:
        agent.ask('hello')
    except MissingCredentials as exc:
        assert 'doctor' in str(exc)
    else:
        raise AssertionError('ask() should surface the missing key')

    import asyncio

    try:
        asyncio.run(agent.arun('hello'))
    except MissingCredentials:
        pass
    else:
        raise AssertionError('arun() should surface the missing key')
    print('ok  ask() and arun() both surface the configuration error')


def test_offline_backends_never_need_credentials():
    '''FakeLLM and plain LLMInterface subclasses run without any key.'''
    agent = _offline_agent()
    assert agent.ask('hello', timeout=8) == 'offline answer'

    class _NullLLM(LLMInterface):
        '''A keyless backend that never talks to a provider.'''

    assert _NullLLM().missing_credentials() is None
    print('ok  keyless backends run normally')


def test_missing_key_message_names_the_alternatives():
    '''The error text includes every supported configuration route.'''
    message = OpenAICompat(api_key='').missing_credentials()
    for fragment in ('API_KEY', 'OPENAI_API_KEY', '.env', 'api_key=', 'FakeLLM'):
        assert fragment in message, fragment
    print('ok  the message names every supported fix')


def main():
    '''Run every test in this module and report a summary.'''
    tests = [
        test_openai_backend_reports_a_missing_key,
        test_agent_run_fails_fast_without_a_key,
        test_ask_and_arun_report_the_same_problem,
        test_offline_backends_never_need_credentials,
        test_missing_key_message_names_the_alternatives,
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
