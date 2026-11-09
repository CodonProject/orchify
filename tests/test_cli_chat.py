import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, orchify_broker
from orchify.cli import EXIT_ERROR, EXIT_OK, _chat, _serve, main
from orchify.testing import FakeLLM, final


def test_chat_runs_one_prompt_against_a_supplied_agent(capsys):
    '''--prompt mode prints the answer and exits cleanly.'''
    agent = Agent(name='chatty', llm=FakeLLM([[final('hello from chat')]]))
    assert _chat(agent=agent, prompt='hi', max_steps=2) == EXIT_OK
    assert 'hello from chat' in capsys.readouterr().out
    assert any(message.get('role') == 'user' for message in agent.messages)


def test_chat_survives_a_failing_turn(monkeypatch):
    '''A broken provider is reported instead of aborting the REPL.'''
    agent = Agent(name='broken', llm=FakeLLM([[final('never used')]]))

    def boom(*args, **kwargs):
        raise RuntimeError('provider is down')

    monkeypatch.setattr(agent, 'ask', boom)
    stdin = io.StringIO('/help\nhello\n/quit\n')
    monkeypatch.setattr('builtins.input', lambda *a, **k: stdin.readline().rstrip('\n'))
    assert _chat(agent=agent) == EXIT_OK


def test_chat_reports_missing_credentials(monkeypatch):
    '''A keyless agent exits with EXIT_MISSING_KEY instead of a traceback.'''
    from orchify.errors import MissingCredentials

    agent = Agent(name='keyless', llm=FakeLLM([[]]))

    def boom(*args, **kwargs):
        raise MissingCredentials('set API_KEY', owner='keyless')

    monkeypatch.setattr(agent, 'ask', boom)
    assert _chat(agent=agent, prompt='hi') == 2


def test_serve_builds_the_app_and_binds_the_requested_address(monkeypatch, capsys):
    '''The serve subcommand forwards host/port and builds a session factory.'''
    pytest.importorskip('fastapi')
    import orchify.server as server_module

    seen = {}

    def fake_create_app(**kwargs):
        seen['create'] = kwargs
        return object()

    def fake_serve(app, *, host=None, port=None):
        seen['app'] = app
        seen['host'] = host
        seen['port'] = port

    monkeypatch.setattr(server_module, 'create_app', fake_create_app)
    monkeypatch.setattr(server_module, 'serve', fake_serve)
    assert _serve(host='0.0.0.0', port=9911, keys='a,b') == EXIT_OK
    assert seen['host'] == '0.0.0.0' and seen['port'] == 9911
    assert seen['create']['api_keys'] == ['a', 'b']
    factory = seen['create']['agent_factory']
    built = factory('session-1')
    assert built.name == 'session-1'
    assert 'orchify serve: http://0.0.0.0:9911' in capsys.readouterr().out


def test_serve_uses_the_agent_from_a_setup_file(monkeypatch):
    '''A setup file that exposes a ready agent is served as-is.'''
    import uuid

    pytest.importorskip('fastapi')
    import orchify.server as server_module

    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '.data', 'cli-%s' % uuid.uuid4().hex[:8])
    os.makedirs(directory, exist_ok=True)
    setup = os.path.join(directory, 'setup.py')
    with open(setup, 'w', encoding='utf-8') as handle:
        handle.write(
            'from orchify import Agent\n'
            'from orchify.testing import FakeLLM, final\n'
            "agent = Agent(name='from-setup', llm=FakeLLM([[final('ok')]]))\n")
    seen = {}
    monkeypatch.setattr(server_module, 'create_app',
                        lambda **kwargs: seen.update(kwargs) or object())
    monkeypatch.setattr(server_module, 'serve', lambda app, **kwargs: None)
    try:
        assert _serve(setup=setup) == EXIT_OK
        assert seen['agent_factory']('ignored').name == 'from-setup'
    finally:
        for name in list(sys.modules):
            if name.startswith('orchify_setup_') or name == 'setup':
                sys.modules.pop(name, None)
        import shutil
        shutil.rmtree(directory, ignore_errors=True)


def test_cli_parser_exposes_the_new_commands():
    '''`orchify serve` and `orchify chat` are registered.'''
    import argparse

    from orchify.cli import _build_parser

    parser = _build_parser()
    serve = parser.parse_args(['serve', '--port', '9000'])
    assert serve.command == 'serve' and serve.port == 9000 and serve.host == '127.0.0.1'
    chat = parser.parse_args(['chat', '--prompt', 'hi', '--max-steps', '2'])
    assert chat.command == 'chat' and chat.prompt == 'hi' and chat.max_steps == 2
    subparsers = [action for action in parser._actions
                  if isinstance(action, argparse._SubParsersAction)]
    assert subparsers and {'serve', 'chat', 'mcp', 'plugins'} <= set(subparsers[0].choices)


def test_main_dispatches_chat(monkeypatch):
    '''main() routes the chat subcommand.'''
    calls = {}
    monkeypatch.setattr('orchify.cli._chat',
                        lambda **kwargs: calls.update(kwargs) or EXIT_OK)
    assert main(['chat', '--prompt', 'ping']) == EXIT_OK
    assert calls['prompt'] == 'ping'


def test_error_exit_code_alias_exists():
    '''EXIT_ERROR is defined (it used to be an undefined name).'''
    assert EXIT_ERROR == 1


def teardown_module(module):
    '''Leave no hooks behind for other test modules.'''
    orchify_broker.drain_hooks(timeout=5.0)
