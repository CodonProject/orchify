import asyncio
import io
import logging
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify import (Agent, CallChain, Group, Plugin, PluginError, PluginManager,
                     Tool, Vote, configure_logging, get_logger, orchify_broker,
                     orchify_manager)
from orchify.backend import DEFAULT_HTTP_TIMEOUT, WebBackend, _read_http_timeout

from test_pause_approval import FakeLLM, final


def _echo_handlers(logger=None):
    root = logger or logging.getLogger('orchify')
    return [h for h in root.handlers if getattr(h, '_orchify_echo_handler', False)]


class _Capture(logging.Handler):
    '''Collects records so tests can assert on messages without touching stdout.'''

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class _EdgePlugin(Plugin):
    '''A minimal declarative plugin: one hook-free tool collected at load time.'''

    name = 'edge-plugin'
    description = 'plugin used by the edge tests'

    @Plugin.tool(name='edge-declared')
    def declared(self, value: str = 'x') -> str:
        return f'edge:{value}'


def test_logger_api_and_env_switch():
    root = logging.getLogger('orchify')
    logger = get_logger(__name__)
    assert logger.name.startswith('orchify.')
    assert get_logger() is root
    assert get_logger('orchify.agent') is logging.getLogger('orchify.agent')

    configure_logging()
    assert root.level == logging.INFO
    assert len(_echo_handlers(root)) == 1
    assert root.propagate is False
    configure_logging()
    assert len(_echo_handlers(root)) == 1

    configure_logging(echo=False)
    assert _echo_handlers(root) == []
    assert root.propagate is True

    configure_logging(level='DEBUG', echo=False)
    assert root.level == logging.DEBUG
    configure_logging(level='WARNING')
    assert root.level == logging.WARNING

    old_echo = os.environ.get('ORCHIFY_LOG_ECHO')
    old_level = os.environ.get('ORCHIFY_LOG_LEVEL')
    try:
        os.environ['ORCHIFY_LOG_ECHO'] = '0'
        os.environ['ORCHIFY_LOG_LEVEL'] = 'ERROR'
        configure_logging()
        assert root.level == logging.ERROR
        assert _echo_handlers(root) == []

        os.environ['ORCHIFY_LOG_ECHO'] = '1'
        os.environ['ORCHIFY_LOG_LEVEL'] = 'nonsense'
        configure_logging()
        assert root.level == logging.INFO
        assert len(_echo_handlers(root)) == 1
    finally:
        for key, value in (('ORCHIFY_LOG_ECHO', old_echo),
                           ('ORCHIFY_LOG_LEVEL', old_level)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        configure_logging(level='INFO', echo=True)


def test_hook_failure_is_logged_not_printed():
    configure_logging(level='DEBUG', echo=False)
    broker_logger = logging.getLogger('orchify.broker')
    capture = _Capture()
    broker_logger.addHandler(capture)

    def boom(event):
        raise ValueError('hook-boom')

    binding = orchify_broker.register_hook('test:boom', boom, owner='edges')
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            orchify_broker.emit(orchify_broker.event('test:boom', payload={'n': 1}))
    finally:
        orchify_broker.remove_binding(binding)
        broker_logger.removeHandler(capture)
        configure_logging(level='INFO', echo=True)

    assert buffer.getvalue() == '', buffer.getvalue()
    messages = [rec.getMessage() for rec in capture.records]
    assert any('hook-boom' in msg and 'test:boom' in msg for msg in messages), messages


def test_call_chain_unit_and_agent_getters():
    chain = CallChain(root_caller='Agent:A')
    assert chain.get_depth() == 0
    assert chain.get_last_frame() is None
    assert chain.get_chain_str() == 'Agent:A (no calls)'
    assert chain.contains('Agent:A') is True
    assert chain.contains('Agent:B') is False

    first = chain.add_frame('Agent:A', 'Agent:B', 'agent', input_data='hello')
    chain.add_frame('Agent:B', 'Agent:C', 'agent', input_data='world')

    assert chain.get_depth() == 2
    assert chain.get_last_frame().callee == 'Agent:C'
    assert chain.get_chain_str() == 'Agent:A -> Agent:B -> Agent:C'
    assert chain.contains('Agent:B') is True
    assert chain.contains('Agent:Z') is False

    data = chain.to_dict()
    assert data['root_caller'] == 'Agent:A'
    assert len(data['frames']) == 2
    assert data['frames'][0] == {
        'caller': 'Agent:A',
        'callee': 'Agent:B',
        'call_type': 'agent',
        'input_data': 'hello',
        'output_data': '',
        'timestamp': first.timestamp,
    }

    agent = Agent(name='edge-chain', llm=FakeLLM([[final('ok')]]))
    assert agent.get_call_chain() is None
    assert agent.get_call_chain_str() == 'Agent:edge-chain (no call chain)'
    agent._call_chain = chain
    assert agent.get_call_chain() is chain
    assert agent.get_call_chain_str() == 'Agent:A -> Agent:B -> Agent:C'


def test_group_call_chain_getters():
    one = Agent(name='edge-g1', llm=FakeLLM([[final('one')]]))
    two = Agent(name='edge-g2', llm=FakeLLM([[final('two')]]))
    group = Group(name='edge-group', agents=[one, two])

    lines = group.get_call_chain()
    assert lines[0] == 'Group:edge-group'
    assert '  -> Agent:edge-g1' in lines
    assert '  -> Agent:edge-g2' in lines

    chain = group.get_full_call_chain()
    assert isinstance(chain, CallChain)
    assert chain.root_caller == 'Group:edge-group'
    assert [frame.callee for frame in chain.frames] == ['Agent:edge-g1', 'Agent:edge-g2']
    assert all(frame.call_type == 'agent' for frame in chain.frames)
    assert chain.get_chain_str() == 'Group:edge-group -> Agent:edge-g1 -> Agent:edge-g2'


def test_plugin_manager_attach_detach_and_errors():
    manager = PluginManager()
    plugin = manager.register(_EdgePlugin)
    assert plugin is not None
    assert plugin.id == 'edge-plugin'
    assert manager.registered()['edge-plugin'] is plugin
    assert plugin.loaded is False
    assert manager.loaded == []

    try:
        manager.register(_EdgePlugin)
        raise AssertionError('expected PluginError for a duplicate registration')
    except PluginError:
        pass

    loaded = manager.load('edge-plugin')
    assert loaded and loaded[0] is plugin
    assert plugin.loaded is True
    assert manager.loaded == ['edge-plugin']
    assert [t.name for t in plugin._tools] == ['edge-declared']
    assert manager.load('edge-plugin')[0] is plugin

    try:
        manager.attach('edge-plugin', Agent(name='edge-plug', llm=FakeLLM([[final('ok')]])))
        agent = Agent(name='edge-plug', llm=FakeLLM([[final('ok')]]))
        manager.attach('edge-plugin', agent)
        assert 'edge-declared' in agent.tools
        assert agent.tools['edge-declared'].execute(value='v') == 'edge:v'

        plugin.detach_tools()
        assert 'edge-declared' not in agent.tools

        try:
            manager.attach('nope', agent)
            raise AssertionError('expected PluginError for an unknown plugin')
        except PluginError as exc:
            assert 'nope' in str(exc)
    finally:
        manager.unload_all() if hasattr(manager, 'unload_all') else None


def test_web_backend_lifecycle():
    backend = WebBackend()
    try:
        assert backend._thread.is_alive() is True
        assert backend._thread.name == 'OrchifyWebBackend'
        future = asyncio.run_coroutine_threadsafe(
            asyncio.sleep(0, result='ok'), backend._loop)
        assert future.result(timeout=5) == 'ok'
        assert isinstance(backend.timeout, type(_read_http_timeout()))
    finally:
        backend.shutdown()

    assert backend._thread.is_alive() is False
    assert backend._loop.is_running() is False
    assert backend._client is None
    backend.shutdown()
    assert DEFAULT_HTTP_TIMEOUT.connect == 15.0
    assert DEFAULT_HTTP_TIMEOUT.read == 600.0


def test_vote_dataclass():
    from dataclasses import asdict, fields

    vote = Vote(agent_name='a1', vote='Yes')
    assert (vote.agent_name, vote.vote, vote.reason) == ('a1', 'Yes', '')
    assert asdict(vote) == {'agent_name': 'a1', 'vote': 'Yes', 'reason': ''}
    assert [f.name for f in fields(Vote)] == ['agent_name', 'vote', 'reason']
    assert Vote('a', 1) == Vote('a', 1)
    assert Vote('a', 1) != Vote('a', 2)
    assert Vote('a', 'Yes', 'because').reason == 'because'


def test_manager_tap_stays_lazy():
    from orchify.manager import _jsonable

    marker = object()
    orchify_manager._events.clear()
    orchify_broker.emit(orchify_broker.event(
        'test:lazy', payload={'blob': {'inner': marker}}))

    stored = orchify_manager._events[-1]['payload']
    assert stored['blob']['inner'] is marker

    read = orchify_manager.events(limit=5)
    assert read[-1]['event_type'] == 'test:lazy'
    converted = read[-1]['payload']
    assert converted['blob'] is not stored['blob']
    assert converted['blob']['inner'].startswith('<object object at 0x')
    assert _jsonable({'nested': marker})['nested'].startswith('<object object at 0x')
    from orchify import OrchifyManager
    quiet = OrchifyManager(history=0, events=0)
    try:
        quiet.close()
    except Exception:
        pass


def main():
    '''Run every test in this file.

    Returns:
        int: 0 when all tests passed, 1 when any failed.
    '''
    tests = [
        test_logger_api_and_env_switch,
        test_hook_failure_is_logged_not_printed,
        test_call_chain_unit_and_agent_getters,
        test_group_call_chain_getters,
        test_plugin_manager_attach_detach_and_errors,
        test_web_backend_lifecycle,
        test_vote_dataclass,
        test_manager_tap_stays_lazy,
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
