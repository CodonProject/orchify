"""MCP server mode: exposing an Orchify agent over stdio."""
import io
import json
import sys

import pytest

from orchify import Agent
from orchify.mcp import MCPClient
from orchify.mcp_server import MCPStdioServer, _error_frame, serve_stdio
from orchify.testing import FakeLLM, final
from orchify.tool import tool


@tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return text


@tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def _agent():
    '''Agent with a scripted answer and two tools.'''
    return Agent(
        name='mcp-agent',
        llm=FakeLLM([[final('orchestrated answer')], [final('second answer')]]),
        tools=[echo, add],
    )


def _server(expose='agent', agent=None, tools=None):
    return MCPStdioServer(agent if agent is not None else _agent(),
                          tools=tools, expose=expose)


def _request(server, method, params=None, request_id='1'):
    return server.handle_message({'jsonrpc': '2.0', 'id': request_id,
                                  'method': method, 'params': params or {}})


def test_initialize_reports_protocol_and_server_info():
    response = _request(_server(), 'initialize')

    assert response['jsonrpc'] == '2.0'
    result = response['result']
    assert result['protocolVersion'] == '2025-06-18'
    assert result['serverInfo']['name'] == 'orchify'
    assert 'tools' in result['capabilities']


def test_agent_mode_exposes_one_ask_tool():
    tools = _request(_server('agent'), 'tools/list')['result']['tools']

    assert [entry['name'] for entry in tools] == ['ask']
    assert tools[0]['inputSchema']['required'] == ['text']


def test_ask_tool_runs_the_agent_and_returns_the_answer():
    server = _server('agent')

    result = server.call_tool('ask', {'text': 'hello'})

    assert result['isError'] is False
    assert result['content'][0]['text'] == 'orchestrated answer'


def test_tools_mode_exposes_agent_tools_directly():
    tools = {entry['name']: entry for entry in _request(_server('tools'), 'tools/list')['result']['tools']}

    assert set(tools) == {'echo', 'add'}
    assert tools['echo']['inputSchema']['required'] == ['text']


def test_tool_call_executes_the_tool():
    result = _server('tools').call_tool('add', {'a': 2, 'b': 3})

    assert result['isError'] is False
    assert result['content'][0]['text'] == '5'


def test_tool_error_becomes_is_error_content():
    result = _server('tools').call_tool('add', {'a': 'two', 'b': 3})

    assert result['isError'] is True
    assert 'TypeError' in result['content'][0]['text']


def test_both_mode_prefixes_agent_tools():
    names = [entry['name'] for entry in _request(_server('both'), 'tools/list')['result']['tools']]

    assert names[0] == 'ask'
    assert set(names[1:]) == {'agent_echo', 'agent_add'}


def test_unknown_tool_is_a_protocol_error():
    response = _request(_server('tools'), 'tools/call', {'name': 'nope', 'arguments': {}})

    assert response['error']['code'] == -32602


def test_unknown_method_is_a_protocol_error():
    response = _request(_server(), 'does/not/exist')

    assert response['error']['code'] == -32601


def test_notifications_get_no_response():
    server = _server()

    assert server.handle_message({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None


def test_parse_error_frame():
    frame = _server().handle_line('{not json')

    assert frame['error']['code'] == -32700
    assert frame['id'] is None


def test_blank_lines_are_ignored():
    assert _server().handle_line('   ') is None


def test_serve_reads_lines_and_writes_responses():
    incoming = io.StringIO(
        json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}}) + '\n'
        + json.dumps({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}}) + '\n'
    )
    outgoing = io.StringIO()
    server = _server()

    server.serve(incoming, outgoing)

    frames = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert [frame['id'] for frame in frames] == [1, 2]
    assert frames[1]['result']['tools'][0]['name'] == 'ask'


def test_serve_one_frame_failure_does_not_stop_the_loop():
    incoming = io.StringIO(
        '{broken\n' + json.dumps({'jsonrpc': '2.0', 'id': 7, 'method': 'ping', 'params': {}}) + '\n'
    )
    outgoing = io.StringIO()
    _server().serve(incoming, outgoing)

    frames = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert frames[0]['error']['code'] == -32700
    assert frames[1]['id'] == 7
    assert frames[1]['result'] == {}


def test_invalid_expose_mode_is_rejected():
    with pytest.raises(ValueError):
        MCPStdioServer(agent=_agent(), expose='everything')


def test_server_needs_an_agent_or_tools():
    with pytest.raises(ValueError):
        MCPStdioServer()


def test_server_tool_list_empty_in_agent_mode_without_agent():
    server = MCPStdioServer(tools=[echo], expose='tools')
    assert [entry['name'] for entry in server.tool_list()] == ['echo']


_SUBPROCESS_SERVER = '''
from orchify.tool import tool
from orchify.mcp_server import serve_stdio

@tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return text

serve_stdio(None, tools=[echo], expose='tools')
'''


def test_real_client_round_trip_over_stdio():
    client = MCPClient(command=sys.executable, args=['-c', _SUBPROCESS_SERVER],
                       name='mcp', timeout=20.0)
    try:
        tools = client.tools()
        names = {entry.name for entry in tools}
        assert names == {'mcp_echo'}
        result = client.call_tool('echo', {'text': 'round trip'})
        assert 'round trip' in json.dumps(result, ensure_ascii=False)
    finally:
        client.close()


def test_cli_mcp_wires_the_server(monkeypatch):
    from orchify import cli

    recorded = {}

    def fake_serve(agent=None, *, tools=None, expose='agent', name='orchify',
                   version='0.0.1', tool_prefix='', instream=None, outstream=None):
        recorded['agent'] = agent
        recorded['expose'] = expose
        recorded['name'] = name

    monkeypatch.setattr('orchify.mcp_server.serve_stdio', fake_serve)
    monkeypatch.setattr(Agent, 'from_env', classmethod(lambda cls, name='orchify', **kw: _agent()))

    assert cli.main(['mcp', '--expose', 'both', '--name', 'demo']) == cli.EXIT_OK
    assert recorded['expose'] == 'both'
    assert recorded['name'] == 'demo'
    assert recorded['agent'] is not None


def test_cli_mcp_requires_a_setup_or_credentials(monkeypatch, capsys):
    from orchify import cli

    def boom(cls, name='orchify', **kw):
        raise RuntimeError('no api key')

    monkeypatch.setattr(Agent, 'from_env', classmethod(boom))

    assert cli.main(['mcp']) == cli.EXIT_ERROR
    assert 'no api key' in capsys.readouterr().out
