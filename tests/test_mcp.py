import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from orchify import Agent
from orchify.mcp import MCPClient, MCPError
from orchify.testing import FakeLLM, final, tool_call

_SERVER = r'''
import json
import sys


def send(message):
    sys.stdout.write(json.dumps(message) + '\n')
    sys.stdout.flush()


TOOLS = [
    {
        'name': 'echo_text',
        'description': 'Echo text back.',
        'inputSchema': {
            'type': 'object',
            'properties': {'text': {'type': 'string', 'description': 'Text to echo.'}},
            'required': ['text'],
        },
    },
    {
        'name': 'fail_tool',
        'description': 'Always fails.',
        'inputSchema': {'type': 'object', 'properties': {}},
    },
]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    method = message.get('method')
    request_id = message.get('id')
    if request_id is None:
        continue
    if method == 'initialize':
        send({'jsonrpc': '2.0', 'id': request_id, 'result': {
            'protocolVersion': '2025-06-18',
            'capabilities': {'tools': {}},
            'serverInfo': {'name': 'fake', 'version': '1.0.0'},
        }})
    elif method == 'tools/list':
        send({'jsonrpc': '2.0', 'id': request_id,
              'result': {'tools': TOOLS, 'nextCursor': None}})
    elif method == 'tools/call':
        params = message.get('params') or {}
        arguments = params.get('arguments') or {}
        if params.get('name') == 'fail_tool':
            send({'jsonrpc': '2.0', 'id': request_id, 'result': {
                'content': [{'type': 'text', 'text': 'deliberate failure'}],
                'isError': True,
            }})
        else:
            send({'jsonrpc': '2.0', 'id': request_id, 'result': {
                'content': [{'type': 'text', 'text': 'echo: ' + str(arguments.get('text'))}],
                'isError': False,
            }})
    else:
        send({'jsonrpc': '2.0', 'id': request_id,
              'error': {'code': -32601, 'message': 'unknown method ' + str(method)}})
'''


def _client(name='mcp'):
    '''Start a stdio client against the fake server, handed over inline.

    The server source is passed with ``-c`` so the test never has to write a
    file: it runs wherever the interpreter runs, including sandboxed CI.
    '''
    return MCPClient(command=sys.executable, args=['-c', _SERVER], name=name, timeout=10.0)


def test_mcp_lists_and_calls_tools():
    '''The client handshakes, lists tools and flattens a call result to text.'''
    with _client() as client:
        assert client.server_info['name'] == 'fake'
        raw = client.list_tools()
        assert [tool['name'] for tool in raw] == ['echo_text', 'fail_tool']
        assert client.call_tool('echo_text', {'text': 'hi'}) == 'echo: hi'
        with pytest.raises(MCPError) as raised:
            client.call_tool('fail_tool', {})
        assert 'deliberate failure' in str(raised.value)


def test_mcp_tools_become_orchify_tools():
    '''MCP tools arrive as Tool objects that a model can call directly.'''
    with _client() as client:
        tools = {tool.name: tool for tool in client.tools()}
        assert set(tools) == {'mcp_echo_text', 'mcp_fail_tool'}
        echo = tools['mcp_echo_text']
        assert echo.parameters_schema['required'] == ['text']
        assert echo.info['function']['parameters']['properties']['text']['type'] == 'string'
        assert echo.execute(text='round trip') == 'echo: round trip'

        prefixed = {tool.name for tool in client.tools(prefix='fs_', refresh=True)}
        assert prefixed == {'fs_echo_text', 'fs_fail_tool'}


def test_mcp_tool_drives_an_agent():
    '''A tool served over MCP is invoked through the normal agent tool loop.'''
    with _client(name='fake') as client:
        tools = client.tools()
        agent = Agent(
            name='mcp-agent',
            llm=FakeLLM([
                [final('', [tool_call('fake_echo_text', '{"text": "from model"}')])],
                [final('the tool said so')],
            ]),
            tools=tools,
        )
        answer = agent.ask('use the tool', max_steps=2)
        assert answer == 'the tool said so'
        assert 'echo: from model' in str(agent.messages)


def test_mcp_reports_protocol_errors():
    '''An unknown method surfaces as MCPError with the server error code.'''
    with _client() as client:
        with pytest.raises(MCPError) as raised:
            client.request('does/not/exist', {})
        assert raised.value.code == -32601


def test_stdio_failure_reports_the_server_stderr():
    '''A server that dies before the handshake explains why in the error.'''
    with pytest.raises(MCPError) as raised:
        MCPClient(
            command=sys.executable,
            args=['-c', 'import sys; sys.stderr.write("boom: missing config\\n"); '
                         'sys.stderr.flush(); sys.exit(1)'],
            name='broken',
            timeout=10.0,
        )
    message = str(raised.value)
    assert 'closed the connection' in message
    assert 'boom: missing config' in message
