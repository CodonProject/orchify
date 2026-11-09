"""Tests for the streamable-HTTP MCP transport and the client side of it."""

import json

import pytest

httpx = pytest.importorskip('httpx')
fastapi = pytest.importorskip('fastapi')

from fastapi.testclient import TestClient

from orchify.mcp import MCPClient
from orchify.mcp_server import MCPHTTPServer, MCPRequestError, MCPStdioServer
from orchify.server import create_app


class _Echo:
    """A tiny tool used to exercise ``tools/call`` results."""

    name = 'echo'
    title = 'Echo'
    description = 'Echo the text back'
    annotations = {'readOnlyHint': True}
    info = {
        'function': {
            'name': 'echo',
            'description': 'Echo the text back',
            'parameters': {
                'type': 'object',
                'properties': {'text': {'type': 'string'}},
                'required': ['text'],
            },
        }
    }

    def execute(self, text: str = '') -> dict:
        """Return the text inside a dictionary (for structured content)."""
        return {'echoed': text}


def _transport(**kwargs) -> MCPHTTPServer:
    """Build an HTTP transport around a server exposing ``echo``."""
    server = MCPStdioServer(None, tools=[_Echo()], expose='tools', **kwargs)
    return MCPHTTPServer(server, stream_interval=0.01)


def _post(transport: MCPHTTPServer, message, headers=None):
    """POST one JSON message and decode the response body."""
    status, out, body = transport.handle_post(json.dumps(message).encode('utf-8'), headers or {})
    return status, out, (json.loads(body) if body else None)


def _initialize(transport: MCPHTTPServer, version: str = '2025-06-18'):
    """Run the handshake and return the session id header."""
    status, headers, body = _post(transport, {
        'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
        'params': {'protocolVersion': version, 'capabilities': {},
                   'clientInfo': {'name': 'test', 'version': '1'}},
    })
    assert status == 200
    assert body['result']['protocolVersion'] in ('2025-06-18', '2025-03-26', '2025-11-25')
    return headers.get('mcp-session-id', '')


def test_initialize_creates_a_session_and_declares_capabilities():
    transport = _transport()
    session_id = _initialize(transport)
    assert session_id
    status, _, body = _post(transport, {'jsonrpc': '2.0', 'id': 2, 'method': 'ping'},
                            {'mcp-session-id': session_id})
    assert status == 200 and body['result'] == {}


def test_initialize_negotiates_the_requested_version():
    transport = _transport()
    status, _, body = _post(transport, {
        'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
        'params': {'protocolVersion': '2025-03-26', 'capabilities': {}},
    })
    assert status == 200
    assert body['result']['protocolVersion'] == '2025-03-26'
    assert 'tools' in body['result']['capabilities']
    assert body['result']['serverInfo']['title']


def test_unknown_protocol_version_falls_back_to_the_latest_supported():
    transport = _transport()
    status, _, body = _post(transport, {
        'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
        'params': {'protocolVersion': '1999-01-01', 'capabilities': {}},
    })
    assert status == 200
    assert body['result']['protocolVersion'] == '2025-06-18'


def test_expired_session_is_a_404():
    transport = _transport()
    status, _, body = _post(transport, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
                            {'mcp-session-id': 'nope'})
    assert status == 404
    assert body['error']['code'] == -32000


def test_unsupported_version_header_is_a_400():
    transport = _transport()
    status, _, body = _post(transport, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
                            {'mcp-protocol-version': '1999-01-01'})
    assert status == 400
    assert 'unsupported protocol version' in body['error']['message']


def test_notifications_are_accepted_without_a_body():
    transport = _transport()
    _initialize(transport)
    status, _, body = _post(transport, {'jsonrpc': '2.0', 'method': 'notifications/initialized'})
    assert status == 202
    assert body is None


def test_invalid_json_is_rejected():
    transport = _transport()
    status, _, body = transport.handle_post(b'{not json', {})
    assert status == 400
    assert json.loads(body)['error']['code'] == -32700


def test_tools_are_paginated_with_an_opaque_cursor():
    tools = []

    class _Named(_Echo):
        def __init__(self, name):
            self.name = name
            self.title = name
            self.info = {'function': {'name': name, 'description': name,
                                      'parameters': {'type': 'object', 'properties': {}}}}

    for index in range(3):
        tools.append(_Named(f'tool_{index}'))
    transport = MCPHTTPServer(MCPStdioServer(None, tools=tools, expose='tools'), page_size=2)
    session_id = _initialize(transport)
    headers = {'mcp-session-id': session_id}
    status, _, body = _post(transport, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
                            headers)
    assert status == 200
    assert len(body['result']['tools']) == 2
    assert body['result']['nextCursor']
    status, _, body = _post(transport, {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/list',
                                        'params': {'cursor': body['result']['nextCursor']}},
                            headers)
    assert len(body['result']['tools']) == 1
    assert 'nextCursor' not in body['result']


def test_invalid_cursor_is_a_protocol_error():
    server = MCPStdioServer(None, tools=[_Echo()], expose='tools')
    with pytest.raises(MCPRequestError) as error:
        server.handle_request('tools/list', {'cursor': 'not-a-cursor'})
    assert error.value.code == -32602


def test_tool_metadata_and_structured_content():
    transport = _transport()
    session_id = _initialize(transport)
    headers = {'mcp-session-id': session_id}
    status, _, body = _post(transport, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'}, headers)
    entry = body['result']['tools'][0]
    assert entry['title'] == 'Echo'
    assert entry['annotations'] == {'readOnlyHint': True}
    status, _, body = _post(transport, {
        'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
        'params': {'name': 'echo', 'arguments': {'text': 'hi'}},
    }, headers)
    assert body['result']['structuredContent'] == {'echoed': 'hi'}
    assert body['result']['isError'] is False


def test_cancellation_notification_is_harmless():
    transport = _transport()
    session_id = _initialize(transport)
    status, _, body = _post(transport, {
        'jsonrpc': '2.0', 'method': 'notifications/cancelled',
        'params': {'requestId': 99, 'reason': 'test'},
    }, {'mcp-session-id': session_id})
    assert status == 202 and body is None


def test_delete_ends_the_session():
    transport = _transport()
    session_id = _initialize(transport)
    status, _, body = transport.handle_delete({'mcp-session-id': session_id})
    assert status == 200 and body == b'{}'
    status, _, _ = _post(transport, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
                         {'mcp-session-id': session_id})
    assert status == 404


def test_delete_without_a_session_is_an_error():
    transport = _transport()
    status, _, _ = transport.handle_delete({})
    assert status == 400


def test_get_returns_an_sse_stream():
    transport = _transport()
    status, headers, stream = transport.handle_get({})
    assert status == 200
    assert headers['content-type'] == 'text/event-stream'
    iterator = iter(stream)
    assert next(iterator) == b': connected\n\n'


def test_create_app_mounts_the_mcp_endpoint():
    transport = MCPHTTPServer(MCPStdioServer(None, tools=[_Echo()], expose='tools'))
    app = create_app(mcp=transport)
    with TestClient(app) as client:
        assert client.get('/api/health').json()['mcp'] is True
        response = client.post('/mcp', json={
            'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
            'params': {'protocolVersion': '2025-06-18', 'capabilities': {}},
        })
        assert response.status_code == 200
        assert response.headers['mcp-session-id']
        session_id = response.headers['mcp-session-id']
        response = client.post('/mcp', headers={'mcp-session-id': session_id}, json={
            'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
            'params': {'name': 'echo', 'arguments': {'text': 'via http'}},
        })
        assert response.json()['result']['structuredContent'] == {'echoed': 'via http'}
        assert client.delete('/mcp', headers={'mcp-session-id': session_id}).status_code == 200


def test_mcp_endpoint_uses_the_api_auth():
    transport = MCPHTTPServer(MCPStdioServer(None, tools=[_Echo()], expose='tools'))
    app = create_app(mcp=transport, api_keys=['secret'])
    body = {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
            'params': {'protocolVersion': '2025-06-18', 'capabilities': {}}}
    with TestClient(app) as client:
        assert client.post('/mcp', json=body).status_code == 401
        assert client.post('/mcp', json=body,
                           headers={'X-API-Key': 'secret'}).status_code == 200
        assert client.delete('/mcp').status_code == 401


def test_client_speaks_the_http_transport_end_to_end():
    transport = MCPHTTPServer(MCPStdioServer(None, tools=[_Echo()], expose='tools'))
    app = create_app(mcp=transport)
    with TestClient(app) as raw:
        client = MCPClient(url='http://testserver/mcp', name='http', client=raw,
                           resume_stream=False, timeout=10.0)
        try:
            tools = {tool.name: tool for tool in client.tools()}
            assert set(tools) == {'http_echo'}
            # A structured result reaches an agent as JSON text, while
            # call_tool() itself still returns the decoded object.
            assert json.loads(tools['http_echo'].execute(text='round trip')) == {
                'echoed': 'round trip'}
            assert client.call_tool('echo', {'text': 'raw'}) == {'echoed': 'raw'}
            # Dropping the session must make the client re-initialise and retry.
            with transport._lock:
                transport._sessions.clear()
            tools = {tool.name for tool in client.tools(refresh=True)}
            assert tools == {'http_echo'}
        finally:
            client.close()


def test_client_rejects_an_unsupported_negotiated_version():
    class _OldServer(MCPStdioServer):
        def handle_request(self, method, params=None):
            if method == 'initialize':
                return {'protocolVersion': '1999-01-01',
                        'capabilities': {'tools': {}},
                        'serverInfo': {'name': 'old', 'version': '0'}}
            return super().handle_request(method, params)

    transport = MCPHTTPServer(_OldServer(None, tools=[_Echo()], expose='tools'))
    app = create_app(mcp=transport)
    with TestClient(app) as raw:
        client = MCPClient(url='http://testserver/mcp', name='old', client=raw,
                           resume_stream=False, timeout=5.0, auto_start=False)
        with pytest.raises(Exception) as error:
            client.connect()
        assert 'protocol version' in str(error.value)
