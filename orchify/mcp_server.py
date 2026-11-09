"""Expose an Orchify agent (or its tools) as an MCP server.

The counterpart of :mod:`orchify.mcp`: instead of calling somebody else's MCP
server, this module speaks the same JSON-RPC protocol so any MCP client
(Claude Desktop, an IDE plugin, ``orchify.mcp.MCPClient`` itself) can call the
agent.

Two transports are available:

stdio
    Newline-delimited JSON on stdin/stdout, started with
    :func:`serve_stdio` or ``orchify mcp``.
streamable HTTP
    One endpoint answering POST (JSON or an SSE body), GET (a server-to-client
    SSE stream) and DELETE (terminate the session), mounted into a FastAPI app
    by :class:`MCPHTTPServer` (``orchify serve --mcp``).

Exposure modes:

``agent`` (default)
    One ``ask`` tool that runs a full agent turn and returns the answer.
``tools``
    Every tool attached to the agent as its own MCP tool, executed directly.
``both``
    The ``ask`` tool plus every agent tool.

The server processes requests one at a time, which matches the runtime's
single-run guard: an MCP client that sends two calls at once gets them in order,
not interleaved.
"""

import base64
import json
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple

from orchify.utils import get_logger

logger = get_logger(__name__)

PROTOCOL_VERSION = '2025-06-18'
SUPPORTED_PROTOCOL_VERSIONS = ('2025-11-25', PROTOCOL_VERSION, '2025-03-26')
DEFAULT_PAGE_SIZE = 50

_MODES = ('agent', 'tools', 'both')


class MCPRequestError(Exception):
    '''A JSON-RPC level error raised while handling one MCP request.'''

    def __init__(self, message: str, *, code: int = -32603, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


def build_ask_tool(agent: Any) -> Dict[str, Any]:
    '''Describe the ``ask`` tool an MCP client sees for a whole agent.

    Args:
        agent (Any): The agent the server runs.

    Returns:
        Dict[str, Any]: An MCP tool description.
    '''
    return {
        'name': 'ask',
        'title': f'Ask {getattr(agent, "name", "the agent")}',
        'description': (
            f'Ask the "{getattr(agent, "name", "agent")}" Orchify agent a question and '
            'return its final answer. The agent may call tools and wait for approvals.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'text': {'type': 'string', 'description': 'The user message to send.'},
                'timeout': {
                    'type': 'number',
                    'description': 'Seconds to wait for the answer (default: the agent default).',
                },
            },
            'required': ['text'],
        },
    }


def build_tool_entry(tool: Any, *, prefix: str = '') -> Dict[str, Any]:
    '''Describe one Orchify tool as an MCP tool.

    Args:
        tool (Any): The :class:`orchify.tool.Tool` to describe.
        prefix (str): Optional name prefix (used to namespace agent tools).

    Returns:
        Dict[str, Any]: An MCP tool description.
    '''
    info = tool.info.get('function', {}) if isinstance(tool.info, dict) else {}
    name = f'{prefix}{tool.name}' if prefix else tool.name
    entry: Dict[str, Any] = {
        'name': name,
        'title': str(getattr(tool, 'title', '') or tool.name),
        'description': info.get('description', '') or getattr(tool, 'description', '') or tool.name,
        'inputSchema': info.get('parameters') or {'type': 'object', 'properties': {}},
    }
    annotations = getattr(tool, 'annotations', None)
    if isinstance(annotations, dict):
        entry['annotations'] = dict(annotations)
    output_schema = getattr(tool, 'output_schema', None)
    if isinstance(output_schema, dict):
        entry['outputSchema'] = dict(output_schema)
    return entry


class MCPStdioServer:
    '''A stdio MCP server that exposes an agent and/or its tools.

    Args:
        agent (Any): The agent to expose; may be ``None`` in ``tools`` mode when
            ``tools`` is given explicitly.
        tools (Optional[list]): Tools to expose in ``tools``/``both`` mode;
            defaults to ``agent.tools``.
        expose (str): ``'agent'``, ``'tools'`` or ``'both'``.
        name (str): Server name reported during ``initialize``.
        version (str): Server version reported during ``initialize``.
        tool_prefix (str): Prefix applied to agent tool names (default ``'agent_'``
            in ``both`` mode so ``ask`` and tool names cannot collide).
    '''

    def __init__(self, agent: Any = None, *, tools: Optional[list] = None,
                 expose: str = 'agent', name: str = 'orchify', version: str = '0.0.1',
                 tool_prefix: str = '', page_size: int = DEFAULT_PAGE_SIZE):
        if expose not in _MODES:
            raise ValueError(f'expose must be one of {_MODES}, got {expose!r}')
        if agent is None and tools is None:
            raise ValueError('MCPStdioServer needs an agent or explicit tools')
        self.agent = agent
        self.expose = expose
        self.name = name
        self.version = version
        self.tool_prefix = tool_prefix or ('agent_' if expose == 'both' else '')
        self.page_size = max(1, int(page_size or DEFAULT_PAGE_SIZE))
        self.protocol_version = PROTOCOL_VERSION
        self.client_info: Dict[str, Any] = {}
        source_tools = tools if tools is not None else list(getattr(agent, 'tools', {}).values())
        self._tools = {tool.name: tool for tool in source_tools}
        self._by_mcp_name = {
            (f'{self.tool_prefix}{tool.name}' if self.tool_prefix else tool.name): tool
            for tool in self._tools.values()
        }

    def tool_list(self) -> List[Dict[str, Any]]:
        '''Return the MCP ``tools/list`` payload for the configured exposure.'''
        entries: List[Dict[str, Any]] = []
        if self.expose in ('agent', 'both') and self.agent is not None:
            entries.append(build_ask_tool(self.agent))
        if self.expose in ('tools', 'both'):
            for tool in self._tools.values():
                entries.append(build_tool_entry(tool, prefix=self.tool_prefix))
        return entries

    def tool_page(self, cursor: Optional[str] = None) -> Dict[str, Any]:
        '''Return one ``tools/list`` page.

        Args:
            cursor (Optional[str]): Opaque cursor from a previous page.

        Returns:
            Dict[str, Any]: ``{'tools': [...]}`` plus ``nextCursor`` when more
            pages follow.

        Raises:
            MCPRequestError: When the cursor cannot be decoded.
        '''
        offset = _decode_cursor(cursor) if cursor else 0
        entries = self.tool_list()
        page = entries[offset:offset + self.page_size]
        result: Dict[str, Any] = {'tools': page}
        following = offset + len(page)
        if following < len(entries):
            result['nextCursor'] = _encode_cursor(following)
        return result

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        '''Execute one MCP ``tools/call`` request.

        Args:
            name (str): MCP tool name.
            arguments (Optional[Dict[str, Any]]): Tool arguments.

        Returns:
            Dict[str, Any]: ``{'content': [...], 'isError': bool}``.

        Raises:
            MCPRequestError: When the tool is unknown.
        '''
        arguments = dict(arguments or {})
        if name == 'ask' and self.expose in ('agent', 'both') and self.agent is not None:
            text = arguments.get('text')
            if not isinstance(text, str) or not text:
                raise MCPRequestError('ask requires a non-empty "text" argument', code=-32602)
            timeout = arguments.get('timeout')
            try:
                answer = self.agent.ask(text, timeout=float(timeout) if timeout else None)
            except Exception as exc:  # a failing run is a tool error, not a protocol error
                logger.warning('mcp-server: ask failed: %s', exc)
                return {'content': [{'type': 'text', 'text': f'{type(exc).__name__}: {exc}'}],
                        'isError': True}
            return {'content': [{'type': 'text', 'text': '' if answer is None else str(answer)}],
                    'isError': False}

        tool = self._by_mcp_name.get(name)
        if tool is None:
            raise MCPRequestError(f'unknown tool: {name}', code=-32602)
        try:
            result = tool.execute(**arguments)
        except Exception as exc:
            logger.warning('mcp-server: tool %s failed: %s', name, exc)
            return {'content': [{'type': 'text', 'text': f'{type(exc).__name__}: {exc}'}],
                    'isError': True}
        payload: Dict[str, Any] = {
            'content': [{'type': 'text', 'text': _as_text(result)}],
            'isError': False,
        }
        if isinstance(result, dict):
            # The specification pairs structured content with serialised JSON
            # text so clients that only understand text still get every value.
            payload['structuredContent'] = result
        return payload

    def handle_request(self, method: str, params: Optional[Dict[str, Any]] = None,
                       protocol_version: Optional[str] = None) -> Any:
        '''Dispatch one JSON-RPC request method.

        Args:
            method (str): JSON-RPC method name.
            params (Optional[Dict[str, Any]]): Request parameters.
            protocol_version (Optional[str]): Version negotiated by the caller
                (HTTP sessions). ``None`` means this stdio server owns it.

        Returns:
            Any: The JSON-RPC result value.

        Raises:
            MCPRequestError: For unknown methods.
        '''
        params = dict(params or {})
        if method == 'initialize':
            requested = str(params.get('protocolVersion') or '')
            version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
            if protocol_version is None:
                # Over stdio one process serves one client, so the negotiated
                # version belongs to the server. The HTTP transport keeps the
                # version per session instead and must not overwrite this field.
                self.protocol_version = version
            self.client_info = dict(params.get('clientInfo') or {})
            return {
                'protocolVersion': version,
                'capabilities': {'tools': {'listChanged': False}},
                'serverInfo': {
                    'name': self.name,
                    'title': self.name,
                    'version': self.version,
                },
            }
        if method == 'ping':
            return {}
        if method == 'tools/list':
            return self.tool_page(params.get('cursor'))
        if method == 'tools/call':
            return self.call_tool(str(params.get('name') or ''), params.get('arguments') or {})
        raise MCPRequestError(f'method not found: {method}', code=-32601)

    def handle_message(self, message: Dict[str, Any],
                       protocol_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        '''Handle one decoded JSON-RPC message.

        Notifications (no ``id``) are acknowledged with ``None`` and never get a
        response frame.

        Args:
            message (Dict[str, Any]): The decoded message.
            protocol_version (Optional[str]): Version of the calling session;
                defaults to the version negotiated on this stdio server.

        Returns:
            Optional[Dict[str, Any]]: The response frame, or None for a notification.
        '''
        if not isinstance(message, dict):
            return _error_frame(None, -32600, 'invalid request')
        method = message.get('method')
        request_id = message.get('id')
        if method is None:
            return None
        if request_id is None:
            if method == 'notifications/cancelled':
                params = message.get('params') or {}
                logger.debug('mcp-server: client cancelled request %s (%s)',
                             params.get('requestId'), params.get('reason', ''))
            return None  # notification (e.g. notifications/initialized)
        active = protocol_version or self.protocol_version
        if active and not _version_supported(active):
            return _error_frame(request_id, -32600,
                                f'unsupported protocol version: {active}')
        try:
            result = self.handle_request(str(method), message.get('params'), active)
        except MCPRequestError as exc:
            return _error_frame(request_id, exc.code, str(exc), exc.data)
        except Exception as exc:
            logger.error('mcp-server: %s failed: %s', method, exc)
            return _error_frame(request_id, -32603, f'{type(exc).__name__}: {exc}')
        return {'jsonrpc': '2.0', 'id': request_id, 'result': result}

    def handle_line(self, line: str) -> Optional[Dict[str, Any]]:
        '''Decode and handle one newline-delimited JSON-RPC frame.

        Args:
            line (str): One input line.

        Returns:
            Optional[Dict[str, Any]]: The response frame, or None when nothing is due.
        '''
        line = line.strip()
        if not line:
            return None
        try:
            message = json.loads(line)
        except ValueError:
            return _error_frame(None, -32700, 'parse error')
        return self.handle_message(message)

    def serve(self, instream: Optional[TextIO] = None, outstream: Optional[TextIO] = None) -> None:
        '''Serve requests from ``instream`` until EOF.

        Args:
            instream (Optional[TextIO]): Input stream; defaults to stdin.
            outstream (Optional[TextIO]): Output stream; defaults to stdout.
        '''
        instream = instream or sys.stdin
        outstream = outstream or sys.stdout
        for line in instream:
            try:
                response = self.handle_line(line)
            except Exception as exc:  # defensive: never die on one frame
                logger.error('mcp-server: frame failed: %s', exc)
                traceback.print_exc(file=sys.stderr)
                continue
            if response is None:
                continue
            outstream.write(json.dumps(response, ensure_ascii=False) + '\n')
            outstream.flush()


def _as_text(value: Any) -> str:
    '''Render a tool result as MCP text content.'''
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _error_frame(request_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    '''Build a JSON-RPC error frame.'''
    error: Dict[str, Any] = {'code': code, 'message': message}
    if data is not None:
        error['data'] = data
    return {'jsonrpc': '2.0', 'id': request_id, 'error': error}


def _encode_cursor(offset: int) -> str:
    '''Encode a pagination cursor (opaque to clients).'''
    raw = json.dumps({'offset': int(offset)}, separators=(',', ':')).encode('utf-8')
    return base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')


def _decode_cursor(cursor: str) -> int:
    '''Decode a pagination cursor.

    Args:
        cursor (str): The cursor from the client.

    Returns:
        int: The offset it encodes.

    Raises:
        MCPRequestError: When the cursor is not one of ours (-32602).
    '''
    try:
        padded = str(cursor) + '=' * (-len(str(cursor)) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode('ascii')).decode('utf-8'))
        offset = int(data['offset'])
    except Exception:
        raise MCPRequestError(f'invalid cursor: {cursor!r}', code=-32602)
    if offset < 0:
        raise MCPRequestError(f'invalid cursor: {cursor!r}', code=-32602)
    return offset


def _version_supported(version: str) -> bool:
    '''Report whether this server speaks a protocol version.'''
    return str(version or '') in SUPPORTED_PROTOCOL_VERSIONS


def _header(headers: Any, name: str) -> str:
    '''Read one header case-insensitively from a mapping or header list.'''
    if headers is None:
        return ''
    lowered = name.lower()
    try:
        items = headers.items()
    except AttributeError:
        items = headers
    for key, value in items:
        if str(key).lower() == lowered:
            return str(value)
    return ''


class MCPHTTPServer:
    '''Streamable-HTTP transport around :class:`MCPStdioServer`.

    One endpoint serves ``POST`` (JSON or an SSE body), ``GET`` (the optional
    server-to-client stream) and ``DELETE`` (end the session). The class has no
    FastAPI dependency: :meth:`handle_post`, :meth:`handle_get` and
    :meth:`handle_delete` return plain ``(status, headers, body)`` tuples that
    :mod:`orchify.server` turns into responses, so the transport can also be
    driven from tests or another framework.

    Args:
        server (Optional[MCPStdioServer]): The protocol handler; one is built
            from ``agent``/``tools`` when omitted.
        session_idle (float): Seconds without traffic before a session expires.
        stream_interval (float): Seconds between keep-alive comments.
        max_sessions (int): Hard cap on live sessions; the least recently used
            sessions are dropped beyond it, so a client that disappears without
            ``DELETE`` cannot grow the table forever.
    '''

    def __init__(self, server: Optional[MCPStdioServer] = None, *, agent: Any = None,
                 tools: Optional[list] = None, expose: str = 'agent',
                 name: str = 'orchify', version: str = '0.0.1', tool_prefix: str = '',
                 page_size: int = DEFAULT_PAGE_SIZE, session_idle: float = 300.0,
                 stream_interval: float = 15.0, max_sessions: int = 4096):
        import threading
        import time

        if server is None:
            server = MCPStdioServer(agent, tools=tools, expose=expose, name=name,
                                    version=version, tool_prefix=tool_prefix,
                                    page_size=page_size)
        elif page_size != DEFAULT_PAGE_SIZE:
            server.page_size = max(1, int(page_size))
        self.server = server
        self.session_idle = float(session_idle)
        self.stream_interval = max(0.1, float(stream_interval))
        self.max_sessions = max(1, int(max_sessions))
        self._time = time
        self._lock = threading.Lock()
        # session_id -> {'touched': float, 'version': str}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._counter = 0
        self._creates = 0
        self.sessions_swept = 0

    # ------------------------------------------------------------- sessions

    def _new_session(self, version: str = '') -> str:
        '''Create (and register) a session id.

        Args:
            version (str): Protocol version negotiated by ``initialize``.
        '''
        with self._lock:
            self._counter += 1
            self._creates += 1
            # Sweep periodically rather than on every request: expired and
            # abandoned sessions would otherwise stay in the table forever.
            if self._creates % 64 == 0 or len(self._sessions) >= self.max_sessions:
                self._sweep_locked()
            session_id = f'orchify-{self._counter}-{int(self._time.time() * 1000)}'
            self._sessions[session_id] = {'touched': self._time.time(),
                                          'version': str(version or '')}
            return session_id

    def _sweep_locked(self) -> int:
        '''Drop expired sessions, then the least recently used ones above the cap.'''
        now = self._time.time()
        dropped = 0
        for session_id in [key for key, value in self._sessions.items()
                           if now - value['touched'] > self.session_idle]:
            self._sessions.pop(session_id, None)
            dropped += 1
        if len(self._sessions) >= self.max_sessions:
            ordered = sorted(self._sessions.items(), key=lambda item: item[1]['touched'])
            for session_id, _ in ordered[:len(self._sessions) - self.max_sessions + 1]:
                self._sessions.pop(session_id, None)
                dropped += 1
        self.sessions_swept += dropped
        return dropped

    def _touch(self, session_id: str) -> Optional[Dict[str, Any]]:
        '''Refresh a session and return its record, or ``None`` when it is gone.'''
        if not session_id:
            return None
        with self._lock:
            entry = self._sessions.get(session_id)
            now = self._time.time()
            if entry is None:
                return None
            if now - entry['touched'] > self.session_idle:
                self._sessions.pop(session_id, None)
                return None
            entry['touched'] = now
            return dict(entry)

    def _drop(self, session_id: str) -> bool:
        '''Remove one session.'''
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    # ------------------------------------------------------------ transports

    def handle_post(self, body: bytes, headers: Any = None) -> Tuple[int, Dict[str, str], bytes]:
        '''Handle one HTTP ``POST``.

        Args:
            body (bytes): The raw request body.
            headers (Any): Incoming headers (mapping or item list).

        Returns:
            Tuple[int, Dict[str, str], bytes]: Status, response headers, body.
        '''
        version = _header(headers, 'mcp-protocol-version')
        if version and not _version_supported(version):
            return self._error(400, None, -32600, f'unsupported protocol version: {version}')
        try:
            payload = json.loads(body.decode('utf-8') if isinstance(body, bytes) else body)
        except ValueError:
            return self._error(400, None, -32700, 'parse error')
        session_id = _header(headers, 'mcp-session-id')
        session = self._touch(session_id) if session_id else None
        if session_id and session is None:
            return self._error(404, None, -32000, 'session expired')
        messages = payload if isinstance(payload, list) else [payload]
        responses: List[Dict[str, Any]] = []
        created = ''
        for message in messages:
            if not isinstance(message, dict):
                responses.append(_error_frame(None, -32600, 'invalid request'))
                continue
            if message.get('method') == 'initialize':
                response = self.server.handle_message(message, self.server.protocol_version)
                result = (response or {}).get('result') or {}
                negotiated = str(result.get('protocolVersion') or PROTOCOL_VERSION)
                created = self._new_session(negotiated)
                active = negotiated
            else:
                # Every session keeps the version IT negotiated, so two clients
                # on different versions cannot change each other's behaviour.
                active = ((session or {}).get('version')
                          or (version if version in SUPPORTED_PROTOCOL_VERSIONS else '')
                          or self.server.protocol_version)
                response = self.server.handle_message(message, active)
            if response is not None:
                responses.append(response)
        out_headers: Dict[str, str] = {'content-type': 'application/json'}
        if created:
            out_headers['mcp-session-id'] = created
        if not responses:
            return 202, out_headers, b''
        return 200, out_headers, json.dumps(responses[0] if not isinstance(payload, list) else responses,
                                            ensure_ascii=False).encode('utf-8')

    def handle_get(self, headers: Any = None) -> Tuple[int, Dict[str, str], Optional[Iterable[bytes]]]:
        '''Handle one HTTP ``GET`` (the server-to-client stream).

        Returns:
            Tuple[int, Dict[str, str], Optional[Iterable[bytes]]]: Status,
            headers and the SSE body generator (``None`` when nothing streams).
        '''
        version = _header(headers, 'mcp-protocol-version')
        if version and not _version_supported(version):
            status, out_headers, body = self._error(400, None, -32600,
                                                    f'unsupported protocol version: {version}')
            return status, out_headers, [body]
        session_id = _header(headers, 'mcp-session-id')
        session = self._touch(session_id) if session_id else None
        if session_id and session is None:
            status, out_headers, body = self._error(404, None, -32000, 'session expired')
            return status, out_headers, [body]
        return (200, {'content-type': 'text/event-stream', 'cache-control': 'no-cache'},
                self.sse_stream(session_id))

    def handle_delete(self, headers: Any = None) -> Tuple[int, Dict[str, str], bytes]:
        '''Handle one HTTP ``DELETE`` (terminate the session).'''
        session_id = _header(headers, 'mcp-session-id')
        if not session_id:
            return self._error(400, None, -32600, 'missing session id')
        if not self._drop(session_id):
            return self._error(404, None, -32000, 'unknown session')
        return 200, {'content-type': 'application/json'}, b'{}'

    def sse_stream(self, session_id: str = '') -> Iterable[bytes]:
        '''Yield SSE keep-alives until the session goes away.

        Args:
            session_id (str): Session the stream belongs to; an empty value
                streams for the whole idle window (clients that never
                initialized).
        '''
        started = self._time.time()
        yield b': connected\n\n'
        while self._time.time() - started < self.session_idle:
            if session_id and self._touch(session_id) is None:
                # The session was dropped (DELETE or idle sweep); the stream
                # must not outlive it.
                return
            yield b': keep-alive\n\n'
            self._time.sleep(self.stream_interval)

    def _error(self, status: int, request_id: Any, code: int, message: str) -> Tuple[int, Dict[str, str], bytes]:
        '''Build a JSON error response tuple.'''
        body = _error_frame(request_id, code, message)
        return status, {'content-type': 'application/json'}, json.dumps(body).encode('utf-8')


def serve_stdio(agent: Any = None, *, tools: Optional[list] = None, expose: str = 'agent',
                name: str = 'orchify', version: str = '0.0.1', tool_prefix: str = '',
                page_size: int = DEFAULT_PAGE_SIZE,
                instream: Optional[TextIO] = None, outstream: Optional[TextIO] = None) -> None:
    '''Serve an agent over stdio MCP; the one-line entry point for scripts.

    Args:
        agent (Any): The agent to expose.
        tools (Optional[list]): Tools to expose in ``tools``/``both`` mode.
        expose (str): ``'agent'``, ``'tools'`` or ``'both'``.
        name (str): Server name.
        version (str): Server version.
        tool_prefix (str): Prefix applied to exposed agent tool names.
        page_size (int): ``tools/list`` page size.
        instream (Optional[TextIO]): Input stream; defaults to stdin.
        outstream (Optional[TextIO]): Output stream; defaults to stdout.
    '''
    server = MCPStdioServer(agent, tools=tools, expose=expose, name=name,
                            version=version, tool_prefix=tool_prefix, page_size=page_size)
    server.serve(instream, outstream)


__all__ = ['MCPStdioServer', 'MCPHTTPServer', 'MCPRequestError', 'PROTOCOL_VERSION',
           'SUPPORTED_PROTOCOL_VERSIONS', 'build_ask_tool', 'build_tool_entry', 'serve_stdio']
