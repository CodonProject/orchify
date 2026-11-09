'''Minimal MCP (Model Context Protocol) client.

The client speaks JSON-RPC 2.0 to an MCP server over **stdio** (a child process)
or over **HTTP** (the streamable-HTTP transport many servers expose). Both
transports are implemented with the standard library plus ``httpx``, which
orchify already depends on, so no MCP SDK is required.

Typical use::

    from orchify import Agent
    from orchify.mcp import MCPClient

    client = MCPClient(command='npx', args=['-y', '@modelcontextprotocol/server-filesystem', '/tmp'])
    tools = client.tools()              # orchify Tool objects, one per MCP tool
    agent = Agent(name='files', tools=tools)
    agent.ask('list the files in /tmp')

The client is thread safe: one reader thread per connection fans JSON-RPC
responses out to the callers waiting on them, so MCP tools can be executed from
the agent's tool executor without serialising the whole transport.
'''

import json
import os
import re
import subprocess
import threading
from typing import Any, Dict, List, Optional

from orchify.errors import OrchifyError
from orchify.tool import Tool
from orchify.utils import get_logger

logger = get_logger(__name__)

PROTOCOL_VERSION = '2025-06-18'
DEFAULT_TIMEOUT = 30.0
_CLIENT_INFO = {'name': 'orchify', 'version': '0.0.1'}
_NAME_RE = re.compile(r'[^A-Za-z0-9_-]+')
_JSON_TO_PYTHON = {
    'string': 'str',
    'integer': 'int',
    'number': 'float',
    'boolean': 'bool',
    'array': 'list',
    'object': 'dict',
}


class MCPError(OrchifyError, RuntimeError):
    '''Raised when an MCP server cannot be reached or reports an error.

    Attributes:
        code (int): JSON-RPC error code, when the server supplied one.
        data (Any): Extra error data from the server, when supplied.
    '''

    def __init__(self, message: str, code: int = 0, data: Any = None):
        '''Build the error.

        Args:
            message (str): Human readable explanation.
            code (int): JSON-RPC error code (0 when the failure was local).
            data (Any): Extra error data from the server.
        '''
        super().__init__(message)
        self.code = code
        self.data = data


def _tool_name(raw: str, prefix: str = '') -> str:
    '''Turn an MCP tool name into a provider-safe function name.

    Args:
        raw (str): The name the MCP server advertises.
        prefix (str): Optional prefix, e.g. ``'fs_'`` to namespace several servers.

    Returns:
        str: A name matching ``[A-Za-z0-9_-]{1,64}``.
    '''
    name = _NAME_RE.sub('_', f'{prefix}{raw}').strip('_') or 'mcp_tool'
    return name[:64]


def _schema_parameters(schema: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Flatten an MCP ``inputSchema`` into the list ``Tool.build_info`` expects.

    Args:
        schema (Optional[Dict[str, Any]]): JSON Schema of the tool arguments.

    Returns:
        List[Dict[str, Any]]: One entry per top-level property.
    '''
    properties = (schema or {}).get('properties') or {}
    required = set((schema or {}).get('required') or ())
    parameters: List[Dict[str, Any]] = []
    for name, spec in properties.items():
        spec = spec or {}
        parameters.append({
            'name': name,
            'annotation': _JSON_TO_PYTHON.get(str(spec.get('type')), 'str'),
            'description': str(spec.get('description') or ''),
            'required': name in required,
        })
    return parameters


class MCPClient:
    '''A connection to one MCP server.

    Attributes:
        name (str): Label used in logs and for tool-name prefixes.
        timeout (float): Default per-request timeout in seconds.
        server_info (dict): ``serverInfo`` reported by the initialize handshake.
        capabilities (dict): Capabilities reported by the server.
    '''

    def __init__(
        self,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        *,
        url: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        name: str = 'mcp',
        timeout: float = DEFAULT_TIMEOUT,
        auto_start: bool = True,
    ):
        '''Configure a client.

        Args:
            command (Optional[str]): Executable of a stdio MCP server.
            args (Optional[List[str]]): Arguments for ``command``.
            url (Optional[str]): HTTP endpoint of an MCP server instead of a
                child process.
            env (Optional[Dict[str, str]]): Extra environment for the child
                process (merged over ``os.environ``).
            cwd (Optional[str]): Working directory for the child process.
            headers (Optional[Dict[str, str]]): Extra HTTP headers.
            name (str): Label for logs and default tool prefixes.
            timeout (float): Default per-request timeout in seconds.
            auto_start (bool): Start the transport immediately; when False the
                first request starts it.

        Raises:
            ValueError: When neither ``command`` nor ``url`` is given.
        '''
        if not command and not url:
            raise ValueError('MCPClient needs either command= (stdio) or url= (http)')
        self.command = command
        self.args = list(args or [])
        self.url = url
        self.env = dict(env or {})
        self.cwd = cwd
        self.headers = dict(headers or {})
        self.name = name
        self.timeout = float(timeout)
        self.server_info: Dict[str, Any] = {}
        self.capabilities: Dict[str, Any] = {}
        self.tools_cache: List[Dict[str, Any]] = []

        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._http = None
        self._session_id = ''
        self._pending: Dict[int, list] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._counter = 0
        self._closed = False
        self._connected = False
        self._initialized = False
        self._start_lock = threading.Lock()

        if auto_start:
            self.connect()

    # ---------------------------------------------------------------- transport

    def connect(self) -> 'MCPClient':
        '''Start the transport and run the initialize handshake.

        Returns:
            MCPClient: ``self``, so calls can be chained.

        Raises:
            MCPError: When the server cannot be started or rejects the handshake.
        '''
        with self._start_lock:
            if self._initialized and not self._closed:
                return self
            if not self._connected:
                self._closed = False
                if self.url:
                    self._start_http()
                else:
                    self._start_stdio()
                self._connected = True
            self._initialize()
            return self

    def _start_stdio(self) -> None:
        '''Spawn the child process and its reader thread.'''
        environ = dict(os.environ)
        environ.update({str(k): str(v) for k, v in self.env.items()})
        try:
            self._proc = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self.cwd,
                env=environ,
                bufsize=0,
            )
        except OSError as exc:
            raise MCPError(f'MCP server {self.command!r} could not be started: {exc}') from exc
        self._reader = threading.Thread(
            target=self._read_loop, name=f'OrchifyMCP-{self.name}', daemon=True
        )
        self._reader.start()

    def _start_http(self) -> None:
        '''Open the HTTP client.'''
        import httpx

        headers = {
            'content-type': 'application/json',
            'accept': 'application/json, text/event-stream',
        }
        headers.update(self.headers)
        self._http = httpx.Client(timeout=self.timeout, headers=headers, follow_redirects=True)

    def _read_loop(self) -> None:
        '''Read newline-delimited JSON-RPC messages until the server exits.'''
        stream = self._proc.stdout if self._proc is not None else None
        if stream is None:
            return
        for raw in stream:
            line = raw.decode('utf-8', 'replace').strip() if isinstance(raw, bytes) else str(raw).strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                logger.debug('%s: ignoring non-JSON MCP output: %s', self.name, line[:200])
                continue
            self._handle_message(message)
        self._fail_pending(MCPError(f'MCP server {self.name!r} closed the connection'))

    def _handle_message(self, message: Dict[str, Any]) -> None:
        '''Route one decoded JSON-RPC message.

        Args:
            message (Dict[str, Any]): The decoded object.
        '''
        if not isinstance(message, dict):
            return
        if 'id' in message and 'method' in message:
            # Server-initiated request (sampling, roots, ...): politely decline.
            self._send({'jsonrpc': '2.0', 'id': message['id'],
                        'error': {'code': -32601, 'message': 'Method not supported by orchify'}})
            return
        if 'id' not in message:
            return
        with self._lock:
            entry = self._pending.pop(message.get('id'), None)
        if entry is None:
            return
        entry['response'] = message
        entry['event'].set()

    def _fail_pending(self, error: Exception) -> None:
        '''Fail every in-flight request with ``error``.

        Args:
            error (Exception): The failure to report to each waiter.
        '''
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for entry in pending:
            entry['error'] = error
            entry['event'].set()

    def _send(self, message: Dict[str, Any]) -> None:
        '''Write one JSON-RPC message to the transport.

        Args:
            message (Dict[str, Any]): The message to send.

        Raises:
            MCPError: When the transport is closed or the write fails.
        '''
        payload = json.dumps(message, ensure_ascii=False)
        if self.url:
            if self._http is None:
                raise MCPError(f'MCP server {self.name!r} is not connected')
            return
        if self._proc is None or self._proc.stdin is None or self._closed:
            raise MCPError(f'MCP server {self.name!r} is not running')
        with self._write_lock:
            try:
                self._proc.stdin.write((payload + '\n').encode('utf-8'))
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise MCPError(f'MCP server {self.name!r} stopped accepting input: {exc}') from exc

    def _http_post(self, message: Dict[str, Any]) -> None:
        '''Send one message over HTTP and absorb the reply.

        The streamable-HTTP transport answers either with ``application/json``
        or with an SSE stream; both shapes are decoded here and routed through
        the same handler the stdio reader uses.

        Args:
            message (Dict[str, Any]): The JSON-RPC message.

        Raises:
            MCPError: On transport failure.
        '''
        if self._http is None:
            raise MCPError(f'MCP server {self.name!r} is not connected')
        headers = {}
        if self._session_id:
            headers['Mcp-Session-Id'] = self._session_id
        try:
            response = self._http.post(self.url, json=message, headers=headers)
        except Exception as exc:  # httpx raises a wide family of errors
            raise MCPError(f'MCP server {self.name!r} request failed: {exc}') from exc
        session = response.headers.get('mcp-session-id')
        if session:
            self._session_id = session
        if response.status_code >= 400:
            raise MCPError(
                f'MCP server {self.name!r} returned HTTP {response.status_code}: '
                f'{response.text[:300]}'
            )
        content_type = response.headers.get('content-type', '')
        if 'text/event-stream' in content_type:
            for block in response.text.split('\n\n'):
                data = '\n'.join(
                    line[5:].strip() for line in block.splitlines() if line.startswith('data:')
                ).strip()
                if not data:
                    continue
                try:
                    self._handle_message(json.loads(data))
                except ValueError:
                    continue
            return
        body = response.text.strip()
        if body:
            try:
                self._handle_message(json.loads(body))
            except ValueError:
                logger.debug('%s: ignoring non-JSON MCP HTTP body: %s', self.name, body[:200])

    # ------------------------------------------------------------- JSON-RPC API

    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None) -> Any:
        '''Send one request and wait for its result.

        Args:
            method (str): JSON-RPC method name.
            params (Optional[Dict[str, Any]]): Method parameters.
            timeout (Optional[float]): Override the client timeout.

        Returns:
            Any: The ``result`` member of the response.

        Raises:
            MCPError: On a JSON-RPC error, a transport failure or a timeout.
        '''
        if not self._connected:
            self.connect()
        with self._lock:
            self._counter += 1
            request_id = self._counter
            entry = {'event': threading.Event(), 'response': None, 'error': None}
            self._pending[request_id] = entry
        message = {'jsonrpc': '2.0', 'id': request_id, 'method': method}
        if params is not None:
            message['params'] = params
        try:
            if self.url:
                self._http_post(message)
            else:
                self._send(message)
        except Exception:
            with self._lock:
                self._pending.pop(request_id, None)
            raise
        wait_for = self.timeout if timeout is None else float(timeout)
        if not entry['event'].wait(wait_for):
            with self._lock:
                self._pending.pop(request_id, None)
            raise MCPError(f'MCP request {method!r} timed out after {wait_for}s')
        if entry['error'] is not None:
            raise entry['error']
        response = entry['response'] or {}
        if 'error' in response:
            error = response['error'] or {}
            raise MCPError(
                f"MCP server {self.name!r} rejected {method}: {error.get('message', error)}",
                code=int(error.get('code') or 0),
                data=error.get('data'),
            )
        return response.get('result')

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        '''Send a notification (no response expected).

        Args:
            method (str): JSON-RPC method name.
            params (Optional[Dict[str, Any]]): Method parameters.
        '''
        message = {'jsonrpc': '2.0', 'method': method}
        if params is not None:
            message['params'] = params
        if self.url:
            self._http_post(message)
        else:
            self._send(message)

    def _initialize(self) -> None:
        '''Run the initialize handshake and the ``initialized`` notification.'''
        result = self.request('initialize', {
            'protocolVersion': PROTOCOL_VERSION,
            'capabilities': {},
            'clientInfo': dict(_CLIENT_INFO),
        })
        self.server_info = dict((result or {}).get('serverInfo') or {})
        self.capabilities = dict((result or {}).get('capabilities') or {})
        self._initialized = True
        try:
            self.notify('notifications/initialized')
        except MCPError:
            logger.debug('%s: server closed before the initialized notification', self.name)

    def list_tools(self, refresh: bool = False) -> List[Dict[str, Any]]:
        '''Fetch the tool list, following pagination cursors.

        Args:
            refresh (bool): Ignore the cached list.

        Returns:
            List[Dict[str, Any]]: Raw MCP tool descriptions.
        '''
        if self.tools_cache and not refresh:
            return list(self.tools_cache)
        collected: List[Dict[str, Any]] = []
        cursor = None
        while True:
            params = {'cursor': cursor} if cursor else {}
            result = self.request('tools/list', params) or {}
            collected.extend(result.get('tools') or [])
            cursor = result.get('nextCursor')
            if not cursor:
                break
        self.tools_cache = collected
        return list(collected)

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None,
                  timeout: Optional[float] = None) -> str:
        '''Invoke one MCP tool and flatten its content to text.

        Args:
            name (str): MCP tool name (as advertised by ``tools/list``).
            arguments (Optional[Dict[str, Any]]): Arguments for the tool.
            timeout (Optional[float]): Override the client timeout.

        Returns:
            str: The concatenated text content of the result.

        Raises:
            MCPError: When the server reports ``isError`` or the call fails.
        '''
        result = self.request('tools/call', {'name': name, 'arguments': dict(arguments or {})},
                              timeout=timeout) or {}
        pieces: List[str] = []
        for item in result.get('content') or []:
            item = item or {}
            kind = item.get('type')
            if kind == 'text':
                pieces.append(str(item.get('text') or ''))
            elif kind == 'resource':
                resource = item.get('resource') or {}
                pieces.append(str(resource.get('text') or resource.get('uri') or ''))
            else:
                pieces.append(json.dumps(item, ensure_ascii=False))
        if not pieces and result.get('structuredContent') is not None:
            pieces.append(json.dumps(result['structuredContent'], ensure_ascii=False))
        text = '\n'.join(piece for piece in pieces if piece)
        if result.get('isError'):
            raise MCPError(f"MCP tool {name!r} failed: {text or 'no detail'}")
        return text

    def tools(self, prefix: Optional[str] = None, require_approval: bool = False,
              refresh: bool = False) -> List[Tool]:
        '''Wrap every MCP tool in an orchify :class:`Tool`.

        Args:
            prefix (Optional[str]): Name prefix; defaults to ``'<server>_'`` where
                the server label is sanitised, so two servers can expose the same
                tool name without colliding.
            require_approval (bool): Park every call for manual approval.
            refresh (bool): Re-fetch the tool list first.

        Returns:
            List[Tool]: Tools ready for ``Agent(tools=[...])``.
        '''
        if prefix is None:
            prefix = f'{_NAME_RE.sub("_", self.name).strip("_")}_' if self.name else ''
        tools: List[Tool] = []
        for spec in self.list_tools(refresh=refresh):
            raw = str(spec.get('name') or '')
            if not raw:
                continue
            schema = spec.get('inputSchema') or {}
            description = str(spec.get('description') or f'MCP tool {raw}.')
            tools.append(Tool(
                self._make_caller(raw),
                name=_tool_name(raw, prefix),
                description=description,
                parameters=_schema_parameters(schema),
                parameters_schema=schema or None,
                require_approval=require_approval,
                cache=True,
            ))
        return tools

    def _make_caller(self, tool_name: str):
        '''Build the function behind one MCP tool.

        Args:
            tool_name (str): MCP-side tool name.

        Returns:
            Any: A callable accepting the tool arguments as keyword arguments.
        '''
        client = self

        def _call(**kwargs):
            '''Call the MCP tool with the model-provided arguments.'''
            return client.call_tool(tool_name, kwargs)

        _call.__name__ = _tool_name(tool_name) or 'mcp_tool'
        return _call

    def prompts(self) -> List[Dict[str, Any]]:
        '''Fetch the prompt templates the server exposes.

        Returns:
            List[Dict[str, Any]]: Raw MCP prompt descriptions (empty when the
            server has no ``prompts`` capability).
        '''
        try:
            result = self.request('prompts/list', {}) or {}
        except MCPError:
            return []
        return list(result.get('prompts') or [])

    def resources(self) -> List[Dict[str, Any]]:
        '''Fetch the resources the server exposes.

        Returns:
            List[Dict[str, Any]]: Raw MCP resource descriptions (empty when the
            server has no ``resources`` capability).
        '''
        try:
            result = self.request('resources/list', {}) or {}
        except MCPError:
            return []
        return list(result.get('resources') or [])

    # ------------------------------------------------------------------ cleanup

    def close(self) -> None:
        '''Stop the transport and fail any in-flight request.'''
        self._closed = True
        self._connected = False
        proc = self._proc
        self._fail_pending(MCPError(f'MCP server {self.name!r} was closed'))
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._proc = None
        http = self._http
        self._http = None
        if http is not None:
            try:
                http.close()
            except Exception:
                pass
        self._initialized = False

    def __enter__(self) -> 'MCPClient':
        '''Start the client as a context manager.'''
        return self.connect()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        '''Close the client when the ``with`` block ends.'''
        self.close()
        return False

    def __del__(self):  # pragma: no cover - best effort
        '''Best-effort cleanup when the client is finalised.'''
        try:
            if not self._closed:
                self.close()
        except Exception:
            pass
