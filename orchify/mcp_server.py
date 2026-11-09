"""Expose an Orchify agent (or its tools) as an MCP server.

The counterpart of :mod:`orchify.mcp`: instead of calling somebody else's MCP
server, this module speaks the same JSON-RPC protocol over stdio so any MCP
client (Claude Desktop, an IDE plugin, ``orchify.mcp.MCPClient`` itself) can
call the agent.

Usage::

    from orchify import Agent
    from orchify.mcp_server import serve_stdio

    agent = Agent.from_env('assistant')
    serve_stdio(agent)

or from the command line::

    orchify mcp --agent assistant --expose both

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

import json
import sys
import traceback
from typing import Any, Dict, List, Optional, TextIO

from orchify.utils import get_logger

logger = get_logger(__name__)

PROTOCOL_VERSION = '2025-06-18'

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
    return {
        'name': name,
        'description': info.get('description', '') or getattr(tool, 'description', '') or tool.name,
        'inputSchema': info.get('parameters') or {'type': 'object', 'properties': {}},
    }


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
                 tool_prefix: str = ''):
        if expose not in _MODES:
            raise ValueError(f'expose must be one of {_MODES}, got {expose!r}')
        if agent is None and tools is None:
            raise ValueError('MCPStdioServer needs an agent or explicit tools')
        self.agent = agent
        self.expose = expose
        self.name = name
        self.version = version
        self.tool_prefix = tool_prefix or ('agent_' if expose == 'both' else '')
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
        return {'content': [{'type': 'text', 'text': _as_text(result)}], 'isError': False}

    def handle_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        '''Dispatch one JSON-RPC request method.

        Args:
            method (str): JSON-RPC method name.
            params (Optional[Dict[str, Any]]): Request parameters.

        Returns:
            Any: The JSON-RPC result value.

        Raises:
            MCPRequestError: For unknown methods.
        '''
        params = dict(params or {})
        if method == 'initialize':
            return {
                'protocolVersion': PROTOCOL_VERSION,
                'capabilities': {'tools': {'listChanged': False}},
                'serverInfo': {'name': self.name, 'version': self.version},
            }
        if method == 'ping':
            return {}
        if method == 'tools/list':
            return {'tools': self.tool_list()}
        if method == 'tools/call':
            return self.call_tool(str(params.get('name') or ''), params.get('arguments') or {})
        if method == 'prompts/list':
            return {'prompts': []}
        if method == 'resources/list':
            return {'resources': []}
        raise MCPRequestError(f'method not found: {method}', code=-32601)

    def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        '''Handle one decoded JSON-RPC message.

        Notifications (no ``id``) are acknowledged with ``None`` and never get a
        response frame.

        Args:
            message (Dict[str, Any]): The decoded message.

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
            return None  # notification (e.g. notifications/initialized)
        try:
            result = self.handle_request(str(method), message.get('params'))
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


def serve_stdio(agent: Any = None, *, tools: Optional[list] = None, expose: str = 'agent',
                name: str = 'orchify', version: str = '0.0.1', tool_prefix: str = '',
                instream: Optional[TextIO] = None, outstream: Optional[TextIO] = None) -> None:
    '''Serve an agent over stdio MCP; the one-line entry point for scripts.

    Args:
        agent (Any): The agent to expose.
        tools (Optional[list]): Tools to expose in ``tools``/``both`` mode.
        expose (str): ``'agent'``, ``'tools'`` or ``'both'``.
        name (str): Server name.
        version (str): Server version.
        tool_prefix (str): Prefix applied to exposed agent tool names.
        instream (Optional[TextIO]): Input stream; defaults to stdin.
        outstream (Optional[TextIO]): Output stream; defaults to stdout.
    '''
    server = MCPStdioServer(agent, tools=tools, expose=expose, name=name,
                            version=version, tool_prefix=tool_prefix)
    server.serve(instream, outstream)


__all__ = ['MCPStdioServer', 'MCPRequestError', 'PROTOCOL_VERSION',
           'build_ask_tool', 'build_tool_entry', 'serve_stdio']
