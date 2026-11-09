import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional

from . import __version__


PASS = 'pass'
WARN = 'warn'
FAIL = 'fail'

STATUS_TEXT = {PASS: 'ok', WARN: 'warn', FAIL: 'fail'}

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_MISSING_KEY = 2
#: Alias used by the commands that only report ``ok``/``error`` (runs, plugins, mcp).
EXIT_ERROR = EXIT_FAILED


class _Report:
    '''Collect the results of the ``doctor`` checks.

    Attributes:
        checks (List[Dict[str, str]]): One entry per check, each with the keys
            ``status``, ``label`` and ``detail``.
    '''

    def __init__(self) -> None:
        '''Create an empty report.'''
        self.checks: List[Dict[str, str]] = []

    def add(self, status: str, label: str, detail: str = '') -> None:
        '''Record one check.

        Args:
            status (str): One of :data:`PASS`, :data:`WARN` or :data:`FAIL`.
            label (str): Short name of the check.
            detail (str): Human readable finding.
        '''
        self.checks.append({'status': status, 'label': label, 'detail': detail})

    def count(self, status: str) -> int:
        '''Count the checks with one status.

        Args:
            status (str): The status to count.

        Returns:
            int: How many checks report that status.
        '''
        return sum(1 for check in self.checks if check['status'] == status)

    @property
    def failed(self) -> bool:
        '''Whether at least one check failed.'''
        return self.count(FAIL) > 0

    def summary_line(self) -> str:
        '''Render a one line summary of the report.

        Returns:
            str: Counts of passing, warning and failing checks.
        '''
        return f'{self.count(PASS)} ok, {self.count(WARN)} warn, {self.count(FAIL)} fail'

    def lines(self) -> List[str]:
        '''Render every check as an aligned line.

        Returns:
            List[str]: Printable lines, one per check.
        '''
        width = max([len(check['label']) for check in self.checks] or [0])
        lines = []
        for check in self.checks:
            status = STATUS_TEXT.get(check['status'], check['status'])
            head = f'[{status:>4}] {check["label"].ljust(width)}'
            lines.append(f'{head}  {check["detail"]}'.rstrip())
        return lines

    def to_dict(self) -> Dict[str, Any]:
        '''Return a JSON serialisable view of the report.

        Returns:
            Dict[str, Any]: The overall verdict and every check.
        '''
        return {
            'ok': not self.failed,
            'passed': self.count(PASS),
            'warnings': self.count(WARN),
            'failed': self.count(FAIL),
            'checks': list(self.checks),
        }


def _check_python(report: _Report) -> None:
    '''Check that the interpreter satisfies the package requirement.

    Args:
        report (_Report): Report to append the finding to.
    '''
    version = platform.python_version()
    if sys.version_info >= (3, 9):
        report.add(PASS, 'python', f'{version} (>= 3.9 required)')
    else:
        report.add(FAIL, 'python', f'{version} is too old; orchify needs 3.9 or newer')


def _check_package(report: _Report) -> None:
    '''Report the imported orchify package and detect a shadowing checkout.

    Args:
        report (_Report): Report to append the findings to.
    '''
    package_dir = os.path.dirname(os.path.abspath(__file__))
    report.add(PASS, 'orchify', f'{__version__} from {package_dir}')
    local_dir = os.path.join(os.getcwd(), 'orchify')
    installed = f'{os.sep}site-packages{os.sep}' in package_dir
    if installed and os.path.isfile(os.path.join(local_dir, '__init__.py')):
        report.add(
            WARN,
            'import path',
            f'the checkout at {local_dir} is shadowed by the installed copy; '
            'run pip install -e . to test local changes',
        )


def _check_dependencies(report: _Report) -> None:
    '''Check that the optional and required third party imports resolve.

    Args:
        report (_Report): Report to append the findings to.
    '''
    required = ['httpx']
    optional = [
        ('dotenv', 'some features fall back to defaults'),
        ('docstring_parser', 'tool descriptions fall back to the first docstring line'),
        ('fastapi', 'orchify serve / create_app need pip install "orchify[web]"'),
        ('uvicorn', 'orchify serve needs pip install "orchify[web]"'),
        ('redis', 'RedisRunStore needs pip install "orchify[redis]"'),
        ('opentelemetry', 'Tracer falls back to in-memory spans; pip install "orchify[otel]"'),
        ('mcp', 'only the official-SDK interoperability path uses it; the client is built in'),
    ]
    for name in required:
        if importlib.util.find_spec(name) is None:
            report.add(FAIL, f'dependency {name}', 'missing; run pip install -r requirements.txt')
        else:
            report.add(PASS, f'dependency {name}', 'importable')
    for name, note in optional:
        if importlib.util.find_spec(name) is None:
            report.add(WARN, f'dependency {name}', f'missing; {note}')
        else:
            report.add(PASS, f'dependency {name}', 'importable')


def _masked_key(key: str) -> str:
    '''Mask an API key for printing.

    Args:
        key (str): The raw key.

    Returns:
        str: The key with its middle removed, or ``'set'`` when it is short.
    '''
    if len(key) <= 12:
        return 'set'
    return f'{key[:6]}...{key[-4:]}'


def _check_credentials(report: _Report) -> None:
    '''Check the credentials and provider settings read from the environment.

    Args:
        report (_Report): Report to append the findings to.
    '''
    from .env import req_key, req_model, req_url

    key = req_key()
    if key:
        report.add(PASS, 'credentials', f'API_KEY/OPENAI_API_KEY {_masked_key(key)}')
    else:
        report.add(
            WARN,
            'credentials',
            'no API_KEY or OPENAI_API_KEY; live runs will fail but offline fakes work',
        )
    report.add(PASS, 'provider', f'base_url={req_url("(unset)")} model={req_model("(unset)")}')


def _check_backend(report: _Report) -> None:
    '''Check the shared web backend thread and event loop.

    Args:
        report (_Report): Report to append the finding to.
    '''
    from .backend import orchify_web_backend

    loop = getattr(orchify_web_backend, '_loop', None)
    thread = getattr(orchify_web_backend, '_thread', None)
    thread_alive = bool(thread is not None and thread.is_alive())
    deadline = time.monotonic() + 2.0
    while thread_alive and loop is not None and not loop.is_running():
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    loop_running = bool(loop is not None and loop.is_running())
    detail = (
        f'thread={getattr(thread, "name", "-")} alive={thread_alive} '
        f'loop={loop_running} timeout={orchify_web_backend.timeout}'
    )
    report.add(PASS if loop_running and thread_alive else FAIL, 'web backend', detail)


def _check_runtime(report: _Report) -> None:
    '''Report the runtime configuration the library is using right now.

    Args:
        report (_Report): Report to append the findings to.
    '''
    from .agent import _read_tool_workers
    from .broker import POLL_INTERVAL, orchify_broker
    from .manager import orchify_manager

    executor = getattr(sys.modules.get('orchify.agent'), '_tool_executor', None)
    report.add(PASS, 'poll interval', f'{POLL_INTERVAL}s')
    report.add(
        PASS,
        'tool workers',
        f'{_read_tool_workers()} configured, executor {"started" if executor is not None else "idle"}',
    )
    report.add(PASS, 'run outcomes', f'limit={orchify_broker.run_outcome_limit}')
    report.add(
        PASS,
        'manager',
        f'enabled={orchify_manager.enabled} history={orchify_manager.history_capacity} '
        f'events={orchify_manager.event_capacity} sink={orchify_manager.sink_path or "off"}',
    )


def _check_gates(report: _Report) -> None:
    '''Report the concurrency gates and rate limiters.

    Args:
        report (_Report): Report to append the findings to.
    '''
    from .gate import gate_stats

    stats = gate_stats()
    for name in sorted(stats):
        item = stats[name]
        if not isinstance(item, dict):
            continue
        if item.get('enabled'):
            detail = (
                f'on limit={item.get("limit")} in_use={item.get("in_use")} '
                f'acquired={item.get("acquired")} waited={item.get("waited")}'
            )
        else:
            detail = 'off'
        report.add(PASS, f'gate {name}', detail)


def _check_plugins(report: _Report) -> None:
    '''Report the registered and loaded plugins.

    Args:
        report (_Report): Report to append the finding to.
    '''
    from .plugin_manager import orchify_plugins

    registered = sorted(orchify_plugins.registered())
    loaded = sorted(orchify_plugins.loaded)
    if registered:
        detail = f'loaded={loaded or "none"} registered={registered}'
    else:
        detail = 'none registered'
    report.add(PASS, 'plugins', detail)


def _check_integrations(report: _Report) -> None:
    '''Report the optional subsystems configured through the environment.

    Args:
        report (_Report): Report to append the findings to.
    '''
    from .policy import policy_from_env
    from .store import STORE_ENV, run_store_from_env

    try:
        store = run_store_from_env()
        report.add(PASS, 'run store',
                   f'{type(store).__name__} ({os.environ.get(STORE_ENV) or "memory"})')
        try:
            store.close()
        except Exception:
            pass
    except Exception as exc:
        report.add(FAIL, 'run store', f'{os.environ.get(STORE_ENV)!r} is unusable: {exc}')

    policy = policy_from_env()
    if policy is None:
        report.add(PASS, 'tool policy', 'off')
    else:
        rules = policy.as_dict().get('rules') or []
        report.add(PASS, 'tool policy',
                   f'{len(rules)} rule(s), default={policy.as_dict().get("default")}')

    sandbox_dir = os.environ.get('ORCHIFY_SANDBOX_DIR') or '(per-call temp dir)'
    report.add(PASS, 'sandbox', f'workdir={sandbox_dir}')

    sink = os.environ.get('ORCHIFY_MANAGER_SINK', '')
    if not sink:
        report.add(PASS, 'manager sink', 'off')
    else:
        parent = os.path.dirname(os.path.abspath(sink)) or os.getcwd()
        if os.access(parent, os.W_OK):
            report.add(PASS, 'manager sink', f'{sink} (parent writable)')
        else:
            report.add(WARN, 'manager sink', f'{sink}: cannot write to {parent}')


def _probe_provider(report: _Report) -> None:
    '''Probe the configured provider over HTTP for the models endpoint.

    Args:
        report (_Report): Report to append the finding to.
    '''
    from .env import req_key, req_url

    key = req_key()
    if not key:
        report.add(FAIL, 'provider probe', 'no API_KEY or OPENAI_API_KEY to probe with')
        return
    try:
        import httpx
    except Exception as exc:
        report.add(FAIL, 'provider probe', f'httpx unavailable: {exc}')
        return
    base = req_url('https://api.openai.com/v1').rstrip('/')
    started = time.perf_counter()
    try:
        response = httpx.get(
            f'{base}/models',
            headers={'Authorization': f'Bearer {key}'},
            timeout=15.0,
        )
    except Exception as exc:
        report.add(FAIL, 'provider probe', f'GET {base}/models failed: {exc}')
        return
    elapsed = time.perf_counter() - started
    status = PASS if response.status_code < 400 else WARN
    report.add(status, 'provider probe', f'GET {base}/models -> {response.status_code} in {elapsed:.2f}s')


def _doctor(live: bool = False, as_json: bool = False) -> int:
    '''Run every environment and configuration check.

    Args:
        live (bool): Also probe the provider over the network.
        as_json (bool): Print the report as JSON instead of lines.

    Returns:
        int: :data:`EXIT_OK` when nothing failed, else :data:`EXIT_FAILED`.
    '''
    report = _Report()
    for check in (
        _check_python,
        _check_package,
        _check_dependencies,
        _check_credentials,
        _check_backend,
        _check_runtime,
        _check_gates,
        _check_plugins,
        _check_integrations,
    ):
        try:
            check(report)
        except Exception as exc:
            report.add(FAIL, getattr(check, '__name__', 'check'), f'crashed: {exc!r}')
    if live:
        _probe_provider(report)
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(f'Orchify doctor - {__version__} on Python {platform.python_version()}')
        for line in report.lines():
            print(line)
        print(f'verdict: {"FAILED" if report.failed else "OK"} ({report.summary_line()})')
    return EXIT_FAILED if report.failed else EXIT_OK


def _demo_offline() -> int:
    '''Run a short, fully offline walkthrough of the main features.

    Runs a scripted answer, a tool round trip and an approval, all with the
    offline fakes, so it needs no credentials and no network.

    Returns:
        int: :data:`EXIT_OK`.
    '''
    from .agent import Agent
    from .api import approve, off, on, status
    from .testing import FakeLLM, final, tool_call
    from .tool import tool

    print('== orchify offline demo ==')

    scripted = Agent(name='scripted', llm=FakeLLM([[final('Hello from the offline demo.')]]))
    print(f'[1] scripted answer: {scripted.ask("Say hello.")}')

    @tool(description='Add two integers.')
    def add(a: int, b: int) -> int:
        '''Add two integers.

        Args:
            a (int): First addend.
            b (int): Second addend.

        Returns:
            int: The sum.
        '''
        return a + b

    planner = Agent(
        name='planner',
        llm=FakeLLM([
            [final('', [tool_call('add', '{"a": 2, "b": 3}')])],
            [final('2 + 3 = 5')],
        ]),
        tools=[add],
    )
    print(f'[2] tool round trip: {planner.ask("What is 2 + 3?")}')

    @tool(description='Deploy the service to a target environment.', require_approval=True)
    def deploy(target: str) -> str:
        '''Deploy the service.

        Args:
            target (str): Environment to deploy to.

        Returns:
            str: Confirmation text.
        '''
        return f'deployed to {target}'

    release = Agent(
        name='release',
        llm=FakeLLM([
            [final('', [tool_call('deploy', '{"target": "prod"}')])],
            [final('deployed to prod')],
        ]),
        tools=[deploy],
    )

    def auto_approve(event: Any) -> None:
        '''Approve each parked tool call, printing the request first.

        Args:
            event (Any): The ``run:approval:required`` event.
        '''
        payload = event.payload
        print(f'[3] approval requested: {payload.get("tool_name")} {payload.get("args")}')
        approve(str(payload.get('approval_id')), reason='offline demo')

    binding = on('run:approval:required', auto_approve)
    try:
        print(f'[3] approved answer: {release.ask("Deploy to prod.")}')
    finally:
        off(binding)

    counts = status().get('counts', {})
    print(f'[4] manager counts: {json.dumps(counts, ensure_ascii=False, default=str)}')
    print('== offline demo done ==')
    return EXIT_OK


def _demo_live() -> int:
    '''Run a single answer through the provider configured in the environment.

    Returns:
        int: :data:`EXIT_OK`, or :data:`EXIT_MISSING_KEY` without credentials.
    '''
    from .agent import Agent
    from .env import req_key, req_model, req_url

    if not req_key():
        print('No API_KEY or OPENAI_API_KEY is set; add one to .env or the environment.')
        return EXIT_MISSING_KEY
    print('== orchify live demo ==')
    print(f'base_url={req_url("(unset)")} model={req_model("(unset)")}')
    agent = Agent.from_env('demo', system_prompt='Answer in one short sentence.')
    print(f'[live] {agent.ask("Say hello and name one thing you can do.", timeout=120)}')
    return EXIT_OK


def _runs(sink: str = '', limit: int = 20, as_json: bool = False) -> int:
    '''Print finished runs with their token usage and cost.

    Without ``--sink`` this reads the in-process manager, which is only useful
    from inside a running application; with ``--sink`` it reads a JSONL file
    written by ``ORCHIFY_MANAGER_SINK``, which is the usual way to inspect a
    finished server.

    Args:
        sink (str): Path of a JSONL sink file; empty means the live manager.
        limit (int): How many of the newest runs to show.
        as_json (bool): Print JSON instead of a table.

    Returns:
        int: :data:`EXIT_OK`, or :data:`EXIT_ERROR` when the sink is unreadable.
    '''
    from .manager import orchify_manager, read_sink

    limit = max(1, int(limit))
    if sink:
        records = read_sink(sink, kinds=('run',))
        if not records:
            print(f'No finished runs in {sink!r} (missing, empty or never finished).')
            return EXIT_ERROR
        rows = [record for record in records if record.get('finished_at')][-limit:]
    else:
        rows = orchify_manager.history(limit=limit)

    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        return EXIT_OK
    if not rows:
        print('No finished runs yet.')
        return EXIT_OK

    headers = ['run_id', 'kind', 'status', 'steps', 'tokens', 'usd', 'model', 'name']
    widths = {header: len(header) for header in headers}
    cells: List[List[str]] = []
    for record in rows:
        row = [
            str(record.get('run_id') or ''),
            str(record.get('kind') or ''),
            str(record.get('status') or ''),
            str(record.get('total_steps') or 0),
            str(record.get('tokens') or 0),
            '{:.6f}'.format(float(record.get('cost_usd') or 0.0)),
            str(record.get('model') or ''),
            str(record.get('name') or ''),
        ]
        cells.append(row)
        for header, value in zip(headers, row):
            widths[header] = max(widths[header], len(value))
    print('  '.join(header.ljust(widths[header]) for header in headers))
    for row in cells:
        print('  '.join(value.ljust(widths[header]) for header, value in zip(headers, row)))
    total = sum(float(record.get('cost_usd') or 0.0) for record in rows)
    print(f'{len(rows)} run(s), ${total:.6f} total (unpriced models count as 0)')
    return EXIT_OK


def _plugins(group: str = '', as_json: bool = False, load: bool = False) -> int:
    '''List installed plugin entry points and the registered plugins.

    Args:
        group (str): Entry point group to inspect; empty means the default group.
        as_json (bool): Print JSON instead of text.
        load (bool): Also import and load every entry point plugin.

    Returns:
        int: :data:`EXIT_OK`, or :data:`EXIT_ERROR` when a plugin fails to load.
    '''
    from .plugin_manager import ENTRY_POINT_GROUP, PluginError, orchify_plugins

    group = group or ENTRY_POINT_GROUP
    entries = orchify_plugins.entry_points(group)
    error = ''
    if load:
        try:
            orchify_plugins.load_entry_points(group)
        except PluginError as exc:
            error = str(exc)
    registered = sorted(orchify_plugins.registered())
    if as_json:
        payload = {
            'group': group,
            'entry_points': entries,
            'registered': registered,
            'error': error,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_ERROR if error else EXIT_OK

    print(f'entry point group: {group}')
    if not entries:
        print('  (none installed - pip install orchify-plugin-<name> to advertise one)')
    for item in entries:
        print('  {:<24} {}'.format(item.get('name', ''), item.get('value', '')))
    print(f'registered plugins: {len(registered)}')
    for name in registered:
        print(f'  {name}')
    if error:
        print(f'error: {error}')
        return EXIT_ERROR
    return EXIT_OK


def _mcp(setup: str = '', agent_name: str = 'orchify', expose: str = 'agent',
         name: str = 'orchify', tool_prefix: str = '', transport: str = 'stdio',
         host: str = '127.0.0.1', port: int = 8010) -> int:
    '''Serve an agent over MCP, on stdio or streamable HTTP.

    Args:
        setup (str): Python file that defines ``agent`` and/or ``tools``.
        agent_name (str): Agent name used when ``--setup`` is not given.
        expose (str): ``agent``, ``tools`` or ``both``.
        name (str): MCP server name reported during ``initialize``.
        tool_prefix (str): Prefix applied to exposed agent tool names.
        transport (str): ``'stdio'`` (default) or ``'http'``.
        host (str): Bind address for ``transport='http'``.
        port (int): Bind port for ``transport='http'``.

    Returns:
        int: :data:`EXIT_ERROR` when the setup file cannot be imported, otherwise
        the process keeps running as a server until stdin reaches EOF (stdio) or
        until interrupted (HTTP).
    '''
    from .mcp_server import MCPHTTPServer, MCPStdioServer, serve_stdio

    agent = None
    tools = None
    if setup:
        module = _load_setup_file(setup)
        if module is None:
            return EXIT_ERROR
        agent = getattr(module, 'agent', None)
        tools = getattr(module, 'tools', None) or None
    if agent is None and not tools:
        from .agent import Agent
        try:
            agent = Agent.from_env(name=agent_name)
        except Exception as exc:
            print(f'Cannot build an agent: {exc}')
            print('Pass --setup FILE with an "agent" or "tools" object, or set the API env vars.')
            return EXIT_ERROR
    try:
        if transport == 'http':
            from .server import create_app, serve

            server = MCPStdioServer(agent, tools=tools, expose=expose, name=name,
                                    tool_prefix=tool_prefix)
            app = create_app(mcp=MCPHTTPServer(server), title='Orchify MCP')
            print(f'orchify mcp: http://{host}:{port}/mcp (Ctrl+C to stop)')
            serve(app, host=host, port=port)
        else:
            serve_stdio(agent, tools=tools, expose=expose, name=name, tool_prefix=tool_prefix)
    except KeyboardInterrupt:
        return EXIT_OK
    return EXIT_OK


def _serve(host: str = '127.0.0.1', port: int = 8010, setup: str = '',
           agent_name: str = 'orchify', keys: str = '', sessions: int = 32,
           tokens: int = 0, cost: float = 0.0, mcp: bool = False) -> int:
    '''Serve the HTTP API (``orchify/server.py``) with uvicorn.

    Args:
        host (str): Bind address.
        port (int): Bind port.
        setup (str): Python file that defines ``agent``, ``agent_factory`` or
            ``tools``.
        agent_name (str): Agent name used when the section above supplies none.
        keys (str): Comma-separated API keys; defaults to ``ORCHIFY_API_KEYS``.
        sessions (int): How many conversations the session pool keeps alive.
        mcp (bool): Also mount the MCP streamable-HTTP endpoint at ``/mcp``.

    Returns:
        int: :data:`EXIT_OK` on a clean shutdown, :data:`EXIT_ERROR` when the
        server cannot start (FastAPI missing, unusable setup file).
    '''
    try:
        from .server import create_app, serve
    except ImportError as exc:
        print(f'Cannot start the API server: {exc}')
        return EXIT_ERROR
    from .agent import Agent

    factory = None
    agent = None
    tools = None
    if setup:
        module = _load_setup_file(setup)
        if module is None:
            return EXIT_ERROR
        factory = getattr(module, 'agent_factory', None)
        agent = getattr(module, 'agent', None)
        tools = getattr(module, 'tools', None) or None
    if factory is None:
        if agent is not None:
            factory = lambda session, _agent=agent: _agent  # noqa: E731
        elif tools:
            factory = lambda session, _tools=list(tools): Agent.from_env(  # noqa: E731
                name=session, tools=_tools)
        else:
            factory = lambda session: Agent.from_env(name=session)  # noqa: E731
    api_keys = [part.strip() for part in (keys or '').split(',') if part.strip()]
    quota = None
    if tokens or cost:
        from .server import Quota
        quota = Quota(max_concurrent=int(os.environ.get('ORCHIFY_MAX_CONCURRENT_RUNS', '0') or 0),
                      max_per_window=int(os.environ.get('ORCHIFY_SESSION_RUNS', '0') or 0),
                      window=float(os.environ.get('ORCHIFY_SESSION_WINDOW', '60') or 60),
                      max_tokens=tokens, max_cost=cost)
    mcp_transport = None
    if mcp:
        from .mcp_server import MCPHTTPServer

        mcp_agent = agent
        if mcp_agent is None:
            try:
                mcp_agent = Agent.from_env(name=f'{agent_name}-mcp')
            except Exception as exc:
                print(f'Cannot build an agent for the MCP endpoint: {exc}')
        if mcp_agent is not None or tools:
            mcp_transport = MCPHTTPServer(agent=mcp_agent, tools=tools)
        else:
            print('The MCP endpoint needs an agent; skipping /mcp.')
    app = create_app(agent_factory=factory, api_keys=api_keys or None,
                     session_capacity=max(1, int(sessions)), quota=quota,
                     mcp=mcp_transport,
                     title='Orchify API')
    print(f'orchify serve: http://{host}:{port} (Ctrl+C to stop)')
    try:
        serve(app, host=host, port=port)
    except KeyboardInterrupt:
        return EXIT_OK
    return EXIT_OK


_CHAT_HELP = '''Commands:
  /help     show this help
  /reset    forget the conversation history
  /status   print the manager status as JSON
  /report   print the manager report
  /runs     list the five most recent runs
  /stop     ask the current run to stop
  /quit     leave the chat'''


def _chat(setup: str = '', agent_name: str = 'orchify', prompt: str = '',
          max_steps: int = 8, agent: Optional[Any] = None) -> int:
    '''Talk to an agent in the terminal.

    Args:
        setup (str): Python file that defines ``agent`` and/or ``tools``.
        agent_name (str): Agent name used when ``--setup`` is not given.
        prompt (str): Run one task and exit instead of starting a REPL.
        max_steps (int): Step budget per message.
        agent (Optional[Any]): Pre-built agent, mainly for tests.

    Returns:
        int: :data:`EXIT_OK`, :data:`EXIT_MISSING_KEY` without credentials, or
        :data:`EXIT_ERROR` when the agent or setup file cannot be built.
    '''
    from .agent import Agent
    from .errors import MissingCredentials

    if agent is None:
        tools = None
        if setup:
            module = _load_setup_file(setup)
            if module is None:
                return EXIT_ERROR
            agent = getattr(module, 'agent', None)
            tools = getattr(module, 'tools', None) or None
        if agent is None:
            try:
                agent = Agent.from_env(name=agent_name, tools=tools)
            except Exception as exc:
                print(f'Cannot build an agent: {exc}')
                print('Pass --setup FILE with an "agent" or "tools" object, or set the API env vars.')
                return EXIT_ERROR

    if prompt:
        try:
            print(agent.ask(prompt, max_steps=max_steps))
        except MissingCredentials as exc:
            print(f'Missing credentials: {exc}')
            return EXIT_MISSING_KEY
        except Exception as exc:
            print(f'error: {type(exc).__name__}: {exc}')
            return EXIT_ERROR
        return EXIT_OK

    print(f'{agent.name} ready (model {getattr(agent, "model", "?")}). '
          f'/help for commands, /quit to leave.')
    while True:
        try:
            line = input('you> ')
        except (EOFError, KeyboardInterrupt):
            print()
            break
        text = line.strip()
        if not text:
            continue
        if text in ('/quit', '/exit'):
            break
        if text == '/help':
            print(_CHAT_HELP)
            continue
        if text == '/reset':
            agent.reset()
            print('history cleared')
            continue
        if text == '/status':
            from .api import status
            print(json.dumps(status(), ensure_ascii=False, indent=2, default=str))
            continue
        if text == '/report':
            from .api import report
            print(report())
            continue
        if text == '/runs':
            from .manager import orchify_manager
            rows = orchify_manager.history(limit=5)
            if not rows:
                print('no finished runs yet')
            for row in rows:
                print(f"{row.get('run_id')}  {row.get('status')}  "
                      f"{row.get('total_steps')} step(s)  ${row.get('cost_usd', 0):.6f}")
            continue
        if text == '/stop':
            print('stop requested' if agent.cancel() else 'no live run')
            continue

        try:
            print(agent.ask(text, max_steps=max_steps))
        except MissingCredentials as exc:
            print(f'Missing credentials: {exc}')
            return EXIT_MISSING_KEY
        except KeyboardInterrupt:
            return EXIT_OK
        except Exception as exc:
            print(f'error: {type(exc).__name__}: {exc}')
    return EXIT_OK


def _eval(data: str = '', setup: str = '', agent_name: str = 'orchify',
          scorer: str = 'contains', max_steps: int = 8,
          as_json: bool = False, agent: Optional[Any] = None) -> int:
    '''Run an evaluation dataset and print the score.

    Args:
        data (str): Dataset file (JSON array or JSONL).
        setup (str): Python file that defines an ``agent`` and/or ``tools``.
        agent_name (str): Agent name used when ``--setup`` is not given.
        scorer (str): ``contains``, ``exact``, ``json`` or ``regex``.
        max_steps (int): Step budget per case.
        as_json (bool): Print the summary as JSON instead of a table.
        agent (Optional[Any]): Pre-built agent, mainly for tests.

    Returns:
        int: :data:`EXIT_OK` when every case passed, :data:`EXIT_FAILED`
        otherwise, :data:`EXIT_ERROR` when the dataset or agent is unusable.
    '''
    from .eval import EvalError, load_dataset, run_dataset

    try:
        cases = load_dataset(data)
    except EvalError as exc:
        print(exc)
        return EXIT_ERROR
    if not cases:
        print(f'{data!r} has no cases.')
        return EXIT_ERROR

    factory = None
    if agent is None:
        if setup:
            module = _load_setup_file(setup)
            if module is None:
                return EXIT_ERROR
            agent = getattr(module, 'agent', None)
            factory = getattr(module, 'agent_factory', None)
            tools = getattr(module, 'tools', None) or None
            if agent is None and factory is None:
                from .agent import Agent
                fixed = tools
                factory = lambda case, _tools=fixed: Agent.from_env(  # noqa: E731
                    name=agent_name, tools=list(_tools or []))
        if agent is None and factory is None:
            from .agent import Agent
            agent = Agent.from_env(name=agent_name)

    from .eval import EvalError as _EvalError
    try:
        report = run_dataset(cases, agent=agent, agent_factory=factory,
                             scorer=scorer, max_steps=max_steps)
    except _EvalError as exc:
        print(exc)
        return EXIT_ERROR
    if as_json:
        print(json.dumps(report.summary(), ensure_ascii=False, indent=2))
    else:
        print(report.render())
    return EXIT_OK if report.passed == len(report) else EXIT_FAILED


def _load_setup_file(path: str) -> Optional[object]:
    '''Import a Python file that supplies an agent or a list of tools.

    Args:
        path (str): File to execute.

    Returns:
        Optional[object]: The imported module, or None when it cannot be loaded.
    '''
    import importlib.util

    spec = importlib.util.spec_from_file_location('orchify_mcp_setup', path)
    if spec is None or spec.loader is None:
        print(f'Cannot import setup file: {path}')
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f'Setup file {path} failed: {type(exc).__name__}: {exc}')
        return None
    return module


def _version() -> int:
    '''Print the orchify and interpreter versions.

    Returns:
        int: :data:`EXIT_OK`.
    '''
    print(f'orchify {__version__} (python {platform.python_version()})')
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    '''Build the command line parser.

    Returns:
        argparse.ArgumentParser: Parser with the ``doctor``, ``demo`` and
        ``version`` subcommands.
    '''
    parser = argparse.ArgumentParser(prog='orchify', description='Orchify command line tools.')
    commands = parser.add_subparsers(dest='command')
    doctor = commands.add_parser('doctor', help='check the environment and configuration')
    doctor.add_argument('--live', action='store_true', help='also probe the provider over the network')
    doctor.add_argument('--json', action='store_true', help='print the report as JSON')
    demo = commands.add_parser('demo', help='run a short walkthrough of agents, tools and approvals')
    demo.add_argument('--live', action='store_true', help='use the provider from the environment')
    runs = commands.add_parser('runs', help='list finished runs with token usage and cost')
    runs.add_argument('--sink', default='', help='JSONL sink file to read (default: the live manager)')
    runs.add_argument('--limit', type=int, default=20, help='how many of the newest runs to show')
    runs.add_argument('--json', action='store_true', help='print the records as JSON')
    evaluate = commands.add_parser('eval', help='run an evaluation dataset through an agent')
    evaluate.add_argument('--data', required=True, help='dataset file (JSON array or JSONL)')
    evaluate.add_argument('--setup', default='', help='Python file defining an "agent" and/or "tools" object')
    evaluate.add_argument('--agent', default='orchify', help='agent name used when --setup is not given')
    evaluate.add_argument('--scorer', default='contains',
                          choices=['contains', 'exact', 'json', 'regex'],
                          help='how answers are scored (default: contains)')
    evaluate.add_argument('--max-steps', type=int, default=8, help='step budget per case')
    evaluate.add_argument('--json', action='store_true', help='print the summary as JSON')
    plugins = commands.add_parser('plugins', help='list plugin entry points and registered plugins')
    plugins.add_argument('--group', default='', help='entry point group (default: orchify.plugins)')
    plugins.add_argument('--json', action='store_true', help='print the report as JSON')
    plugins.add_argument('--load', action='store_true', help='also import and load every entry point plugin')
    mcp = commands.add_parser('mcp', help='serve an agent over MCP on stdio (for Claude Desktop and other clients)')
    mcp.add_argument('--setup', default='', help='Python file defining an "agent" and/or "tools" object')
    mcp.add_argument('--agent', default='orchify', help='agent name used when --setup is not given')
    mcp.add_argument('--expose', choices=['agent', 'tools', 'both'], default='agent',
                     help='what to expose: the whole agent as "ask", its tools, or both')
    mcp.add_argument('--name', default='orchify', help='server name reported during initialize')
    mcp.add_argument('--tool-prefix', default='', help='prefix applied to exposed agent tool names')
    mcp.add_argument('--transport', choices=['stdio', 'http'], default='stdio',
                     help='stdio (default, for Claude Desktop) or streamable HTTP')
    mcp.add_argument('--host', default='127.0.0.1', help='bind address for --transport http')
    mcp.add_argument('--port', type=int, default=8010, help='bind port for --transport http')
    serve = commands.add_parser('serve', help='serve the HTTP API with uvicorn')
    serve.add_argument('--host', default='127.0.0.1', help='bind address (default: 127.0.0.1)')
    serve.add_argument('--port', type=int, default=8010, help='bind port (default: 8010)')
    serve.add_argument('--setup', default='', help='Python file defining "agent", "agent_factory" and/or "tools"')
    serve.add_argument('--agent', default='orchify', help='agent name used when --setup supplies none')
    serve.add_argument('--keys', default='', help='comma-separated API keys (default: ORCHIFY_API_KEYS)')
    serve.add_argument('--sessions', type=int, default=32, help='conversations kept in the session pool')
    serve.add_argument('--tokens', type=int, default=0,
                       help='token budget per session (default: ORCHIFY_SESSION_TOKENS or unlimited)')
    serve.add_argument('--cost', type=float, default=0.0,
                       help='USD budget per session (default: ORCHIFY_SESSION_COST or unlimited)')
    serve.add_argument('--mcp', action='store_true',
                       help='also mount the MCP streamable-HTTP endpoint at /mcp')
    chat = commands.add_parser('chat', help='talk to an agent in the terminal')
    chat.add_argument('--setup', default='', help='Python file defining an "agent" and/or "tools" object')
    chat.add_argument('--agent', default='orchify', help='agent name used when --setup is not given')
    chat.add_argument('--prompt', default='', help='run one task and exit instead of starting a REPL')
    chat.add_argument('--max-steps', type=int, default=8, help='step budget per message (default: 8)')
    commands.add_parser('version', help='print the orchify and Python versions')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    '''Run the ``orchify`` command line interface.

    Args:
        argv (Optional[List[str]]): Arguments to parse; defaults to
            ``sys.argv[1:]``.

    Returns:
        int: The process exit code.
    '''
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        return _doctor(live=args.live, as_json=args.json)
    if args.command == 'demo':
        return _demo_live() if args.live else _demo_offline()
    if args.command == 'runs':
        return _runs(sink=args.sink, limit=args.limit, as_json=args.json)
    if args.command == 'plugins':
        return _plugins(group=args.group, as_json=args.json, load=args.load)
    if args.command == 'mcp':
        return _mcp(setup=args.setup, agent_name=args.agent, expose=args.expose,
                    name=args.name, tool_prefix=args.tool_prefix,
                    transport=args.transport, host=args.host, port=args.port)
    if args.command == 'serve':
        return _serve(host=args.host, port=args.port, setup=args.setup,
                      agent_name=args.agent, keys=args.keys, sessions=args.sessions,
                      tokens=args.tokens, cost=args.cost, mcp=args.mcp)
    if args.command == 'eval':
        return _eval(data=args.data, setup=args.setup, agent_name=args.agent,
                     scorer=args.scorer, max_steps=args.max_steps, as_json=args.json)
    if args.command == 'chat':
        return _chat(setup=args.setup, agent_name=args.agent, prompt=args.prompt,
                     max_steps=args.max_steps)
    if args.command == 'version':
        return _version()
    parser.print_help()
    return EXIT_OK


__all__ = ['main']
