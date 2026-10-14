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
    optional = ['dotenv', 'docstring_parser']
    for name in required:
        if importlib.util.find_spec(name) is None:
            report.add(FAIL, f'dependency {name}', 'missing; run pip install -r requirements.txt')
        else:
            report.add(PASS, f'dependency {name}', 'importable')
    for name in optional:
        if importlib.util.find_spec(name) is None:
            report.add(WARN, f'dependency {name}', 'missing; some features fall back to defaults')
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
    if args.command == 'version':
        return _version()
    parser.print_help()
    return EXIT_OK


__all__ = ['main']
