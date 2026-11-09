'''Run a tool in a subprocess sandbox.

A dangerous tool (shell access, code execution, a scraper) can run in its own
interpreter so a crash, a timeout or a stray ``os.chdir`` cannot hurt the agent
process::

    from orchify import sandbox_tool

    def run_python(code: str) -> str:
        """Execute a snippet and return its output."""
        ...

    agent = Agent(name='coder', llm=llm, tools=[sandbox_tool(run_python, timeout=15)])

The child gets a fresh scratch directory as its working directory and a minimal
environment (``PATH``/``SYSTEMROOT`` only), so API keys from ``os.environ`` are
not visible to it. This is *process* isolation, not a security boundary: a
determined snippet can still read the filesystem. For real untrusted input, run
the sandbox on a container or a separate user account.

The wrapped callable must be importable by source: no closures, no lambdas. The
check happens when the tool is built, so a bad tool fails at startup rather
than mid-run.
'''

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import types
from typing import Any, Callable, Dict, Optional

from orchify.tool import Tool

_RUNNER = r'''
import json
import sys

payload = json.loads(sys.stdin.read())
namespace = {'__name__': 'orchify_sandbox'}
exec(payload['source'], namespace)
function = namespace[payload['name']]
try:
    result = function(**payload['arguments'])
except Exception as exc:
    print(json.dumps({'ok': False, 'error': type(exc).__name__, 'message': str(exc)}))
    sys.exit(0)
if not isinstance(result, (dict, list, str, int, float, bool, type(None))):
    result = str(result)
print(json.dumps({'ok': True, 'result': result}))
'''


class SandboxError(RuntimeError):
    '''Raised when a sandboxed call fails, times out or is misconfigured.'''


class SandboxTimeout(SandboxError):
    '''Raised when the child process exceeds its time budget.'''


def _module_imports(func: Callable[..., Any]) -> str:
    '''Recreate the module imports ``func`` refers to inside the child.

    Args:
        func (Callable): The wrapped function.

    Returns:
        str: ``import`` lines for every global it uses that is a module.
    '''
    globals_ = getattr(func, '__globals__', {}) or {}
    names = set(getattr(getattr(func, '__code__', None), 'co_names', ()) or ())
    lines = []
    for name in sorted(names):
        value = globals_.get(name)
        if isinstance(value, types.ModuleType):
            module = getattr(value, '__name__', '')
            if module == name:
                lines.append('import %s' % name)
            elif module and module.isidentifier() is False and name.isidentifier():
                lines.append('import %s as %s' % (module, name))
            elif module:
                lines.append('import %s as %s' % (module, name))
    return '\n'.join(lines)


def _function_source(func: Callable[..., Any]) -> str:
    '''Return the dedented source of ``func``, plus its module imports.

    Args:
        func (Callable): The callable to wrap.

    Returns:
        str: Its source, usable inside ``exec``.

    Raises:
        SandboxError: When the source cannot be recovered (builtins, C code) or
            the function needs a closure, which a child process cannot see.
    '''
    if getattr(func, '__closure__', None):
        raise SandboxError(
            '%r closes over local variables; a sandboxed tool cannot see them. '
            'Pass the values as arguments instead.' % getattr(func, '__name__', func))
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as exc:
        raise SandboxError(
            'cannot read the source of %r (%s); define it in a real module'
            % (getattr(func, '__name__', func), exc)) from exc
    imports = _module_imports(func)
    body = textwrap.dedent(source)
    return (imports + '\n' + body) if imports else body


def _child_env() -> Dict[str, str]:
    '''Minimal environment for the child: no secrets, only what imports need.'''
    env: Dict[str, str] = {}
    for key in ('PATH', 'SYSTEMROOT', 'SYSTEMDRIVE', 'WINDIR', 'TEMP', 'TMP',
                'LANG', 'LC_ALL'):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def run_sandboxed(
    func: Callable[..., Any],
    arguments: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = 10.0,
    workdir: Optional[str] = None,
    keep_workdir: bool = False,
) -> Any:
    '''Call ``func(**arguments)`` inside a child interpreter.

    Args:
        func (Callable): Function to run; must have readable source.
        arguments (Optional[Dict[str, Any]]): Keyword arguments to pass.
        timeout (float): Seconds before the child is killed.
        workdir (Optional[str]): Directory the child runs in; a temporary one is
            created (and removed afterwards) by default.
        keep_workdir (bool): Leave the temporary directory in place.

    Returns:
        Any: The value the function returned, decoded from JSON.

    Raises:
        SandboxTimeout: When the child exceeds ``timeout``.
        SandboxError: On a crash, a non-zero exit or a failed call.
    '''
    source = _function_source(func)
    name = getattr(func, '__name__', '')
    payload = json.dumps({'source': source, 'name': name,
                          'arguments': dict(arguments or {})})
    scratch = None
    directory = workdir or os.environ.get('ORCHIFY_SANDBOX_DIR')
    if not directory:
        scratch = tempfile.mkdtemp(prefix='orchify-sandbox-')
        directory = scratch
    try:
        try:
            completed = subprocess.run(
                [sys.executable, '-I', '-c', _RUNNER],
                input=payload,
                cwd=directory,
                env=_child_env(),
                capture_output=True,
                text=True,
                timeout=max(0.1, float(timeout)),
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeout(
                'sandboxed %s did not finish within %ss' % (name, timeout)) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or '').strip()[-600:]
            raise SandboxError('sandboxed %s exited with %s: %s'
                               % (name, completed.returncode, detail or 'no output'))
        try:
            decoded = json.loads((completed.stdout or '').strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise SandboxError('sandboxed %s produced no JSON result: %s'
                               % (name, (completed.stdout or '')[:300])) from exc
        if not decoded.get('ok'):
            raise SandboxError('%s: %s' % (decoded.get('error') or 'error',
                                           decoded.get('message') or ''))
        return decoded.get('result')
    finally:
        if scratch is not None and not keep_workdir:
            import shutil
            shutil.rmtree(scratch, ignore_errors=True)


class Sandbox:
    '''Bundle of sandbox settings used to build tools.

    Attributes:
        timeout (float): Seconds per call.
        workdir (Optional[str]): Fixed working directory for the child.
        keep_workdir (bool): Keep the per-call scratch directory.
    '''

    def __init__(self, *, timeout: float = 10.0, workdir: Optional[str] = None,
                 keep_workdir: bool = False):
        '''Create the settings bundle.'''
        self.timeout = max(0.1, float(timeout))
        self.workdir = workdir
        self.keep_workdir = bool(keep_workdir)

    def run(self, func: Callable[..., Any], arguments: Optional[Dict[str, Any]] = None) -> Any:
        '''Run one call with these settings.'''
        return run_sandboxed(func, arguments, timeout=self.timeout,
                             workdir=self.workdir, keep_workdir=self.keep_workdir)

    def tool(self, func: Callable[..., Any], **tool_kwargs: Any) -> Tool:
        '''Wrap ``func`` as a :class:`orchify.tool.Tool` that runs in the sandbox.

        Args:
            func (Callable): Function to wrap.
            **tool_kwargs: Extra :class:`Tool` options; ``timeout`` defaults to
                the sandbox timeout plus a small margin for the child to start.

        Returns:
            Tool: The sandboxed tool.
        '''
        source = _function_source(func)
        del source
        info = Tool(func).info['function']
        timeout = tool_kwargs.pop('timeout', None) or (self.timeout + 5.0)

        def run(**arguments: Any) -> Any:
            return run_sandboxed(func, arguments, timeout=self.timeout,
                                 workdir=self.workdir,
                                 keep_workdir=self.keep_workdir)

        tool_kwargs.setdefault('name', info.get('name'))
        tool_kwargs.setdefault('description', info.get('description'))
        tool_kwargs.setdefault('parameters_schema', info.get('parameters'))
        tool_kwargs.setdefault('timeout', timeout)
        tool_kwargs.setdefault('cache', False)
        return Tool(run, **tool_kwargs)

    def snapshot(self) -> Dict[str, Any]:
        '''Return the settings as a plain dict.'''
        return {'timeout': self.timeout, 'workdir': self.workdir,
                'keep_workdir': self.keep_workdir}


def sandbox_tool(func: Callable[..., Any], *, timeout: float = 10.0,
                 workdir: Optional[str] = None, keep_workdir: bool = False,
                 **tool_kwargs: Any) -> Tool:
    '''Wrap ``func`` as a tool that runs in a child interpreter.

    Args:
        func (Callable): Function to run.
        timeout (float): Seconds per call.
        workdir (Optional[str]): Fixed working directory for the child.
        keep_workdir (bool): Keep the per-call scratch directory.
        **tool_kwargs: Extra :class:`Tool` options.

    Returns:
        Tool: The sandboxed tool.
    '''
    return Sandbox(timeout=timeout, workdir=workdir,
                   keep_workdir=keep_workdir).tool(func, **tool_kwargs)
