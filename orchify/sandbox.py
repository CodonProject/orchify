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
import queue
import subprocess
import sys
import tempfile
import textwrap
import threading
import types
from typing import Any, Callable, Dict, Optional, Tuple

from orchify.tool import Tool

#: Default cap on the text a sandboxed call may return, in characters.
DEFAULT_MAX_OUTPUT = 100000

_RUNNER = r'''
import io
import json
import sys

payload = json.loads(sys.stdin.read())
max_output = int(payload.get('max_output') or 100000)


class _Capped(io.TextIOBase):
    def __init__(self):
        self.parts = []
        self.size = 0

    def write(self, text):
        if self.size < max_output:
            chunk = text[:max_output - self.size]
            self.parts.append(chunk)
            self.size += len(chunk)
        return len(text)


namespace = {'__name__': 'orchify_sandbox'}
exec(payload['source'], namespace)
function = namespace[payload['name']]
captured = _Capped()
real_stdout = sys.stdout
sys.stdout = captured
try:
    result = function(**payload['arguments'])
except Exception as exc:
    sys.stdout = real_stdout
    print(json.dumps({'ok': False, 'error': type(exc).__name__, 'message': str(exc)}))
    sys.exit(0)
finally:
    sys.stdout = real_stdout
if not isinstance(result, (dict, list, str, int, float, bool, type(None))):
    result = str(result)
if isinstance(result, str):
    text = result
else:
    text = json.dumps(result)
if len(text) > max_output:
    print(json.dumps({'ok': True, 'result': text[:max_output],
                      'truncated': True, 'original_chars': len(text)}))
else:
    print(json.dumps({'ok': True, 'result': result}))
'''

_WORKER = r'''
import io
import json
import sys


class _Capped(io.TextIOBase):
    def __init__(self, limit):
        self.limit = limit
        self.size = 0

    def write(self, text):
        if self.size < self.limit:
            self.size += len(text[:self.limit - self.size])
        return len(text)


def _run(payload):
    max_output = int(payload.get('max_output') or 100000)
    namespace = {'__name__': 'orchify_sandbox'}
    exec(payload['source'], namespace)
    function = namespace[payload['name']]
    real_stdout = sys.stdout
    sys.stdout = _Capped(max_output)
    try:
        result = function(**payload.get('arguments') or {})
    finally:
        sys.stdout = real_stdout
    if not isinstance(result, (dict, list, str, int, float, bool, type(None))):
        result = str(result)
    text = result if isinstance(result, str) else json.dumps(result)
    if len(text) > max_output:
        return {'ok': True, 'result': text[:max_output],
                'truncated': True, 'original_chars': len(text)}
    return {'ok': True, 'result': result}


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        payload = json.loads(line)
        reply = _run(payload)
    except Exception as exc:
        reply = {'ok': False, 'error': type(exc).__name__, 'message': str(exc)}
    sys.stdout.write(json.dumps(reply) + '\n')
    sys.stdout.flush()
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


def _read_capped(stream: Any, limit: int) -> Tuple[bytes, bool]:
    '''Read ``stream`` to the end, keeping at most ``limit`` bytes.

    Args:
        stream (Any): A binary stream such as ``Popen.stdout``.
        limit (int): Maximum number of bytes to keep.

    Returns:
        Tuple[bytes, bool]: The kept bytes and whether the stream was longer.
    '''
    chunks = []
    total = 0
    while True:
        chunk = stream.read(8192)
        if not chunk:
            break
        if total < limit:
            chunks.append(chunk[:limit - total])
        total += len(chunk)
    return b''.join(chunks), total > limit


def run_sandboxed(
    func: Callable[..., Any],
    arguments: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = 10.0,
    workdir: Optional[str] = None,
    keep_workdir: bool = False,
    max_output: int = DEFAULT_MAX_OUTPUT,
) -> Any:
    '''Call ``func(**arguments)`` inside a child interpreter.

    Args:
        func (Callable): Function to run; must have readable source.
        arguments (Optional[Dict[str, Any]]): Keyword arguments to pass.
        timeout (float): Seconds before the child is killed.
        workdir (Optional[str]): Directory the child runs in; a temporary one is
            created (and removed afterwards) by default.
        keep_workdir (bool): Leave the temporary directory in place.
        max_output (int): Cap on the returned text; longer results are truncated
            and marked instead of being buffered whole.

    Returns:
        Any: The value the function returned, decoded from JSON.

    Raises:
        SandboxTimeout: When the child exceeds ``timeout``.
        SandboxError: On a crash, a non-zero exit or a failed call.
    '''
    source = _function_source(func)
    name = getattr(func, '__name__', '')
    limit = max(32, int(max_output))
    payload = json.dumps({'source': source, 'name': name,
                          'arguments': dict(arguments or {}),
                          'max_output': limit})
    scratch = None
    directory = workdir or os.environ.get('ORCHIFY_SANDBOX_DIR')
    if not directory:
        scratch = tempfile.mkdtemp(prefix='orchify-sandbox-')
        directory = scratch
    try:
        process = subprocess.Popen(
            [sys.executable, '-I', '-c', _RUNNER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=directory,
            env=_child_env(),
        )
        collected: Dict[str, Tuple[bytes, bool]] = {}

        def _drain(stream: Any, key: str, cap: int) -> None:
            collected[key] = _read_capped(stream, cap)

        readers = [
            threading.Thread(target=_drain, args=(process.stdout, 'out', limit * 2), daemon=True),
            threading.Thread(target=_drain, args=(process.stderr, 'err', 8192), daemon=True),
        ]
        for reader in readers:
            reader.start()
        try:
            process.stdin.write(payload.encode('utf-8'))
            process.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass
        try:
            returncode = process.wait(timeout=max(0.1, float(timeout)))
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise SandboxTimeout(
                'sandboxed %s did not finish within %ss' % (name, timeout)) from exc
        for reader in readers:
            reader.join(timeout=5.0)
        stdout, stdout_long = collected.get('out', (b'', False))
        stderr, _ = collected.get('err', (b'', False))
        if returncode != 0:
            detail = stderr.decode('utf-8', 'replace').strip()[-600:]
            raise SandboxError('sandboxed %s exited with %s: %s'
                               % (name, returncode, detail or 'no output'))
        text = stdout.decode('utf-8', 'replace').strip()
        if stdout_long:
            text = text[:limit].strip()
        try:
            decoded = json.loads(text.splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise SandboxError('sandboxed %s produced no JSON result: %s'
                               % (name, text[:300])) from exc
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
        persistent (bool): Keep one child interpreter alive across calls instead
            of paying a new interpreter start per call.
        max_output (int): Cap on the text a call may return.
    '''

    def __init__(self, *, timeout: float = 10.0, workdir: Optional[str] = None,
                 keep_workdir: bool = False, persistent: bool = True,
                 max_output: int = DEFAULT_MAX_OUTPUT):
        '''Create the settings bundle.'''
        self.timeout = max(0.1, float(timeout))
        self.workdir = workdir
        self.keep_workdir = bool(keep_workdir)
        self.persistent = bool(persistent)
        self.max_output = max(32, int(max_output))
        self._proc: Optional[subprocess.Popen] = None
        self._responses: 'queue.Queue' = queue.Queue()
        self._lock = threading.Lock()
        self._scratch: Optional[str] = None
        self._directory = workdir or os.environ.get('ORCHIFY_SANDBOX_DIR')

    def _child_directory(self) -> str:
        '''Return the directory the child runs in, creating the scratch one.'''
        if self._directory:
            return self._directory
        if self._scratch is None:
            self._scratch = tempfile.mkdtemp(prefix='orchify-sandbox-')
        return self._scratch

    def _ensure_worker(self) -> None:
        '''Start the long-lived child when it is missing or dead.'''
        if self._proc is not None and self._proc.poll() is None:
            return
        self._responses = queue.Queue()
        self._proc = subprocess.Popen(
            [sys.executable, '-I', '-c', _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self._child_directory(),
            env=_child_env(),
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._pump, args=(self._proc, self._responses),
                         name='OrchifySandboxWorker', daemon=True).start()

    @staticmethod
    def _pump(process: Any, responses: 'queue.Queue') -> None:
        '''Feed decoded replies from the worker into a queue.'''
        try:
            for line in process.stdout:
                line = (line or '').strip()
                if not line:
                    continue
                try:
                    responses.put(json.loads(line))
                except ValueError:
                    responses.put({'ok': False, 'error': 'ValueError',
                                   'message': line[:200]})
        except Exception:  # the reader dies with the process
            pass
        finally:
            responses.put(None)

    def _stop_worker(self) -> None:
        '''Kill the child and drop its reply queue.'''
        process = self._proc
        self._proc = None
        if process is not None:
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=5.0)
            except Exception:
                pass

    def run(self, func: Callable[..., Any], arguments: Optional[Dict[str, Any]] = None) -> Any:
        '''Run one call with these settings.'''
        if not self.persistent:
            return run_sandboxed(func, arguments, timeout=self.timeout,
                                 workdir=self.workdir, keep_workdir=self.keep_workdir,
                                 max_output=self.max_output)
        source = _function_source(func)
        name = getattr(func, '__name__', '')
        request = json.dumps({'source': source, 'name': name,
                              'arguments': dict(arguments or {}),
                              'max_output': self.max_output}) + '\n'
        with self._lock:
            self._ensure_worker()
            try:
                self._proc.stdin.write(request)
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                self._stop_worker()
                raise SandboxError('sandbox worker died before running %s: %s'
                                   % (name, exc)) from exc
            try:
                decoded = self._responses.get(timeout=self.timeout)
            except queue.Empty:
                # A hung call leaves the worker unusable: kill it and report.
                self._stop_worker()
                raise SandboxTimeout(
                    'sandboxed %s did not finish within %ss' % (name, self.timeout))
        if decoded is None:
            self._stop_worker()
            raise SandboxError('sandbox worker exited while running %s' % name)
        if not decoded.get('ok'):
            raise SandboxError('%s: %s' % (decoded.get('error') or 'error',
                                           decoded.get('message') or ''))
        return decoded.get('result')

    def close(self) -> None:
        '''Stop the worker and remove the scratch directory.'''
        self._stop_worker()
        if self._scratch is not None and not self.keep_workdir:
            import shutil
            shutil.rmtree(self._scratch, ignore_errors=True)
        self._scratch = None

    def __enter__(self) -> 'Sandbox':
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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
            return self.run(func, arguments)

        tool_kwargs.setdefault('name', info.get('name'))
        tool_kwargs.setdefault('description', info.get('description'))
        tool_kwargs.setdefault('parameters_schema', info.get('parameters'))
        tool_kwargs.setdefault('timeout', timeout)
        tool_kwargs.setdefault('cache', False)
        return Tool(run, **tool_kwargs)

    def snapshot(self) -> Dict[str, Any]:
        '''Return the settings as a plain dict.'''
        return {'timeout': self.timeout, 'workdir': self.workdir,
                'keep_workdir': self.keep_workdir, 'persistent': self.persistent,
                'max_output': self.max_output}


def sandbox_tool(func: Callable[..., Any], *, timeout: float = 10.0,
                 workdir: Optional[str] = None, keep_workdir: bool = False,
                 max_output: int = DEFAULT_MAX_OUTPUT,
                 **tool_kwargs: Any) -> Tool:
    '''Wrap ``func`` as a tool that runs in a child interpreter.

    Args:
        func (Callable): Function to run.
        timeout (float): Seconds per call.
        workdir (Optional[str]): Fixed working directory for the child.
        keep_workdir (bool): Keep the per-call scratch directory.
        max_output (int): Cap on the text a call may return.
        **tool_kwargs: Extra :class:`Tool` options.

    Returns:
        Tool: The sandboxed tool.
    '''
    return Sandbox(timeout=timeout, workdir=workdir, keep_workdir=keep_workdir,
                   max_output=max_output).tool(func, **tool_kwargs)

