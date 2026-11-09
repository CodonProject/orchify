"""Subprocess sandbox for tools."""

import os
import shutil
import uuid

import pytest

from orchify.sandbox import Sandbox, SandboxError, SandboxTimeout, run_sandboxed, sandbox_tool


def add(left: int, right: int) -> int:
    """Add two numbers.

    Args:
        left (int): First number.
        right (int): Second number.

    Returns:
        int: The sum.
    """
    return left + right


def slow(seconds: float) -> str:
    """Sleep, to exercise the timeout.

    Args:
        seconds (float): How long to sleep.

    Returns:
        str: A marker.
    """
    import time
    time.sleep(seconds)
    return 'slept'


def explode(message: str) -> str:
    """Raise, to exercise error reporting.

    Args:
        message (str): Text of the error.

    Returns:
        str: Never returns.
    """
    raise ValueError(message)


def where_am_i() -> dict:
    """Report the child working directory and environment keys.

    Returns:
        dict: Keys the child can see.
    """
    return {'cwd': os.getcwd(), 'keys': sorted(os.environ)}


def write_note(text: str) -> str:
    """Write a file in the child working directory.

    Args:
        text (str): Content to write.

    Returns:
        str: Absolute path written.
    """
    path = os.path.join(os.getcwd(), 'note.txt')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    return path


class _Workspace(object):
    '''A writable scratch directory inside the test tree.'''

    def __enter__(self):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data',
                                 'sandbox-%s' % uuid.uuid4().hex[:8])
        os.makedirs(self.path, exist_ok=True)
        return self.path

    def __exit__(self, *exc_info):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


def test_run_sandboxed_round_trips_a_result():
    '''Arguments and the return value cross the process boundary as JSON.'''
    assert run_sandboxed(add, {'left': 2, 'right': 3}) == 5
    with _Workspace() as directory:
        assert run_sandboxed(write_note, {'text': 'hello'}, workdir=directory) == \
            os.path.join(directory, 'note.txt')


def test_the_child_runs_in_the_configured_directory_without_parent_secrets(monkeypatch):
    '''Secrets in the parent environment are not visible to the child.'''
    monkeypatch.setenv('ORCHIFY_SANDBOX_SECRET', 'do-not-leak')
    with _Workspace() as directory:
        where = run_sandboxed(where_am_i, workdir=directory)
        assert os.path.abspath(where['cwd']) == os.path.abspath(directory)
        assert 'ORCHIFY_SANDBOX_SECRET' not in where['keys']


def test_errors_come_back_as_sandbox_errors():
    '''A child exception is reported with its type and message.'''
    with pytest.raises(SandboxError) as caught:
        run_sandboxed(explode, {'message': 'kaboom'})
    assert 'ValueError' in str(caught.value)
    assert 'kaboom' in str(caught.value)


def test_timeouts_kill_the_child():
    '''A hung call raises SandboxTimeout instead of blocking forever.'''
    with pytest.raises(SandboxTimeout) as caught:
        run_sandboxed(slow, {'seconds': 5}, timeout=0.5)
    assert '0.5' in str(caught.value)


def test_a_function_with_a_closure_is_rejected_before_running():
    '''Closures cannot be shipped to a child, so the tool refuses early.'''
    value = 10

    def uses_closure(number: int) -> int:
        """Add the captured value.

        Args:
            number (int): Number to add.

        Returns:
            int: The sum.
        """
        return number + value

    with pytest.raises(SandboxError) as caught:
        run_sandboxed(uses_closure, {'number': 1})
    assert 'closes over' in str(caught.value)


def test_sandbox_tool_exposes_the_wrapped_signature_and_runs():
    '''The tool carries the original name, description and schema.'''
    sandbox = Sandbox(timeout=10)
    wrapped = sandbox_tool(add, timeout=20)
    info = wrapped.info['function']
    assert info['name'] == 'add'
    assert 'Add two numbers' in info['description']
    assert info['parameters']['properties']['left']
    assert wrapped.execute(left=4, right=6) == 10
    assert wrapped.timeout >= 10
    assert sandbox.snapshot()['timeout'] == 10


def test_sandbox_tool_errors_then_raises_by_default():
    '''Failures surface as exceptions unless the tool is configured otherwise.'''
    wrapped = sandbox_tool(explode)
    with pytest.raises(SandboxError):
        wrapped.execute(message='nope')


def test_sandbox_run_uses_the_configured_workdir():
    '''An explicit working directory is reused across calls.'''
    with _Workspace() as directory:
        sandbox = Sandbox(workdir=directory)
        result = sandbox.run(write_note, {'text': 'kept'})
        assert result == os.path.join(directory, 'note.txt')
        with open(result, encoding='utf-8') as handle:
            assert handle.read() == 'kept'
