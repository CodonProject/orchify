from typing import Optional


class OrchifyError(Exception):
    '''Base class for the errors orchify raises on its own behalf.

    Catching ``OrchifyError`` gives one stable handle on library failures. Every
    subclass below also inherits from the builtin exception it replaces
    (``RuntimeError``, ``TimeoutError`` or ``ValueError``), so code written
    against the previous error types keeps working.
    '''


class RunAlreadyActive(OrchifyError, RuntimeError):
    '''Raised when a second run would overlap a run already in flight.

    Attributes:
        run_id (str): Identifier of the run that is already active.
        owner (str): Name of the agent or group holding that run.
    '''

    def __init__(self, message: str, run_id: str = '', owner: str = ''):
        '''Build the error.

        Args:
            message (str): Explanation that also names the fix to apply.
            run_id (str): Identifier of the run already in flight.
            owner (str): Name of the agent or group holding the run.
        '''
        super().__init__(message)
        self.run_id = run_id
        self.owner = owner


class WaitOnLoopThread(OrchifyError, RuntimeError):
    '''Raised when a blocking wait is attempted on the orchestrator loop thread.

    That thread is the one executing the run, so blocking it would deadlock the
    caller. Await ``Agent.arun()`` / ``Agent.aask()`` instead, or wait from a
    thread of your own.
    '''


class RunTimeout(OrchifyError, TimeoutError):
    '''Raised by ``ask()`` / ``aask()`` when a run misses its deadline.

    Attributes:
        run_id (str): Identifier of the run that timed out.
        timeout (Optional[float]): The limit that was exceeded, in seconds.
    '''

    def __init__(self, message: str, run_id: str = '', timeout: Optional[float] = None):
        '''Build the error.

        Args:
            message (str): Explanation that names the run and the deadline.
            run_id (str): Identifier of the run that timed out.
            timeout (Optional[float]): The limit that was exceeded, in seconds.
        '''
        super().__init__(message)
        self.run_id = run_id
        self.timeout = timeout


class RunFailed(OrchifyError):
    '''Raised by ``ask()`` / ``aask()`` when the run ended with an error.

    Attributes:
        run_id (str): Identifier of the failed run.
        error (str): Traceback text reported by the runtime.
    '''

    def __init__(self, message: str, run_id: str = '', error: str = ''):
        '''Build the error.

        Args:
            message (str): Short explanation naming the run that failed.
            run_id (str): Identifier of the failed run.
            error (str): Traceback text reported by the runtime.
        '''
        super().__init__(message)
        self.run_id = run_id
        self.error = error


class UnknownRun(OrchifyError, ValueError):
    '''Raised when a run id is required but none was supplied or known.

    Attributes:
        owner (str): Name of the agent or group that has no run to point at.
    '''

    def __init__(self, message: str, owner: str = ''):
        '''Build the error.

        Args:
            message (str): Explanation that names the missing run id.
            owner (str): Name of the agent or group with no run to point at.
        '''
        super().__init__(message)
        self.owner = owner
