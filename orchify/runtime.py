'''Shared runtime plumbing for the Agent and Group runnable classes.

Agent and Group keep a parallel set of runtime mechanics - the one-run-at-a-time
slot, the loop-thread deadlock guard, the run timeout resolution, the call-path
planning/emission around nested calls, and the single-input tool wrapper. Those
pieces live here once so the two classes cannot drift apart:

* :class:`RunSlot` - the per-instance run guard (``RunAlreadyActive``).
* :func:`on_loop_thread` / :func:`assert_not_on_loop_thread` - the orchestrator
  loop-thread checks that back ``WaitOnLoopThread``.
* :func:`resolve_run_timeout` - ``ORCHIFY_RUN_TIMEOUT`` with the unset/invalid
  handling both classes used to duplicate.
* :func:`wait_for_run` - resolve + guard + ``orchify_broker.wait_run()``.
* :func:`plan_outgoing_call` - ``call_chain.plan_call`` plus the
  ``run:call:blocked`` event emitted before a blocked call is refused.
* :func:`prepare_call` - plan one outgoing call and record it on the caller's
  :class:`~orchify.call_chain.CallChain`.
* :func:`set_call_path` - record a call path (and its depth) on an Agent/Group.
* :func:`make_single_input_tool` - build the tool wrapper around a runner whose
  schema advertises a single string parameter (agent/group wrappers, ask_user).

Nothing in here holds state beyond one :class:`RunSlot` per owner instance, and
the module never imports Agent or Group, so it stays free of import cycles.
'''
from typing import Any, Callable, List, Optional
from inspect import Parameter, Signature

import os
import threading

from .backend import orchify_web_backend
from .broker import orchify_broker
from .call_chain import CallBlocked, CallChain, plan_call
from .errors import RunAlreadyActive, WaitOnLoopThread
from .event import RuntimeEvent
from .tool import Tool


class RunSlot:
    '''The one-run-at-a-time guard shared by Agent and Group.

    One slot per owner instance. ``acquire()`` records the active run id and
    refuses a second run while the first is in flight, unless the owner opted
    into concurrent runs; ``release()`` is idempotent and safe from any thread;
    ``locked()`` reports whether a run is in flight.

    Attributes:
        owner (str): Name of the owning Agent or Group.
    '''

    def __init__(self, owner: str) -> None:
        '''Create the slot.

        Args:
            owner (str): Name of the owning Agent or Group, reported as
                ``RunAlreadyActive.owner``.
        '''
        self.owner = owner
        self._lock = threading.Lock()
        self._active_run_id = ''

    @property
    def active_run_id(self) -> str:
        '''The run id recorded by the last acquire (or '' once released).'''
        return self._active_run_id

    def acquire(
        self,
        run_id: str,
        *,
        allow_concurrent: bool = False,
        busy_message: str = '',
    ) -> None:
        '''Take the slot for one run.

        Args:
            run_id (str): The run this slot now represents.
            allow_concurrent (bool): Skip the lock entirely, opting into
                overlapping runs.
            busy_message (str): Message carried by the ``RunAlreadyActive``
                raised when the slot is still held.

        Raises:
            RunAlreadyActive: When the slot is held and ``allow_concurrent`` is
                False, so a second overlapping run is refused.
        '''
        if allow_concurrent:
            self._active_run_id = run_id
            return
        if not self._lock.acquire(blocking=False):
            raise RunAlreadyActive(busy_message, run_id=self._active_run_id, owner=self.owner)
        self._active_run_id = run_id

    def release(self) -> None:
        '''Release the slot; safe to call more than once and from any thread.'''
        self._active_run_id = ''
        try:
            self._lock.release()
        except RuntimeError:
            pass

    def locked(self) -> bool:
        '''Whether a run is currently in flight.'''
        return self._lock.locked()


def on_loop_thread() -> bool:
    '''Whether the current thread is the orchestrator event-loop thread.'''
    return threading.current_thread() is orchify_web_backend._thread


def assert_not_on_loop_thread(message: str) -> None:
    '''Refuse to block when running on the orchestrator loop thread.

    Args:
        message (str): Message carried by the ``WaitOnLoopThread``; callers
            phrase it for their class (the fix differs per entry point).

    Raises:
        WaitOnLoopThread: When the current thread is the loop thread, where
            blocking would deadlock the run being waited for.
    '''
    if on_loop_thread():
        raise WaitOnLoopThread(message)


def resolve_run_timeout(timeout: Optional[float]) -> Optional[float]:
    '''Resolve a wait timeout, falling back to ``ORCHIFY_RUN_TIMEOUT``.

    Args:
        timeout (Optional[float]): Caller-supplied timeout; None falls back to
            the environment.

    Returns:
        Optional[float]: The timeout in seconds, or None to wait forever. An
        unset, unparsable or non-positive environment value means "no timeout".
    '''
    if timeout is not None:
        return timeout
    raw = os.getenv('ORCHIFY_RUN_TIMEOUT')
    if raw is None or raw.strip() == '':
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def wait_for_run(run_id: str, timeout: Optional[float], deadlock_message: str) -> bool:
    '''Wait for one run to finish, refusing to block the orchestrator loop.

    Args:
        run_id (str): The run to wait for.
        timeout (Optional[float]): Seconds to wait; resolved through
            :func:`resolve_run_timeout` (None = wait forever).
        deadlock_message (str): Message used when called on the loop thread.

    Returns:
        bool: True when the run finished (including one that had already
        finished), False for an unknown run or on timeout.

    Raises:
        WaitOnLoopThread: When called from the orchestrator loop thread.
    '''
    assert_not_on_loop_thread(deadlock_message)
    return orchify_broker.wait_run(run_id, timeout=resolve_run_timeout(timeout))


def plan_outgoing_call(
    owner_label: str,
    path: Optional[List[str]],
    callee_label: str,
    max_depth: Optional[int],
    *,
    name: str,
    code: str,
    run_id: str,
) -> List[str]:
    '''Plan one outgoing call, reporting a refusal before it is raised.

    Depth and cycle checks live in ``call_chain.plan_call``; a blocked call is
    reported as a ``run:call:blocked`` event before the error is re-raised.

    Args:
        owner_label (str): Label of the calling entity, e.g. ``'Agent:writer'``.
        path (Optional[List[str]]): The caller's call path.
        callee_label (str): Label of the entity about to be called.
        max_depth (Optional[int]): Depth cap in force (0 or None disables it).
        name (str): Caller name used on the emitted event.
        code (str): Caller code used on the emitted event.
        run_id (str): Run the refusing call belongs to, when known.

    Returns:
        List[str]: The call path the callee will run on.

    Raises:
        CallBlocked: When the call would exceed the depth cap or form a cycle.
    '''
    try:
        return plan_call(owner_label, path, callee_label, max_depth)
    except CallBlocked as exc:
        orchify_broker.emit(RuntimeEvent(
            agent_name=name,
            agent_code=code,
            run_id=run_id,
            turn_id='',
            event_type='run:call:blocked',
            payload={
                'reason': exc.reason,
                'caller': exc.caller,
                'callee': exc.callee,
                'call_path': list(exc.path),
                'call_depth': max(0, len(exc.path) - 1),
                'max_call_depth': max_depth,
            },
        ))
        raise


def prepare_call(caller: Any, callee_label: str, input_data: str, call_type: str) -> List[str]:
    '''Plan an outgoing call and record it on the caller's call chain.

    Args:
        caller (Any): The calling Agent (the only entity that keeps a chain).
        callee_label (str): Label of the entity being called.
        input_data (str): Input handed to the callee.
        call_type (str): ``'agent'`` or ``'group'``, recorded on the frame.

    Returns:
        List[str]: The call path the callee will run on.

    Raises:
        CallBlocked: When the call is refused (a ``run:call:blocked`` event is
            emitted first, see :func:`plan_outgoing_call`).
    '''
    path = caller._outgoing_path(callee_label)
    if caller._call_chain is None:
        caller._call_chain = CallChain(root_caller=f'Agent:{caller.name}')
    caller._call_chain.add_frame(
        caller=f'Agent:{caller.name}',
        callee=callee_label,
        call_type=call_type,
        input_data=input_data,
    )
    return path


def set_call_path(target: Any, path: List[str]) -> None:
    '''Record a call path and its depth on an Agent or Group.

    Args:
        target (Any): The entity inheriting the call path.
        path (List[str]): Call path the entity will run on.
    '''
    target._call_path = path
    target._call_depth = max(0, len(path) - 1)


def make_single_input_tool(
    name: str,
    description: str,
    runner: Callable[..., Any],
    *,
    doc: Optional[str] = None,
    param_name: str = 'input',
    **tool_kwargs: Any,
) -> Tool:
    '''Wrap ``runner`` as a Tool whose schema advertises one string parameter.

    The agent/group tool wrappers and the ``ask_user`` question tool all expose
    a single string argument; this builds the synthetic ``__signature__`` (via
    ``setattr``, which also keeps type checkers happy) in one place.

    Args:
        name (str): Tool name advertised to the model.
        description (str): Tool description advertised to the model.
        runner (Callable): The function executed for a call.
        doc (Optional[str]): Python docstring set on the runner; defaults to
            ``description``.
        param_name (str): Name of the single parameter (``'input'`` by default,
            ``'question'`` for the ask_user tool).
        **tool_kwargs (Any): Extra ``Tool()`` arguments (e.g. ``ask_question``
            or an explicit ``parameters`` schema).

    Returns:
        Tool: The ready tool.
    '''
    setattr(runner, '__name__', name)
    setattr(runner, '__doc__', doc if doc is not None else description)
    setattr(runner, '__signature__', Signature([
        Parameter(name=param_name, kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
    ]))
    return Tool(func=runner, name=name, description=description, **tool_kwargs)
