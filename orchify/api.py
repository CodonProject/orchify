import threading
from typing import Any, Callable, Dict, List, Optional, Union

from . import broker as _broker
from .broker import HookBinding, orchify_broker
from .manager import orchify_manager
from .utils import get_logger

logger = get_logger(__name__)

__all__ = [
    'on',
    'off',
    'console',
    'ConsoleLoop',
    'approvals',
    'pending_approvals',
    'questions',
    'approve',
    'reject',
    'answer',
    'status',
    'report',
    'runs',
    'run',
    'history',
]


def on(
    event_type: Union[str, List[str]] = '*',
    hook: Optional[Callable[[Any], Any]] = None,
    *,
    run_id: Optional[str] = None,
    mode: str = 'normal',
    once: bool = False,
    agent_name: Optional[str] = None,
    match: Optional[Callable[[Any], bool]] = None,
    owner: str = 'user',
    turn_id: Optional[str] = None,
) -> Union[HookBinding, Callable[[Callable[[Any], Any]], HookBinding]]:
    '''Subscribe a function to orchestration events.

    Works both as a plain call and as a decorator, and unlike
    ``orchify_broker.hook()`` the subscription does not depend on the receiver
    being an enabled plugin, so any owner name is kept.

    Args:
        event_type (Union[str, List[str]]): Event type, ``'*'`` for everything, or
            a list of event types.
        hook (Optional[Callable[[Any], Any]]): Callback taking the event; when
            omitted a decorator is returned instead.
        run_id (Optional[str]): Only deliver events of that run.
        mode (str): ``'normal'``, ``'once'``, ``'agent'`` or ``'hook'``.
        once (bool): Shorthand for ``mode='once'``.
        agent_name (Optional[str]): Only deliver events of that agent.
        match (Optional[Callable[[Any], bool]]): Extra predicate over the event.
        owner (str): Owner label used by ``off()`` and ``remove_hooks()``.
        turn_id (Optional[str]): Only deliver events of that turn.

    Returns:
        Union[HookBinding, Callable[[Callable[[Any], Any]], HookBinding]]: The
        binding, or a decorator when ``hook`` is omitted.

    Raises:
        TypeError: When ``hook`` is given and is not callable.
    '''
    def decorate(func: Callable[[Any], Any]) -> HookBinding:
        return orchify_broker.register_hook(
            event_type,
            func,
            run_id=run_id,
            mode='once' if once else mode,
            agent_name=agent_name,
            match=match,
            owner=owner,
            turn_id=turn_id,
        )

    if hook is None:
        return decorate
    return decorate(hook)


def off(target: Union[HookBinding, str, Callable[[Any], Any]]) -> bool:
    '''Remove event subscriptions.

    Args:
        target (Union[HookBinding, str, Callable[[Any], Any]]): A binding from
            ``on()``/``hook()``, an owner label, or the callback itself.

    Returns:
        bool: True when at least one subscription was removed.

    Raises:
        TypeError: When ``target`` is none of the accepted kinds.
    '''
    if isinstance(target, HookBinding):
        return orchify_broker.remove_binding(target)
    if isinstance(target, str):
        return orchify_broker.remove_hooks(target) > 0
    if callable(target):
        removed = False
        for binding in list(orchify_broker.hooks):
            if binding.func is target and orchify_broker.remove_binding(binding):
                removed = True
        return removed
    raise TypeError(
        f'off() expects a HookBinding, an owner name or a callback, got {type(target)}'
    )


def approvals(*, pending_only: bool = False, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    '''List tool approvals, waiting or already decided.

    Args:
        pending_only (bool): Only return approvals that are still waiting.
        run_id (Optional[str]): Only return the approvals of that run.

    Returns:
        List[Dict[str, Any]]: One record per approval.
    '''
    return orchify_manager.approvals(pending_only=pending_only, run_id=run_id)


def pending_approvals(run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    '''List the tool approvals that are still waiting for an answer.

    Args:
        run_id (Optional[str]): Only return the approvals of that run.

    Returns:
        List[Dict[str, Any]]: One record per waiting approval.
    '''
    return orchify_manager.approvals(pending_only=True, run_id=run_id)


def questions(*, pending_only: bool = False, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    '''List the free-text questions asked through the approval pool.

    Args:
        pending_only (bool): Only return questions that are still waiting.
        run_id (Optional[str]): Only return the questions of that run.

    Returns:
        List[Dict[str, Any]]: One record per question, with the text under
        ``args['question']``.
    '''
    return orchify_manager.questions(pending_only=pending_only, run_id=run_id)


def approve(approval_id: str, *, reason: str = '') -> bool:
    '''Approve a waiting tool call.

    Args:
        approval_id (str): Identifier of the approval.
        reason (str): Note handed back to the agent.

    Returns:
        bool: True when the approval was waiting and is now approved.
    '''
    return orchify_manager.approve(approval_id, reason=reason)


def reject(approval_id: str, *, reason: str = 'denied') -> bool:
    '''Reject a waiting tool call.

    Args:
        approval_id (str): Identifier of the approval.
        reason (str): Note handed back to the agent.

    Returns:
        bool: True when the approval was waiting and is now rejected.
    '''
    return orchify_manager.reject(approval_id, reason=reason)


def answer(approval_id: str, text: str) -> bool:
    '''Answer a waiting question.

    Args:
        approval_id (str): Identifier of the question.
        text (str): Free-text answer handed to the agent.

    Returns:
        bool: True when the question was waiting and is now answered.
    '''
    return orchify_manager.answer(approval_id, text)


def status() -> Dict[str, Any]:
    '''Summarise the live state of the library.

    Returns:
        Dict[str, Any]: Counts, live runs, totals and backend information.
    '''
    return orchify_manager.status()


def report(*, history_limit: int = 10, event_limit: int = 10) -> str:
    '''Render a human-readable status report.

    Args:
        history_limit (int): How many finished runs to include.
        event_limit (int): How many recent events to include.

    Returns:
        str: The rendered report.
    '''
    return orchify_manager.report(history_limit=history_limit, event_limit=event_limit)


def runs(*, kind: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    '''List the runs that are still alive.

    Args:
        kind (Optional[str]): Only ``'agent'`` or ``'group'`` runs.
        status (Optional[str]): Only runs in that status.

    Returns:
        List[Dict[str, Any]]: One record per live run.
    '''
    return orchify_manager.runs(kind=kind, status=status)


def run(run_id: str) -> Optional[Dict[str, Any]]:
    '''Look one run up in the live runs and then in the history.

    Args:
        run_id (str): Identifier of the run.

    Returns:
        Optional[Dict[str, Any]]: The record, or None when the run is unknown.
    '''
    return orchify_manager.run(run_id)


def history(
    *,
    limit: int = 50,
    kind: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    '''List the finished runs, newest first.

    Args:
        limit (int): Maximum number of records.
        kind (Optional[str]): Only ``'agent'`` or ``'group'`` runs.
        status (Optional[str]): Only runs that ended in that status.

    Returns:
        List[Dict[str, Any]]: One record per finished run.
    '''
    return orchify_manager.history(limit=limit, kind=kind, status=status)


class ConsoleLoop:
    '''Surface the approval pool on a terminal.

    The loop looks at the pending questions and tool approvals, asks the
    operator through ``inp`` and delivers the answer back to the parked run.

    Attributes:
        inp (Callable[[str], str]): Prompt function, ``input`` by default.
        out (Callable[[str], Any]): Output function, ``print`` by default.
        poll_interval (float): Seconds between two pools scans.
    '''

    def __init__(
        self,
        inp: Callable[[str], str] = input,
        out: Callable[[str], Any] = print,
        poll_interval: Optional[float] = None,
    ) -> None:
        '''Create a console loop.

        Args:
            inp (Callable[[str], str]): Prompt function used to read an answer.
            out (Callable[[str], Any]): Output function used to report what was
                sent back to the agent.
            poll_interval (Optional[float]): Seconds between two scans; defaults
                to ``ORCHIFY_POLL_INTERVAL``.

        Raises:
            ValueError: When numbers in the interval are not usable by the loop.
        '''
        self.inp = inp
        self.out = out
        self.poll_interval = float(poll_interval) if poll_interval else _broker.POLL_INTERVAL
        if self.poll_interval <= 0:
            raise ValueError(f'poll_interval must be a positive number, got {poll_interval!r}')
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        '''Whether the background thread is alive.

        Returns:
            bool: True while the loop is running.
        '''
        return bool(self._thread and self._thread.is_alive())

    def run_once(self) -> bool:
        '''Handle at most one pending question or approval.

        Returns:
            bool: True when something was handled, False when the pool was
            empty.
        '''
        for item in orchify_manager.questions(pending_only=True):
            question = str(item.get('args', {}).get('question', ''))
            text = self.inp(f"{item.get('agent_name') or 'agent'} asks: {question}\nanswer> ")
            if text is None:
                return False
            delivered = orchify_manager.answer(str(item['approval_id']), str(text))
            self.out(
                f"[answered {item['approval_id']}]" if delivered
                else f"[{item['approval_id']} was already resolved]"
            )
            return True
        for item in orchify_manager.approvals(pending_only=True):
            if item.get('kind') == 'question':
                continue
            label = item.get('tool_name') or item.get('kind') or 'tool'
            text = self.inp(
                f"{item.get('agent_name') or 'agent'} needs approval for "
                f"{label} {item.get('args')}\napprove? [y/N] "
            )
            if text is None:
                return False
            if str(text).strip().lower() in ('y', 'yes', 'true', '1', 'ok', 'approve'):
                delivered = orchify_manager.approve(str(item['approval_id']), reason='approved from console')
                verdict = 'approved'
            else:
                delivered = orchify_manager.reject(str(item['approval_id']), reason='denied from console')
                verdict = 'rejected'
            self.out(
                f"[{verdict} {item['approval_id']}]" if delivered
                else f"[{item['approval_id']} was already resolved]"
            )
            return True
        return False

    def start(self) -> 'ConsoleLoop':
        '''Start the background loop.

        Returns:
            ConsoleLoop: This loop, so calls can be chained.
        '''
        if self.running:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name='OrchifyConsole',
            daemon=True,
        )
        self._thread.start()
        return self

    def _loop(self) -> None:
        '''Scan the pool until ``stop()`` is called.'''
        while not self._stop.is_set():
            try:
                handled = self.run_once()
            except Exception as exc:
                logger.error('[Console] %s', exc)
                handled = False
            if not handled:
                self._stop.wait(self.poll_interval)

    def stop(self, timeout: Optional[float] = 5.0) -> bool:
        '''Stop the background loop and wait for the thread.

        Args:
            timeout (Optional[float]): Seconds to join the thread; None waits
                until it ends.

        Returns:
            bool: True when the thread is no longer alive.
        '''
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        return not (thread and thread.is_alive())

    def join(self, timeout: Optional[float] = None) -> bool:
        '''Wait for the background loop to finish.

        Args:
            timeout (Optional[float]): Seconds to wait; None waits forever.

        Returns:
            bool: True when the thread is no longer alive.
        '''
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def __enter__(self) -> 'ConsoleLoop':
        '''Start the loop when entering a ``with`` block.

        Returns:
            ConsoleLoop: The started loop.
        '''
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        '''Stop the loop when leaving a ``with`` block.

        Args:
            exc_type (Any): Exception type, when one was raised.
            exc (Any): Exception instance, when one was raised.
            tb (Any): Traceback, when one was raised.

        Returns:
            bool: False, so exceptions keep propagating.
        '''
        self.stop()
        return False


def console(
    inp: Callable[[str], str] = input,
    out: Callable[[str], Any] = print,
    poll_interval: Optional[float] = None,
) -> ConsoleLoop:
    '''Build a console loop for the approval pool.

    Args:
        inp (Callable[[str], str]): Prompt function used to read an answer.
        out (Callable[[str], Any]): Output function used to report the decision.
        poll_interval (Optional[float]): Seconds between two scans.

    Returns:
        ConsoleLoop: The loop, not started yet.
    '''
    return ConsoleLoop(inp=inp, out=out, poll_interval=poll_interval)
