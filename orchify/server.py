'''A ready-made FastAPI application for orchify agents.

The module turns an agent factory into an HTTP service with the pieces a
deployment actually needs, instead of the three lines a demo needs:

* ``POST /api/chat`` starts a run and returns immediately
* ``GET /api/runs/{run_id}/events`` incremental polling for any client
* ``GET /api/runs/{run_id}/events/stream`` the same events as server-sent events
* ``GET``/``POST`` approval endpoints for human-in-the-loop tools
* ``POST /api/runs/{run_id}/stop|pause|resume`` runtime control of a run
* optional API-key authentication, per-session quotas and ``GET /metrics``

Typical use::

    from orchify.server import create_app

    app = create_app(agent_factory=lambda session: Agent.from_env(name=session))
    # uvicorn module:app

``create_app`` accepts either ``agent`` (one shared agent) or ``agent_factory``
(called once per session, so conversations keep their context). Everything else
defaults to a sane development configuration: no authentication, no quotas, and
metrics enabled.

The contract of every endpoint is documented in ``docs/api-contract.md``.
FastAPI is an optional dependency: ``pip install "orchify[web]"``.
'''

import json
import os
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

try:  # optional extra
    from fastapi import Depends, FastAPI, HTTPException, Query, Request
    from fastapi.responses import PlainTextResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        'orchify.server needs FastAPI: pip install "orchify[web]"'
    ) from exc

import orchify.api as api
from orchify import Agent
from orchify.broker import orchify_broker
from orchify.utils import get_logger

logger = get_logger(__name__)

#: ``run_id`` contains ``#``, which would truncate a URL path as a fragment.
_ID_RAW = '#'
_ID_SAFE = '!'


def url_id(run_id: str) -> str:
    '''Convert a run id into its URL-safe form.

    Args:
        run_id (str): Real run id, e.g. ``'demo#a1b2c3d4'``.

    Returns:
        str: The id with ``#`` replaced by ``!``.
    '''
    return str(run_id).replace(_ID_RAW, _ID_SAFE)


def norm_id(raw: str) -> str:
    '''Convert a URL-safe id back into a real run id.

    Args:
        raw (str): URL form.

    Returns:
        str: The real run id.
    '''
    return str(raw).replace(_ID_SAFE, _ID_RAW)


def with_url_id(record: Dict[str, Any]) -> Dict[str, Any]:
    '''Add ``run_id_url`` to a run record.

    Args:
        record (Dict[str, Any]): Manager run record.

    Returns:
        Dict[str, Any]: A copy carrying the URL-safe id.
    '''
    record = dict(record)
    if record.get('run_id'):
        record['run_id_url'] = url_id(record['run_id'])
    return record


#: Events worth sending to a UI, mapped to a coarse ``kind``.
KEEP_TYPES = {
    'run:start': 'status',
    'run:queued': 'status',
    'run:dequeued': 'status',
    'run:next': 'status',
    'run:paused': 'status',
    'run:resumed': 'status',
    'run:finish': 'status',
    'run:error': 'error',
    'run:approval:required': 'approval',
    'run:approval:resolved': 'approval',
    'run:budget:exceeded': 'budget',
    'run:call:blocked': 'budget',
    'run:llm:retry': 'budget',
    'agent:start': 'status',
    'agent:reason:finish': 'reason',
    'agent:finish': 'status',
    'agent:abort': 'error',
    'tool:call:start': 'tool',
    'tool:call:finish': 'tool',
    'group:start': 'status',
    'group:step': 'reason',
    'group:end': 'status',
    'group:error': 'error',
    'group:agent_call': 'tool',
    'group:call_agent': 'tool',
    'group:call_group': 'tool',
}

_TEXT_KEYS = ('content', 'text', 'answer', 'message', 'reason', 'error', 'summary')
_TERMINAL_TYPES = ('run:finish', 'run:error', 'agent:abort', 'group:error', 'group:end')


class EventLog:
    '''Ring buffer of per-run events, readable by polling or by blocking wait.

    Subscribes with ``background=True`` on the shared dispatcher, so a slow
    consumer never slows a run down. Token-level ``agent:answer`` and
    ``tool:assembly:*`` events are dropped (they are noise for a UI), while
    ``agent:reason:step`` chunks are merged into one ``agent:reason:finish``.

    Attributes:
        per_run (int): Events kept per run (oldest are dropped).
        kept (int): Events stored.
        dropped (int): Events filtered out.
    '''

    def __init__(self, per_run: int = 300, owner: str = 'orchify-server') -> None:
        '''Configure the buffer.

        Args:
            per_run (int): Ring size per run.
            owner (str): Hook owner used for cleanup.
        '''
        self.per_run = per_run
        self.owner = owner
        self._lock = threading.Lock()
        # One condition per run, all sharing ``_lock``: a new event wakes only
        # the streams waiting on that run instead of every connected client.
        self._conds: Dict[str, threading.Condition] = {}
        self._buf: Dict[str, Deque[Dict[str, Any]]] = {}
        self._reason: Dict[str, List[str]] = {}
        self._finished: set = set()
        self._seq = 0
        self._binding: Any = None
        self.kept = 0
        self.dropped = 0

    # ----------------------------------------------------------------- wiring

    def start(self) -> 'EventLog':
        '''Subscribe to the broker (idempotent).

        Returns:
            EventLog: ``self``.
        '''
        if self._binding is None:
            self._binding = api.on('*', self._on_event, owner=self.owner, background=True)
        return self

    def stop(self) -> None:
        '''Unsubscribe.'''
        if self._binding is not None:
            try:
                api.off(self._binding)
            except Exception:
                pass
            self._binding = None

    # ---------------------------------------------------------------- records

    @staticmethod
    def _detail(event: Any, event_type: str) -> str:
        '''Summarise one event in a single line.

        Args:
            event (Any): Broker event.
            event_type (str): Event type.

        Returns:
            str: Human readable summary (possibly empty).
        '''
        payload = getattr(event, 'payload', None) or {}
        if event_type == 'tool:call:start':
            name = getattr(event, 'tool_name', '') or payload.get('tool_name', '')
            args = getattr(event, 'args', None) or payload.get('args')
            if args:
                try:
                    shown = json.dumps(args, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    shown = str(args)
                return f'{name} {shown[:300]}'
            return str(name)
        if event_type == 'tool:call:finish':
            name = getattr(event, 'tool_name', '')
            result = payload.get('result')
            if result is not None:
                return f'{name} -> {str(result)[:300]}'
            return str(name)
        if event_type == 'run:finish':
            parts: List[str] = []
            if payload.get('answer'):
                parts.append(' '.join(str(payload['answer']).split())[:300])
            if payload.get('total_steps'):
                parts.append(f"steps={payload['total_steps']}")
            if str(payload.get('stopped')).lower() == 'true':
                parts.append('stopped')
            return ' | '.join(parts)
        if event_type == 'run:start':
            return str(payload.get('user_input', ''))[:200]
        if event_type == 'agent:finish':
            total = payload.get('total_tokens')
            if total:
                return f'tokens={total}'
        for key in _TEXT_KEYS:
            value = payload.get(key) or getattr(event, key, None)
            if value:
                return ' '.join(str(value).split())[:400]
        return ''

    def _cond_for(self, run_id: str) -> threading.Condition:
        '''Return the condition of one run; call it with ``_lock`` held.

        Args:
            run_id (str): Real run id.

        Returns:
            threading.Condition: Condition shared by every waiter of that run.
        '''
        cond = self._conds.get(run_id)
        if cond is None:
            cond = self._conds[run_id] = threading.Condition(self._lock)
        return cond

    def _on_event(self, event: Any) -> None:
        '''Receive one broker event.

        Args:
            event (Any): Broker event.
        '''
        event_type = str(getattr(event, 'event_type', '') or '')
        run_id = str(getattr(event, 'run_id', '') or '')
        if event_type == 'agent:reason:step':
            content = str(getattr(event, 'content', '') or '')
            if content and run_id:
                with self._lock:
                    self._reason.setdefault(run_id, []).append(content)
            return
        if event_type == 'agent:answer' or event_type.startswith('tool:assembly:'):
            with self._lock:
                self.dropped += 1
            return
        kind = KEEP_TYPES.get(event_type)
        if kind is None or not run_id:
            with self._lock:
                self.dropped += 1
            return
        detail = self._detail(event, event_type)
        with self._lock:
            if event_type == 'agent:reason:finish':
                merged = ' '.join(''.join(self._reason.pop(run_id, [])).split())[:600]
                if merged:
                    detail = merged
            self._seq += 1
            self.kept += 1
            record: Dict[str, Any] = {
                'seq': self._seq,
                'run_id': run_id,
                'type': event_type,
                'kind': kind,
                'step': getattr(event, 'turn', None),
                'ts': getattr(event, 'created_at', None) or time.time(),
            }
            if detail:
                record['detail'] = detail
            buf = self._buf.get(run_id)
            if buf is None:
                buf = self._buf[run_id] = deque(maxlen=self.per_run)
            elif len(buf) == self.per_run:
                self.dropped += 1
            buf.append(record)
            cond = self._cond_for(run_id)
            if event_type in _TERMINAL_TYPES:
                self._finished.add(run_id)
                self._reason.pop(run_id, None)
            cond.notify_all()
            if event_type in _TERMINAL_TYPES:
                # Waiters already woke and saw ``finished``; a late stream re-reads
                # ``_finished`` before it starts waiting, so the condition can go.
                self._conds.pop(run_id, None)

    # ------------------------------------------------------------------- reads

    def since(self, run_id: str, seq: int = 0) -> List[Dict[str, Any]]:
        '''Return the events of one run with a sequence number above ``seq``.

        Args:
            run_id (str): Real run id.
            seq (int): Last sequence number the caller already has.

        Returns:
            List[Dict[str, Any]]: Newest events last.
        '''
        with self._lock:
            buf = self._buf.get(run_id)
            if not buf or buf[-1]['seq'] <= seq:
                return []
            return [dict(item) for item in buf if item['seq'] > seq]

    def finished(self, run_id: str) -> bool:
        '''Whether a run reached a terminal event.

        Args:
            run_id (str): Real run id.

        Returns:
            bool: ``True`` once ``run:finish``/``run:error``/``agent:abort`` was seen.
        '''
        with self._lock:
            return run_id in self._finished

    def wait_for(self, run_id: str, seq: int = 0,
                 timeout: float = 15.0) -> Tuple[List[Dict[str, Any]], bool]:
        '''Block until new events arrive or the run finishes.

        Used by the SSE endpoint so streaming costs no polling.

        Args:
            run_id (str): Real run id.
            seq (int): Last sequence number already sent.
            timeout (float): Seconds to wait before returning empty-handed.

        Returns:
            Tuple[List[Dict[str, Any]], bool]: New events and whether the run is done.
        '''
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._lock:
            cond = self._cond_for(run_id)
        while True:
            with cond:
                buf = self._buf.get(run_id)
                if buf and buf[-1]['seq'] <= seq:
                    items: List[Dict[str, Any]] = []
                else:
                    items = [dict(item) for item in buf if item['seq'] > seq] if buf else []
                done = run_id in self._finished
                if items or done:
                    return items, done
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return [], False
                cond.wait(min(remaining, 1.0))

    def stats(self) -> Dict[str, Any]:
        '''Return buffer statistics.

        Returns:
            Dict[str, Any]: Tracked runs, kept and dropped counts.
        '''
        with self._lock:
            return {
                'runs_tracked': len(self._buf),
                'runs_finished': len(self._finished),
                'events_kept': self.kept,
                'events_dropped': self.dropped,
            }

    def size(self) -> int:
        '''Number of runs buffered.

        Returns:
            int: Tracked run count.
        '''
        with self._lock:
            return len(self._buf)


class SessionPool:
    '''Reuse one agent per session so conversations keep their context.

    Attributes:
        capacity (int): Maximum live sessions (LRU eviction).
        created (int): Sessions ever created.
    '''

    def __init__(self, factory: Callable[[str], Agent], capacity: int = 32) -> None:
        '''Configure the pool.

        Args:
            factory (Callable[[str], Agent]): Builds the agent for a session id.
            capacity (int): LRU capacity.
        '''
        self.factory = factory
        self.capacity = max(1, int(capacity))
        self._items: 'OrderedDict[str, Agent]' = OrderedDict()
        self._lock = threading.Lock()
        self.created = 0

    def get(self, session_id: str) -> Agent:
        '''Return the agent of a session, creating it when needed.

        Args:
            session_id (str): Session identifier.

        Returns:
            Agent: The session agent.
        '''
        with self._lock:
            agent = self._items.get(session_id)
            if agent is not None:
                self._items.move_to_end(session_id)
                return agent
        agent = self.factory(session_id)
        with self._lock:
            self._items[session_id] = agent
            self._items.move_to_end(session_id)
            self.created += 1
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return agent

    def size(self) -> int:
        '''Number of live sessions.

        Returns:
            int: Session count.
        '''
        with self._lock:
            return len(self._items)

    def sessions(self) -> List[str]:
        '''List live session ids, least recently used first.

        Returns:
            List[str]: Session ids.
        '''
        with self._lock:
            return list(self._items)

    def drop(self, session_id: str) -> bool:
        '''Forget one session (its run, if any, keeps running).

        Args:
            session_id (str): Session identifier.

        Returns:
            bool: Whether the session existed.
        '''
        with self._lock:
            return self._items.pop(session_id, None) is not None

    def eviction(self) -> Optional[str]:
        '''Placeholder for API symmetry.

        Returns:
            Optional[str]: Always ``None``; eviction happens inside :meth:`get`.
        '''
        return None


class Quota:
    '''Per-session limits: runs, tokens and cost.

    The token and cost limits are checked when a run starts and charged when a
    run finishes, so a session is stopped on the first run that crosses the
    line instead of being interrupted mid-answer.

    Attributes:
        max_concurrent (int): Runs allowed in flight at once (0 = unlimited).
        max_per_window (int): Runs allowed per session per window (0 = unlimited).
        window (float): Window length in seconds.
        max_tokens (int): Tokens allowed per session in total (0 = unlimited).
        max_cost (float): USD allowed per session in total (0 = unlimited).
    '''

    def __init__(self, max_concurrent: int = 0, max_per_window: int = 0,
                 window: float = 60.0, max_tokens: int = 0,
                 max_cost: float = 0.0) -> None:
        '''Configure the limits.

        Args:
            max_concurrent (int): Global in-flight runs allowed.
            max_per_window (int): Runs per session per window.
            window (float): Window length in seconds.
            max_tokens (int): Tokens per session in total.
            max_cost (float): USD per session in total.
        '''
        self.max_concurrent = max(0, int(max_concurrent))
        self.max_per_window = max(0, int(max_per_window))
        self.window = max(1.0, float(window))
        self.max_tokens = max(0, int(max_tokens or 0))
        self.max_cost = max(0.0, float(max_cost or 0.0))
        self._lock = threading.Lock()
        self._active = 0
        self._events: Dict[str, Deque[float]] = {}
        self._usage: Dict[str, Dict[str, float]] = {}

    def _trim(self, session_id: str, now: float) -> Deque[float]:
        events = self._events.setdefault(session_id, deque())
        cutoff = now - self.window
        while events and events[0] <= cutoff:
            events.popleft()
        return events

    def _usage_locked(self, session_id: str) -> Dict[str, float]:
        usage = self._usage.setdefault(session_id, {'tokens': 0.0, 'cost': 0.0, 'runs': 0.0})
        return usage

    def check(self, session_id: str) -> Optional[str]:
        '''Return a reason when the session may not start a run.

        Args:
            session_id (str): Session identifier.

        Returns:
            Optional[str]: ``None`` when allowed, else a human readable reason.
        '''
        now = time.time()
        with self._lock:
            if self.max_concurrent and self._active >= self.max_concurrent:
                return f'server busy: {self._active} runs already in flight'
            events = self._trim(session_id, now)
            if self.max_per_window and len(events) >= self.max_per_window:
                return (f'session {session_id!r} exceeded {self.max_per_window} runs '
                        f'in {int(self.window)}s')
            usage = self._usage_locked(session_id)
            if self.max_tokens and usage['tokens'] >= self.max_tokens:
                return (f'session {session_id!r} used {int(usage["tokens"])} tokens '
                        f'(limit {self.max_tokens})')
            if self.max_cost and usage['cost'] >= self.max_cost:
                return (f'session {session_id!r} spent ${usage["cost"]:.6f} '
                        f'(limit ${self.max_cost:.6f})')
        return None

    def acquire(self, session_id: str) -> None:
        '''Record a started run.

        Args:
            session_id (str): Session identifier.
        '''
        now = time.time()
        with self._lock:
            self._active += 1
            self._trim(session_id, now).append(now)
            self._usage_locked(session_id)['runs'] += 1

    def release(self) -> None:
        '''Record a finished run.'''
        with self._lock:
            self._active = max(0, self._active - 1)

    def charge(self, session_id: str, *, tokens: int = 0, cost: float = 0.0) -> None:
        '''Add usage to a session.

        Args:
            session_id (str): Session identifier.
            tokens (int): Tokens consumed by the finished run.
            cost (float): USD spent by the finished run.
        '''
        with self._lock:
            usage = self._usage_locked(session_id)
            usage['tokens'] += max(0.0, float(tokens or 0))
            usage['cost'] += max(0.0, float(cost or 0.0))

    def record_run(self, event: Any) -> None:
        '''Charge a ``run:finish`` event to the run's agent name.

        The server names every session agent after its session id, so the event's
        ``agent_name`` is the session key.

        Args:
            event (Any): A ``run:finish`` event.
        '''
        payload = getattr(event, 'payload', None) or {}
        usage = payload.get('usage') or {}
        tokens = usage.get('total_tokens') or (
            (usage.get('prompt_tokens') or 0) + (usage.get('completion_tokens') or 0))
        budget = payload.get('budget') or {}
        cost = budget.get('usd') or 0.0
        session_id = getattr(event, 'agent_name', '') or ''
        if session_id:
            self.charge(session_id, tokens=int(tokens or 0), cost=float(cost or 0.0))

    def attach(self, broker: Any = None) -> Any:
        '''Charge finished runs automatically.

        Args:
            broker (Any): Broker to hook; defaults to the global broker.

        Returns:
            Any: The hook binding, meant for :meth:`detach`.
        '''
        if broker is None:
            from .broker import orchify_broker as broker
        return broker.register_hook('run:finish', self.record_run, owner='quota',
                                    background=True)

    def detach(self, binding: Any) -> bool:
        '''Remove a binding returned by :meth:`attach`.'''
        from .broker import orchify_broker
        return orchify_broker.remove_binding(binding)

    def usage(self, session_id: str) -> Dict[str, float]:
        '''Return the accumulated usage of one session.

        Args:
            session_id (str): Session identifier.

        Returns:
            Dict[str, float]: ``tokens``, ``cost`` and ``runs``.
        '''
        with self._lock:
            return dict(self._usage_locked(session_id))

    def reset(self, session_id: Optional[str] = None) -> None:
        '''Forget usage for one session (or every session).

        Args:
            session_id (Optional[str]): Session identifier, or ``None`` for all.
        '''
        with self._lock:
            if session_id is None:
                self._usage.clear()
                self._events.clear()
            else:
                self._usage.pop(session_id, None)
                self._events.pop(session_id, None)

    def snapshot(self) -> Dict[str, Any]:
        '''Return current counters.

        Returns:
            Dict[str, Any]: Active runs and per-session window usage.
        '''
        now = time.time()
        with self._lock:
            usage = {sid: len(self._trim(sid, now)) for sid in list(self._events)}
            return {
                'active': self._active,
                'max_concurrent': self.max_concurrent,
                'max_per_window': self.max_per_window,
                'window': self.window,
                'max_tokens': self.max_tokens,
                'max_cost': self.max_cost,
                'sessions': {k: v for k, v in usage.items() if v},
                'usage': {k: dict(v) for k, v in self._usage.items()},
            }


class Authenticator:
    '''API-key check for every route that should not be public.

    Keys are compared with :func:`secrets.compare_digest`. With no keys the
    server runs in open mode and logs a warning at startup.

    Attributes:
        keys (frozenset): Accepted keys.
    '''

    def __init__(self, keys: Optional[Iterable[str]] = None) -> None:
        '''Configure accepted keys.

        Args:
            keys (Optional[Iterable[str]]): Accepted API keys.
        '''
        self.keys = frozenset(k for k in (keys or ()) if k)

    @property
    def open(self) -> bool:
        '''Whether authentication is disabled.

        Returns:
            bool: ``True`` when no key is configured.
        '''
        return not self.keys

    def check(self, key: Optional[str]) -> bool:
        '''Validate one presented key.

        Args:
            key (Optional[str]): Bearer token or ``X-API-Key`` value.

        Returns:
            bool: Whether the key is accepted.
        '''
        if self.open:
            return True
        if not key:
            return False
        import secrets as _secrets
        return any(_secrets.compare_digest(str(key), known) for known in self.keys)

    def __call__(self, request: Request) -> None:
        '''FastAPI dependency.

        Args:
            request (Request): The incoming request.

        Raises:
            HTTPException: 401 when the key is missing or wrong.
        '''
        if self.open:
            return None
        header = request.headers.get('authorization') or ''
        key = header[7:].strip() if header.lower().startswith('bearer ') else None
        key = key or request.headers.get('x-api-key')
        if not self.check(key):
            raise HTTPException(status_code=401, detail='missing or invalid API key')
        return None


def _agent_from_parts(session_id: str, tools: Optional[List[Any]],
                      system_prompt: Optional[str]) -> Agent:
    '''Build an agent from ``Agent.from_env`` for one session.

    Args:
        session_id (str): Session identifier (becomes the agent name).
        tools (Optional[List[Any]]): Tools to install.
        system_prompt (Optional[str]): System prompt override.

    Returns:
        Agent: A configured agent.
    '''
    kwargs: Dict[str, Any] = {'name': session_id}
    if tools is not None:
        kwargs['tools'] = list(tools)
    if system_prompt:
        kwargs['system_prompt'] = system_prompt
    return Agent.from_env(**kwargs)


def create_app(*, agent: Optional[Agent] = None,
               agent_factory: Optional[Callable[[str], Agent]] = None,
               tools: Optional[List[Any]] = None,
               system_prompt: Optional[str] = None,
               api_keys: Optional[Iterable[str]] = None,
               session_capacity: int = 32,
               quota: Optional[Quota] = None,
               metrics: Any = None,
               tracer: Any = None,
               events: Optional[EventLog] = None,
               poll_per_run: int = 300,
               title: str = 'Orchify API') -> FastAPI:
    '''Create the FastAPI application.

    Args:
        agent (Optional[Agent]): One shared agent for every session.
        agent_factory (Optional[Callable[[str], Agent]]): Per-session factory.
        tools (Optional[List[Any]]): Tools for the default ``from_env`` factory.
        system_prompt (Optional[str]): System prompt for the default factory.
        api_keys (Optional[Iterable[str]]): API keys; defaults to ``ORCHIFY_API_KEYS``.
        session_capacity (int): LRU capacity of the session pool.
        quota (Optional[Quota]): Per-session/global limits.
        metrics (Any): :class:`orchify.telemetry.Metrics` to expose at ``/metrics``.
        tracer (Any): :class:`orchify.telemetry.Tracer` to trace runs.
        events (Optional[EventLog]): Event buffer to reuse.
        poll_per_run (int): Ring size per run in a new buffer.
        title (str): OpenAPI title.

    Returns:
        FastAPI: The application.
    '''
    keys = api_keys
    if keys is None:
        raw = os.environ.get('ORCHIFY_API_KEYS', '')
        keys = [part.strip() for part in raw.split(',') if part.strip()]
    auth = Authenticator(keys)
    quota = quota if quota is not None else Quota(
        max_concurrent=int(os.environ.get('ORCHIFY_MAX_CONCURRENT_RUNS', '0') or 0),
        max_per_window=int(os.environ.get('ORCHIFY_SESSION_RUNS', '0') or 0),
        window=float(os.environ.get('ORCHIFY_SESSION_WINDOW', '60') or 60),
        max_tokens=int(os.environ.get('ORCHIFY_SESSION_TOKENS', '0') or 0),
        max_cost=float(os.environ.get('ORCHIFY_SESSION_COST', '0') or 0),
    )
    if agent is not None and agent_factory is None:
        agent_factory = lambda _session: agent  # noqa: E731
    if agent_factory is None:
        agent_factory = lambda session: _agent_from_parts(session, tools, system_prompt)  # noqa: E731
    pool = SessionPool(agent_factory, capacity=session_capacity)
    log = events or EventLog(per_run=poll_per_run)

    local_metrics = metrics
    local_tracer = tracer

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        quota_binding = None
        if quota.max_tokens or quota.max_cost:
            quota_binding = quota.attach(orchify_broker)
        if local_metrics is not None:
            local_metrics.attach(orchify_broker)
        if local_tracer is not None:
            local_tracer.attach(orchify_broker)
        log.start()
        if auth.open:
            logger.warning('orchify.server: no API key configured, every endpoint is public')
        try:
            yield
        finally:
            log.stop()
            if quota_binding is not None:
                quota.detach(quota_binding)
            if local_metrics is not None:
                local_metrics.detach()
            if local_tracer is not None:
                local_tracer.detach()

    app = FastAPI(title=title, version='0.1.0', lifespan=lifespan)
    app.state.pool = pool
    app.state.events = log
    app.state.quota = quota
    app.state.auth = auth
    app.state.metrics = local_metrics

    class ChatIn(BaseModel):
        '''Request body of ``POST /api/chat``.'''

        text: str = Field(..., min_length=1, description='user input')
        session_id: str = Field(
            'default',
            pattern=r'^[A-Za-z0-9_-]{1,32}$',
            description='session identifier; equal values share context',
        )
        max_steps: int = Field(8, ge=1, le=30, description='step ceiling')

    class AnswerIn(BaseModel):
        '''Request body of ``POST /api/approvals/{id}/answer``.'''

        text: str = Field(..., min_length=1)

    class ReasonIn(BaseModel):
        '''Request body of the approve/reject endpoints.'''

        reason: str = ''

    # ------------------------------------------------------------- public routes

    @app.get('/api/health')
    def health() -> Dict[str, Any]:
        '''Liveness probe with a deployment summary.

        Returns:
            Dict[str, Any]: Status, model, sessions, events, quota and orchify state.
        '''
        return {
            'ok': True,
            'model': os.environ.get('MODEL_ID', ''),
            'base_url': os.environ.get('BASE_URL', ''),
            'auth': 'open' if auth.open else 'api-key',
            'sessions': pool.size(),
            'sessions_created': pool.created,
            'events': log.stats(),
            'quota': quota.snapshot(),
            'orchify': api.status(),
        }

    @app.get('/metrics', response_class=PlainTextResponse)
    def scrape() -> PlainTextResponse:
        '''Prometheus scrape endpoint.

        Returns:
            PlainTextResponse: Metrics in text exposition format.
        '''
        if local_metrics is None:
            from orchify.telemetry import Metrics as _Metrics
            fallback = _Metrics('orchify').attach(orchify_broker)
            return PlainTextResponse(fallback.render())
        return PlainTextResponse(local_metrics.render())

    # --------------------------------------------------------------- run routes

    @app.post('/api/chat')
    def chat(body: ChatIn, _: None = Depends(auth)) -> Dict[str, Any]:
        '''Start a run and return its id immediately.

        Args:
            body (ChatIn): The task.
            _ (None): Authentication dependency.

        Returns:
            Dict[str, Any]: ``run_id``, ``run_id_url`` and ``session_id``.

        Raises:
            HTTPException: 409 when the session is busy, 429 over quota, 500 on error.
        '''
        session_agent = pool.get(body.session_id)
        reason = quota.check(body.session_id)
        if reason:
            raise HTTPException(status_code=429, detail=reason)
        try:
            run_id = session_agent.run(body.text, max_steps=body.max_steps)
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            if 'AlreadyActive' in name:
                raise HTTPException(
                    status_code=409,
                    detail='this session already has a run in flight',
                ) from exc
            raise HTTPException(status_code=500, detail=f'{name}: {exc}') from exc
        quota.acquire(body.session_id)
        return {
            'run_id': run_id,
            'run_id_url': url_id(run_id),
            'session_id': body.session_id,
        }

    @app.get('/api/sessions')
    def list_sessions(_: None = Depends(auth)) -> Dict[str, Any]:
        '''List live sessions, least recently used first.

        Args:
            _ (None): Authentication dependency.

        Returns:
            Dict[str, Any]: ``total`` and ``items``.
        '''
        return {'total': pool.size(), 'items': pool.sessions()}

    @app.delete('/api/sessions/{session_id}')
    def drop_session(session_id: str, _: None = Depends(auth)) -> Dict[str, Any]:
        '''Forget one session's conversation (a running run keeps going).

        Args:
            session_id (str): Session identifier.
            _ (None): Authentication dependency.

        Returns:
            Dict[str, Any]: ``ok`` and ``dropped``.
        '''
        return {'ok': True, 'dropped': pool.drop(session_id)}

    @app.get('/api/runs')
    def list_runs(status: Optional[str] = None,
                  limit: int = Query(20, ge=1, le=100),
                  _: None = Depends(auth)) -> Dict[str, Any]:
        '''List runs, newest first, merging live and finished records.

        Args:
            status (Optional[str]): Filter by status.
            limit (int): Maximum rows.
            _ (None): Authentication dependency.

        Returns:
            Dict[str, Any]: ``total``, ``active`` and ``items``.
        '''
        live = api.runs(status=status)
        finished = api.history(limit=limit, status=status)
        rows = [with_url_id(row) for row in live + finished]
        rows.sort(key=lambda row: row.get('started_at') or 0, reverse=True)
        rows = rows[:limit]
        return {'total': len(rows), 'active': len(live), 'items': rows}

    @app.get('/api/runs/{run_id}')
    def get_run(run_id: str, _: None = Depends(auth)) -> Dict[str, Any]:
        '''Return one run record (status, answer, usage, cost).

        Args:
            run_id (str): URL-safe run id.
            _ (None): Authentication dependency.

        Returns:
            Dict[str, Any]: The record.

        Raises:
            HTTPException: 404 when unknown.
        '''
        record = api.run(norm_id(run_id))
        if record is None:
            raise HTTPException(status_code=404, detail=f'unknown run: {run_id}')
        return with_url_id(record)

    @app.get('/api/runs/{run_id}/events')
    def get_events(run_id: str, since: int = 0,
                   _: None = Depends(auth)) -> Dict[str, Any]:
        '''Poll new events of one run.

        Args:
            run_id (str): URL-safe run id.
            since (int): Last sequence number already seen.
            _ (None): Authentication dependency.

        Returns:
            Dict[str, Any]: ``run_id``, ``items`` and ``last_seq``.
        '''
        real = norm_id(run_id)
        items = log.since(real, since)
        last = items[-1]['seq'] if items else since
        return {'run_id': real, 'items': items, 'last_seq': last,
                'finished': log.finished(real)}

    @app.get('/api/runs/{run_id}/events/stream')
    def stream_events(run_id: str, since: int = 0,
                      _: None = Depends(auth)) -> StreamingResponse:
        '''Stream the events of one run as server-sent events.

        Emits ``data: {json}`` per event, a ``: keep-alive`` comment when idle,
        ``event: done`` once the run reached a terminal event, and stops after
        ``ORCHIFY_SSE_TIMEOUT`` seconds without progress (default 1800).

        Args:
            run_id (str): URL-safe run id.
            since (int): Last sequence number already seen.
            _ (None): Authentication dependency.

        Returns:
            StreamingResponse: ``text/event-stream`` body.
        '''
        real = norm_id(run_id)
        idle_limit = float(os.environ.get('ORCHIFY_SSE_TIMEOUT', '1800') or 1800)
        run_known = api.run(real) is not None

        def generate():
            cursor = int(since)
            idle_since = time.monotonic()
            yield ': connected\n\n'
            while True:
                items, done = log.wait_for(real, cursor, timeout=15.0)
                for item in items:
                    cursor = max(cursor, int(item['seq']))
                    idle_since = time.monotonic()
                    yield f'data: {json.dumps(item, ensure_ascii=False)}\n\n'
                if done and not items:
                    record = api.run(real) or {}
                    summary = {
                        'run_id': real,
                        'status': record.get('status', 'finished'),
                        'tokens': record.get('tokens', 0),
                        'cost_usd': record.get('cost_usd', 0.0),
                    }
                    yield f'event: done\ndata: {json.dumps(summary, ensure_ascii=False)}\n\n'
                    return
                if not items:
                    if not run_known and not log.finished(real):
                        run_known = api.run(real) is not None
                        if not run_known:
                            payload = {'error': f'unknown run: {run_id}'}
                            yield f'event: error\ndata: {json.dumps(payload)}\n\n'
                            return
                    if time.monotonic() - idle_since > idle_limit:
                        yield 'event: timeout\ndata: {}\n\n'
                        return
                    yield ': keep-alive\n\n'

        return StreamingResponse(generate(), media_type='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        })

    @app.post('/api/runs/{run_id}/control/{action}')
    def control(run_id: str, action: str, _: None = Depends(auth)) -> Dict[str, Any]:
        '''Send ``stop``, ``pause`` or ``resume`` to a run.

        Args:
            run_id (str): URL-safe run id.
            action (str): ``stop``, ``pause`` or ``resume``.
            _ (None): Authentication dependency.

        Returns:
            Dict[str, Any]: ``ok`` and ``accepted``.

        Raises:
            HTTPException: 400 for an unknown action, 404 when the run is gone.
        '''
        if action not in ('stop', 'pause', 'resume'):
            raise HTTPException(status_code=400, detail=f'unknown action: {action}')
        real = norm_id(run_id)
        if api.run(real) is None:
            raise HTTPException(status_code=404, detail=f'unknown run: {run_id}')
        accepted = bool(orchify_broker.feedback('control:%s' % action, run_id=real))
        return {'ok': True, 'accepted': accepted, 'action': action, 'run_id': real}

    @app.get('/api/approvals')
    def list_approvals(run_id: Optional[str] = None,
                       _: None = Depends(auth)) -> Dict[str, Any]:
        '''List pending approvals and questions.

        Args:
            run_id (Optional[str]): Only this run.
            _ (None): Authentication dependency.

        Returns:
            Dict[str, Any]: ``total`` and ``items``.
        '''
        items = api.pending_approvals(norm_id(run_id) if run_id else None)
        return {'total': len(items), 'items': [with_url_id(item) for item in items]}

    @app.post('/api/approvals/{approval_id}/approve')
    def approve(approval_id: str, body: ReasonIn,
                _: None = Depends(auth)) -> Dict[str, Any]:
        '''Approve one pending tool call.

        Args:
            approval_id (str): Approval id.
            body (ReasonIn): Optional reason.
            _ (None): Authentication dependency.

        Returns:
            Dict[str, Any]: Confirmation.

        Raises:
            HTTPException: 404 when unknown or already resolved.
        '''
        if not api.approve(approval_id, reason=body.reason):
            raise HTTPException(status_code=404, detail='unknown or already resolved')
        return {'ok': True, 'approval_id': approval_id, 'action': 'approve'}

    @app.post('/api/approvals/{approval_id}/reject')
    def reject(approval_id: str, body: ReasonIn,
               _: None = Depends(auth)) -> Dict[str, Any]:
        '''Reject one pending tool call; the reason is returned to the agent.

        Args:
            approval_id (str): Approval id.
            body (ReasonIn): Optional reason.
            _ (None): Authentication dependency.

        Returns:
            Dict[str, Any]: Confirmation.

        Raises:
            HTTPException: 404 when unknown or already resolved.
        '''
        if not api.reject(approval_id, reason=body.reason or 'denied'):
            raise HTTPException(status_code=404, detail='unknown or already resolved')
        return {'ok': True, 'approval_id': approval_id, 'action': 'reject'}

    @app.post('/api/approvals/{approval_id}/answer')
    def answer(approval_id: str, body: AnswerIn,
               _: None = Depends(auth)) -> Dict[str, Any]:
        '''Answer a question asked through ``ask_user``.

        Args:
            approval_id (str): Question id.
            body (AnswerIn): The answer.
            _ (None): Authentication dependency.

        Returns:
            Dict[str, Any]: Confirmation.

        Raises:
            HTTPException: 404 when unknown or already answered.
        '''
        if not api.answer(approval_id, body.text):
            raise HTTPException(status_code=404, detail='unknown or already answered')
        return {'ok': True, 'approval_id': approval_id, 'action': 'answer'}

    return app


def serve(app: Optional[FastAPI] = None, *, host: Optional[str] = None,
          port: Optional[int] = None, **kwargs: Any) -> None:
    '''Run the application with uvicorn.

    Args:
        app (Optional[FastAPI]): An application; created from ``kwargs`` when omitted.
        host (Optional[str]): Bind address (default ``HOST`` or ``127.0.0.1``).
        port (Optional[int]): Bind port (default ``PORT`` or ``8010``).
        **kwargs (Any): Forwarded to :func:`create_app`.
    '''
    import uvicorn

    application = app or create_app(**kwargs)
    host = host or os.environ.get('HOST', '127.0.0.1')
    port = int(port or os.environ.get('PORT', '8010'))
    logger.info('orchify API -> http://%s:%d/docs', host, port)
    uvicorn.run(application, host=host, port=port, log_level='info')
