import json
import os
import threading
import time
import weakref
from collections import deque
from fnmatch import fnmatchcase
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .backend import orchify_web_backend
from .broker import orchify_broker
from .gate import gate_stats
from .utils import get_logger

__all__ = ['OrchifyManager', 'orchify_manager', 'RUN_STATUSES', 'read_sink']

logger = get_logger(__name__)


RUNNING = 'running'
QUEUED = 'queued'
PAUSED = 'paused'
AWAITING_APPROVAL = 'awaiting_approval'
DONE = 'done'
STOPPED = 'stopped'
ERROR = 'error'

RUN_STATUSES = (QUEUED, RUNNING, PAUSED, AWAITING_APPROVAL, DONE, STOPPED, ERROR)
FINISHED_STATUSES = (DONE, STOPPED, ERROR)

DEFAULT_HISTORY = 200
DEFAULT_EVENTS = 500

MAX_STRING = 2000
MAX_ANSWER = 20000
MAX_PAYLOAD_DEPTH = 6


def _read_int_env(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _read_bool_env(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in (
        '0', 'false', 'no', 'off', 'disable', 'disabled', 'none'
    )


def _read_str_env(name: str, default: str = '') -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip()


def _truncate(value: Any, limit: int = MAX_STRING) -> str:
    text = '' if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + '...(truncated)'.format()


def _jsonable(value: Any, depth: int = 0) -> Any:
    '''Best-effort conversion of any event payload into JSON-safe data.'''
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value)
    if depth >= MAX_PAYLOAD_DEPTH:
        return repr(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item, depth + 1) for item in value]
    for attr in ('to_dict', 'model_dump', 'dict'):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _jsonable(method(), depth + 1)
            except Exception:
                break
    return repr(value)


def _elapsed(started_at: Any, now: float, finished_at: Any = None) -> float:
    try:
        end = float(finished_at if finished_at else now)
        return round(max(0.0, end - float(started_at)), 3)
    except (TypeError, ValueError):
        return 0.0


def _kind_of(obj: Any) -> str:
    name = type(obj).__name__
    if name == 'Group':
        return 'group'
    if name == 'Agent':
        return 'agent'
    return name.lower() or 'object'


def _key_of(obj: Any) -> str:
    name = str(getattr(obj, 'name', '') or type(obj).__name__)
    code = str(getattr(obj, 'code', '') or '')
    return '{}#{}'.format(name, code) if code else '{}@{}'.format(name, id(obj))


def _timeout_repr(timeout: Any) -> Any:
    if timeout is None:
        return 'disabled'
    for attr in ('connect', 'read', 'write', 'pool'):
        if hasattr(timeout, attr):
            return {name: getattr(timeout, name) for name in ('connect', 'read', 'write', 'pool')}
    try:
        return float(timeout)
    except (TypeError, ValueError):
        return repr(timeout)


def _clip(value: Any, limit: int = 28) -> str:
    text = ' '.join(str(value or '').split())
    if len(text) <= limit:
        return text
    return text[:limit - 1] + '~'


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return '  (none)'
    columns = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[index]) for row in columns) for index in range(len(headers))]
    lines: List[str] = []
    for position, row in enumerate(columns):
        lines.append('  ' + '  '.join(row[index].ljust(widths[index]) for index in range(len(headers))).rstrip())
        if position == 0:
            lines.append('  ' + '  '.join('-' * widths[index] for index in range(len(headers))))
    return '\n'.join(lines)


class OrchifyManager:
    '''Passive, bounded, read-mostly view of everything running in this process.

    Do not construct your own: use the module singleton ``orchify_manager``. (A
    second instance is supported for tests - pass ``enabled=.../history=.../
    events=...`` to bypass the environment.)

    The manager is passive: it installs one hook on the broker and maintains its
    state purely from the events it observes, so it can never change how a run
    behaves. It is truthful: live state is merged with the broker's own run set,
    so a run started before the manager existed is still reported. It is bounded:
    history and events live in deques sized by ``history_capacity`` /
    ``event_capacity``, so a long-lived process cannot grow without limit.

    Agents and Groups register themselves in the constructor; construction is
    therefore the only path that touches the manager from library code.

    Attributes:
        enabled (bool): False makes the manager inert (no tap, no readers).
        history_capacity (int): Finished runs kept by ``history()``.
        event_capacity (int): Events kept by ``events()``; 0 keeps none.
        sink_path (str): JSONL sink path, or '' when no sink is configured.
    '''

    def __init__(self, *, enabled: Optional[bool] = None, history: Optional[int] = None,
                 events: Optional[int] = None, sink: Optional[str] = None):
        '''Create a manager, optionally overriding the environment.

        Args:
            enabled (Optional[bool]): Master switch; defaults to
                ``ORCHIFY_MANAGER_ENABLED`` (true unless set to 0/false/no/off).
            history (Optional[int]): Finished-run capacity; defaults to
                ``ORCHIFY_MANAGER_HISTORY`` (200).
            events (Optional[int]): Event capacity; defaults to
                ``ORCHIFY_MANAGER_EVENTS`` (500); 0 keeps no events in memory.
            sink (Optional[str]): JSONL append path; defaults to
                ``ORCHIFY_MANAGER_SINK``; '' disables file IO.
        '''
        self._lock = threading.RLock()
        self._created_at = time.time()
        self._enabled = _read_bool_env('ORCHIFY_MANAGER_ENABLED', True) if enabled is None else bool(enabled)
        history_capacity = (_read_int_env('ORCHIFY_MANAGER_HISTORY', DEFAULT_HISTORY)
                            if history is None else max(0, int(history)))
        event_capacity = (_read_int_env('ORCHIFY_MANAGER_EVENTS', DEFAULT_EVENTS)
                          if events is None else max(0, int(events)))

        self._history: deque = deque(maxlen=history_capacity)
        self._events: deque = deque(maxlen=event_capacity)
        self._instances: Dict[str, weakref.ref] = {}
        self._kinds: Dict[str, str] = {}
        self._group_taps: Dict[str, dict] = {}
        self._runs: Dict[str, dict] = {}
        self._finished: Dict[str, dict] = {}
        self._totals: Dict[str, Any] = {
            'runs': 0,
            'stopped': 0,
            'errors': 0,
            'steps': 0,
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'prompt_cache_hit_tokens': 0,
            'prompt_cache_miss_tokens': 0,
            'total_tokens': 0,
            'cost_usd': 0.0,
        }
        self._approval_log: Dict[str, dict] = {}
        self._sink_path: str = (
            _read_str_env('ORCHIFY_MANAGER_SINK') if sink is None else str(sink or '')
        )
        self._sink = None
        self._sink_lock = threading.Lock()
        self._sink_written = 0
        self._tap = None
        self._closed = False
        if self._enabled:
            self._install_tap()


    @property
    def enabled(self) -> bool:
        '''False when ORCHIFY_MANAGER_ENABLED=0: no tap, no registry.'''
        return self._enabled

    @property
    def history_capacity(self) -> int:
        '''Capacity of the finished-run history.

        Returns:
            int: Maximum number of finished runs kept by ``history()``.
        '''
        return self._history.maxlen or 0

    @property
    def event_capacity(self) -> int:
        '''Capacity of the in-memory event buffer.

        Returns:
            int: Maximum number of events kept by ``events()``; 0 keeps none.
        '''
        return self._events.maxlen or 0

    def _install_tap(self) -> None:
        if self._tap is not None or self._closed or not self._enabled:
            return
        self._tap = orchify_broker.register_hook('*', self._on_event, owner='')

    def _ensure_tap(self) -> None:
        '''Re-install the tap if something removed it (e.g. remove_hooks('')).'''
        if not self._enabled or self._closed:
            return
        tap = self._tap
        if tap is not None and not tap.disposed and any(binding is tap for binding in orchify_broker.hooks):
            return
        with self._lock:
            self._tap = None
        self._install_tap()

    def close(self) -> bool:
        '''Detach the event tap and every group callback. Read APIs keep working.'''
        with self._lock:
            tap, self._tap = self._tap, None
            self._closed = True
            for key in list(self._group_taps):
                self._detach_group(key)
        self._close_sink()
        if tap is None:
            return False
        remover = getattr(orchify_broker, 'remove_binding', None)
        if callable(remover):
            remover(tap)
        else:
            tap.disposed = True
        return True


    @property
    def sink_path(self) -> str:
        '''Path of the JSONL sink in use ('' when disabled).'''
        return self._sink_path

    def _write_sink(self, record: dict) -> None:
        '''Append one JSON object to the sink file, best-effort.

        Called from the event tap, so it must never raise: the first write error
        is logged and the sink is switched off for the rest of the process.

        Args:
            record (dict): The record to append as one JSON line.
        '''
        path = self._sink_path
        if not path:
            return
        try:
            with self._sink_lock:
                if self._sink is None:
                    self._sink = open(path, 'a', encoding='utf-8')
                self._sink.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                self._sink.flush()
                self._sink_written += 1
        except Exception as exc:
            logger.warning('OrchifyManager sink disabled after write error: %s', exc)
            self._close_sink()
            self._sink_path = ''

    def _close_sink(self) -> None:
        with self._sink_lock:
            sink, self._sink = self._sink, None
        if sink is not None:
            try:
                sink.close()
            except Exception:
                pass

    def replay(self, path: Optional[str] = None, **kwargs) -> List[dict]:
        '''Read back the sink file (defaults to this manager's sink). See read_sink().'''
        return read_sink(path if path is not None else self._sink_path, **kwargs)


    def register(self, obj: Any) -> str:
        '''Make an Agent/Group visible to the manager.

        Called automatically when an Agent or Group is constructed; call it by
        hand for objects created before the manager was enabled.

        Args:
            obj (Any): The Agent or Group to track.

        Returns:
            str: The registry key, or '' when the manager is disabled or the
            object cannot be weakly referenced.
        '''
        if not self._enabled or obj is None:
            return ''
        key = _key_of(obj)
        kind = _kind_of(obj)
        try:
            with self._lock:
                existing = self._instances.get(key)
                if existing is not None and existing() is obj:
                    return key
                self._instances[key] = weakref.ref(obj, self._make_reaper(key))
                self._kinds[key] = kind
        except TypeError:
            return ''
        if kind == 'group':
            self._attach_group(key, obj)
        return key

    def _make_reaper(self, key: str) -> Callable:
        def _reap(_reference) -> None:
            with self._lock:
                self._instances.pop(key, None)
                self._kinds.pop(key, None)
            self._detach_group(key)
        return _reap

    def unregister(self, obj: Any) -> bool:
        '''Forget a registered Agent/Group (the weakref would drop it anyway).'''
        key = _key_of(obj)
        with self._lock:
            removed = self._instances.pop(key, None) is not None
            self._kinds.pop(key, None)
            self._detach_group(key)
        return removed

    def _attach_group(self, key: str, group: Any) -> None:
        broker = getattr(group, 'broker', None)
        on_event = getattr(broker, 'on_event', None)
        if not callable(on_event):
            return
        with self._lock:
            state = self._group_taps.get(key)
            if state is not None:
                return
            state = {'active': True}
            self._group_taps[key] = state

        def _callback(event, state=state):
            if state.get('active'):
                self._on_group_event(event)

        try:
            on_event(_callback)
        except Exception:
            with self._lock:
                self._group_taps.pop(key, None)

    def _detach_group(self, key: str) -> None:
        with self._lock:
            state = self._group_taps.pop(key, None)
        if state is not None:
            state['active'] = False

    def _instances_snapshot(self, kind: Optional[str] = None) -> List[Tuple[str, Any]]:
        out: List[Tuple[str, Any]] = []
        with self._lock:
            entries = list(self._instances.items())
        for key, reference in entries:
            if kind is not None and self._kinds.get(key) != kind:
                continue
            obj = reference()
            if obj is not None:
                out.append((key, obj))
        return out


    def _on_event(self, event: Any) -> None:
        '''Global broker tap. Append-only: never blocks, never raises.'''
        record = None
        try:
            now = time.time()
            record = self._handle_event(event, now)
        except Exception as exc:
            logger.error('Error in OrchifyManager event tap: %s', exc)
        if record is not None and self._sink_path:
            self._write_sink({'record': 'event', **record})

    def _handle_event(self, event: Any, now: float) -> Optional[dict]:
        '''Apply one event to the manager state; returns its record (or None).'''
        event_type = str(getattr(event, 'event_type', '') or '')
        run_id = str(getattr(event, 'run_id', '') or '')
        with self._lock:
            record = self._event_record(event, now) if (self._events.maxlen or self._sink_path) else None
            if record is not None:
                self._events.append(record)
            if event_type in ('run:approval:required', 'run:approval:resolved'):
                self._apply_approval_event(event_type, run_id,
                                           str(getattr(event, 'agent_name', '') or ''),
                                           getattr(event, 'payload', None) or {}, now)
            if event_type in ('run:start', 'run:queued'):
                self._runs[run_id] = self._new_run_record(event, now)
                if event_type == 'run:queued':
                    self._runs[run_id]['status'] = QUEUED
                return record
            live = self._runs.get(run_id)
            if live is None:
                return record
            self._touch(live, event, now)
            if event_type == 'run:finish':
                self._archive(live, event, now)
            elif event_type == 'agent:abort':
                self._archive_error(live, event, now)
        return record

    def _event_record(self, event: Any, now: float) -> dict:
        '''Build the cheap, shallow record of one event.

        The tap runs on the emitting thread (the run's own loop), so building the
        record must stay shallow - ``_jsonable()`` deep-copies the payload and is
        paid once per event, for events nobody may ever read. A shallow copy keeps
        the record stable if the emitter mutates its payload afterwards.

        Args:
            event (Any): The emitted event.
            now (float): Current ``time.time()``.

        Returns:
            dict: The event record.
        '''
        payload = getattr(event, 'payload', None) or {}
        if isinstance(payload, dict):
            payload = dict(payload)
        return {
            'event_type': str(getattr(event, 'event_type', '') or ''),
            'time': now,
            'run_id': str(getattr(event, 'run_id', '') or ''),
            'turn_id': str(getattr(event, 'turn_id', '') or ''),
            'turn': int(getattr(event, 'turn', 0) or 0),
            'agent_name': str(getattr(event, 'agent_name', '') or ''),
            'source': str(getattr(event, 'source', '') or ''),
            'payload': payload,
        }

    def _new_run_record(self, event: Any, now: float) -> dict:
        run_id = str(getattr(event, 'run_id', '') or '')
        context = orchify_broker.get_run_context(run_id) or {}
        record = self._context_record(run_id, context, now)
        if not record.get('name'):
            record['name'] = str(getattr(event, 'agent_name', '') or '')
        if not record.get('agent_code'):
            record['agent_code'] = str(getattr(event, 'agent_code', '') or '')
        record['turn_id'] = str(getattr(event, 'turn_id', '') or record.get('turn_id', ''))
        record['step'] = int(getattr(event, 'turn', 0) or 0)
        return record

    def _context_record(self, run_id: str, context: dict, now: float) -> dict:
        context = context or {}
        kind = context.get('kind') or ('group' if (context.get('group_name') or context.get('group_code')) else 'agent')
        return {
            'run_id': run_id,
            'kind': kind,
            'name': str(context.get('group_name') or context.get('agent_name') or ''),
            'agent_name': str(context.get('agent_name', '') or ''),
            'agent_code': str(context.get('agent_code', '') or context.get('group_code', '') or ''),
            'mode': str(context.get('mode', '') or ''),
            'status': RUNNING,
            'started_at': context.get('started_at') or now,
            'last_event_at': now,
            'finished_at': None,
            'step': 0,
            'turn_id': str(context.get('turn_id', '') or ''),
            'steps_done': 0,
            'total_steps': 0,
            'paused': False,
            'error': '',
            'answer': '',
            'usage': {},
            'stopped': False,
            'overridden': False,
            'winner': None,
            'votes': [],
            'agent_run_ids': [],
            'active': True,
            'source': kind,
        }

    def _touch(self, live: dict, event: Any, now: float) -> None:
        event_type = str(getattr(event, 'event_type', '') or '')
        payload = getattr(event, 'payload', None) or {}
        live['last_event_at'] = now
        turn = int(getattr(event, 'turn', 0) or 0)
        if turn and turn > int(live.get('step', 0) or 0):
            live['step'] = turn
        turn_id = str(getattr(event, 'turn_id', '') or '')
        if turn_id:
            live['turn_id'] = turn_id
        if event_type == 'run:queued':
            live['status'] = QUEUED
        elif event_type == 'run:dequeued':
            if live.get('status') == QUEUED:
                live['status'] = RUNNING
        elif event_type == 'run:paused':
            live['paused'] = True
            live['status'] = PAUSED
        elif event_type == 'run:resumed':
            live['paused'] = False
            if live.get('status') == PAUSED:
                live['status'] = RUNNING
        elif event_type == 'run:approval:required':
            live['status'] = AWAITING_APPROVAL
        elif event_type == 'run:approval:resolved':
            if live.get('status') == AWAITING_APPROVAL:
                live['status'] = RUNNING
        elif event_type == 'agent:answer':
            content = str(getattr(event, 'content', '') or '')
            if content:
                live['answer'] = (str(live.get('answer') or '') + content)[-MAX_ANSWER:]

    def _archive(self, live: dict, event: Any, now: float) -> None:
        payload = getattr(event, 'payload', None) or {}
        stopped = bool(payload.get('stopped'))
        live['status'] = STOPPED if stopped else DONE
        live['stopped'] = stopped
        live['overridden'] = bool(payload.get('overridden'))
        total_steps = int(payload.get('total_steps') or live.get('step') or 0)
        live['total_steps'] = total_steps
        live['steps_done'] = total_steps
        usage = payload.get('usage')
        if usage:
            live['usage'] = _jsonable(usage)
        budget = payload.get('budget')
        if isinstance(budget, dict):
            live['budget'] = _jsonable(budget)
        answer = payload.get('answer')
        if answer:
            live['answer'] = _truncate(answer, MAX_ANSWER)
        self._finish_run(live, now)

    def _archive_error(self, live: dict, event: Any, now: float) -> None:
        payload = getattr(event, 'payload', None) or {}
        live['status'] = ERROR
        live['error'] = _truncate(
            payload.get('error') or payload.get('message') or payload.get('reason') or '',
            MAX_STRING,
        )
        self._finish_run(live, now)

    def _finish_run(self, live: dict, now: float) -> None:
        run_id = str(live.get('run_id') or '')
        previous = self._finished.get(run_id)
        if previous is not None and previous.get('status') == ERROR and live.get('status') != ERROR:
            self._runs.pop(run_id, None)
            return
        live['finished_at'] = now
        live['elapsed'] = _elapsed(live.get('started_at'), now)
        live['active'] = False
        live['pending_approvals'] = 0
        record = dict(live)
        self._runs.pop(run_id, None)
        if previous is not None and self._history.maxlen != 0:
            self._history = deque(
                (item for item in self._history if item.get('run_id') != run_id),
                maxlen=self._history.maxlen,
            )
        self._finished[run_id] = record
        self._history.append(record)
        self._add_totals(record)
        if self._sink_path:
            self._write_sink({'record': 'run', **record})
        if self._history.maxlen is not None:
            keep = {item.get('run_id') for item in self._history}
            for stale in [key for key in self._finished if key not in keep]:
                self._finished.pop(stale, None)

    def _add_totals(self, record: dict) -> None:
        totals = self._totals
        totals['runs'] += 1
        totals['steps'] += int(record.get('total_steps') or 0)
        if record.get('status') == STOPPED:
            totals['stopped'] += 1
        elif record.get('status') == ERROR:
            totals['errors'] += 1
        usage = record.get('usage') or {}
        for key in ('completion_tokens', 'prompt_tokens', 'prompt_cache_hit_tokens',
                    'prompt_cache_miss_tokens', 'total_tokens'):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += value
        budget = record.get('budget') or {}
        cost = budget.get('cost_usd') if isinstance(budget, dict) else None
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            totals['cost_usd'] += cost


    def _on_group_event(self, event: Any) -> None:
        try:
            now = time.time()
            event_type = str(getattr(event, 'event_type', '') or '')
            payload = getattr(event, 'payload', None) or {}
            run_id = str(getattr(event, 'run_id', '') or payload.get('run_id') or '')
            with self._lock:
                self._events.append({
                    'event_type': 'group:{}'.format(event_type),
                    'time': now,
                    'run_id': run_id,
                    'turn_id': str(getattr(event, 'turn_id', '') or ''),
                    'turn': int(getattr(event, 'turn', 0) or 0),
                    'agent_name': str(getattr(event, 'agent_code', '') or ''),
                    'source': 'group',
                    'payload': _jsonable(payload),
                })
                if not run_id:
                    return
                live = self._runs.get(run_id)
                if live is None:
                    live = self._context_record(run_id, orchify_broker.get_run_context(run_id) or {}, now)
                    self._runs[run_id] = live
                live['last_event_at'] = now
                if event_type == 'start':
                    live['status'] = RUNNING
                elif event_type == 'vote':
                    if payload.get('winner'):
                        live['winner'] = str(payload.get('winner'))
                    if payload.get('answer'):
                        live['answer'] = _truncate(payload.get('answer'), MAX_ANSWER)
                elif event_type == 'end':
                    live['status'] = DONE
                    live['stopped'] = bool(payload.get('stopped'))
                    live['winner'] = payload.get('winner')
                    live['votes'] = _jsonable(payload.get('votes') or [])
                    live['agent_run_ids'] = [str(item) for item in (payload.get('agent_run_ids') or [])]
                    live['total_steps'] = len(live['agent_run_ids'])
                    if payload.get('answer'):
                        live['answer'] = _truncate(payload.get('answer'), MAX_ANSWER)
                    self._finish_run(live, now)
        except Exception as exc:
            logger.error('Error in OrchifyManager group tap: %s', exc)


    def _reconcile(self, now: Optional[float] = None) -> float:
        '''Sync event-derived state with the broker's live run set.

        Adopts runs the tap never saw (started before the manager was enabled, or
        a Group's own run) and archives runs whose context disappeared without a
        terminal event.

        Args:
            now (Optional[float]): ``time.time()`` to use; defaults to now.

        Returns:
            float: The timestamp used.
        '''
        now = now or time.time()
        self._ensure_tap()
        contexts = dict(orchify_broker.run_contexts)
        with self._lock:
            for run_id, context in contexts.items():
                if run_id not in self._runs:
                    self._runs[run_id] = self._context_record(run_id, context, now)
            for run_id in [key for key in self._runs if key not in contexts]:
                live = self._runs[run_id]
                if live.get('finished_at'):
                    continue
                if live.get('status') not in FINISHED_STATUSES:
                    live['status'] = DONE
                self._finish_run(live, now)
        return now

    def _pending_feedback(self) -> Dict[str, int]:
        queues = getattr(orchify_broker, '_feedback', None) or {}
        depths: Dict[str, int] = {}
        for run_id, queue in list(queues.items()):
            try:
                depth = len(queue)
            except TypeError:
                depth = 0
            if depth:
                depths[str(run_id)] = depth
        return depths

    def _live_record(self, run_id: str, now: float) -> Optional[dict]:
        with self._lock:
            live = self._runs.get(run_id)
            finished = self._finished.get(run_id)
            record = dict(live) if live is not None else (dict(finished) if finished is not None else None)
        if record is None:
            context = orchify_broker.get_run_context(run_id)
            if context is None:
                return None
            record = self._context_record(run_id, context, now)
        approvals = orchify_broker.pending_approvals(run_id)
        record['pending_approvals'] = len(approvals)
        record['approval_ids'] = [item.approval_id for item in approvals]
        record['feedback_depth'] = self._pending_feedback().get(run_id, 0)
        record['active'] = run_id in orchify_broker.run_contexts
        if approvals and record.get('status') == RUNNING:
            record['status'] = AWAITING_APPROVAL
        if record.get('status') == PAUSED:
            record['paused'] = True
        record['elapsed'] = _elapsed(record.get('started_at'), now, record.get('finished_at'))
        return record

    def runs(self, *, kind: Optional[str] = None, status: Optional[str] = None) -> List[dict]:
        '''Every run currently registered on the broker (finished ones move to history()).'''
        now = self._reconcile()
        out: List[dict] = []
        for run_id in orchify_broker.active_runs():
            record = self._live_record(run_id, now)
            if record is None:
                continue
            if kind is not None and record.get('kind') != kind:
                continue
            if status is not None and record.get('status') != status:
                continue
            out.append(record)
        out.sort(key=lambda item: (item.get('started_at') or 0, item.get('run_id') or ''))
        return out

    def run(self, run_id: str) -> Optional[dict]:
        '''One run by id: live if it is still registered, else its history record.'''
        if not run_id:
            return None
        now = self._reconcile()
        if run_id in orchify_broker.run_contexts:
            return self._live_record(run_id, now)
        with self._lock:
            finished = self._finished.get(run_id)
        if finished is not None:
            record = dict(finished)
            record['pending_approvals'] = 0
            record['feedback_depth'] = 0
            return record
        return self._live_record(run_id, now)

    def history(self, *, limit: Optional[int] = 50, kind: Optional[str] = None,
                status: Optional[str] = None) -> List[dict]:
        '''Finished runs, newest first.'''
        self._reconcile()
        with self._lock:
            items = [dict(item) for item in reversed(self._history)]
        if kind is not None:
            items = [item for item in items if item.get('kind') == kind]
        if status is not None:
            items = [item for item in items if item.get('status') == status]
        if limit is not None and limit >= 0:
            items = items[:limit]
        return items

    def agents(self) -> List[dict]:
        '''Every live registered Agent, with its last run and answer.'''
        now = time.time()
        self._reconcile(now)
        out = [self._agent_record(key, obj, now) for key, obj in self._instances_snapshot('agent')]
        return sorted(out, key=lambda item: item['key'])

    def _agent_record(self, key: str, agent: Any, now: float) -> dict:
        run_id = str(getattr(agent, 'last_run_id', '') or '')
        is_running_fn = getattr(agent, 'is_running', None)
        try:
            is_running = bool(is_running_fn()) if callable(is_running_fn) else bool(run_id and run_id in orchify_broker.run_contexts)
        except Exception:
            is_running = False
        with self._lock:
            last = self._runs.get(run_id) or self._finished.get(run_id) or {}
        return {
            'key': key,
            'name': str(getattr(agent, 'name', '') or ''),
            'code': str(getattr(agent, 'code', '') or ''),
            'description': _truncate(getattr(agent, 'description', '') or ''),
            'model': str(getattr(agent, 'model', '') or ''),
            'messages_mode': str(getattr(agent, 'messages_mode', '') or ''),
            'is_running': is_running,
            'status': RUNNING if is_running else (last.get('status') or 'idle'),
            'last_run_id': run_id,
            'last_answer': _truncate(getattr(agent, 'last_answer', '') or '', MAX_STRING),
            'tools': sorted(str(name) for name in (getattr(agent, 'tools', None) or {})),
            'plugins': sorted(str(name) for name in (getattr(agent, 'enabled_plugins', None) or ())),
            'pending_approvals': len(orchify_broker.pending_approvals(run_id)) if run_id else 0,
        }

    def groups(self) -> List[dict]:
        '''Every live registered Group, with its members and last voting result.'''
        now = time.time()
        self._reconcile(now)
        out = [self._group_record(key, group, now) for key, group in self._instances_snapshot('group')]
        return sorted(out, key=lambda item: item['key'])

    def _group_record(self, key: str, group: Any, now: float) -> dict:
        run_id = str(getattr(group, 'last_run_id', '') or '')
        is_running_fn = getattr(group, 'is_running', None)
        try:
            is_running = bool(is_running_fn()) if callable(is_running_fn) else bool(run_id and run_id in orchify_broker.run_contexts)
        except Exception:
            is_running = False
        with self._lock:
            last = self._runs.get(run_id) or self._finished.get(run_id) or {}
        members = []
        for member in (getattr(group, 'agent_sequence', None) or []):
            members.append({'name': str(getattr(member, 'name', '') or ''), 'kind': _kind_of(member)})
        return {
            'key': key,
            'name': str(getattr(group, 'name', '') or ''),
            'code': str(getattr(group, 'code', '') or ''),
            'description': _truncate(getattr(group, 'description', '') or ''),
            'mode': str(getattr(group, 'mode', '') or ''),
            'is_running': is_running,
            'status': RUNNING if is_running else (last.get('status') or 'idle'),
            'members': members,
            'member_count': len(members),
            'last_run_id': run_id,
            'last_run_ids': [str(item) for item in (getattr(group, 'last_run_ids', None) or [])],
            'last_winner': getattr(group, 'last_winner', None),
            'last_answer': _truncate(getattr(group, 'last_answer', '') or '', MAX_STRING),
            'vote_timeout': getattr(group, 'vote_timeout', None),
        }

    def _apply_approval_event(self, event_type: str, run_id: str, agent_name: str,
                              payload: dict, now: float) -> None:
        '''Retain approval decisions: finish_task() drops them from the pool.'''
        approval_id = str(payload.get('approval_id', '') or '')
        if not approval_id:
            return
        record = self._approval_log.get(approval_id)
        if record is None:
            record = {
                'approval_id': approval_id,
                'run_id': run_id,
                'agent_name': agent_name,
                'kind': str(payload.get('kind', '') or ''),
                'tool_id': str(payload.get('tool_id', '') or ''),
                'tool_name': str(payload.get('tool_name', '') or ''),
                'args': _jsonable(payload.get('args') or {}),
                'step': int(payload.get('step', 0) or 0),
                'timeout': payload.get('timeout'),
                'created_at': now,
                'age': 0.0,
                'resolved': False,
                'approved': None,
                'decision': None,
                'reason': '',
                'timed_out': False,
                'answer': '',
            }
            self._approval_log[approval_id] = record
        if record.get('run_id') == '':
            record['run_id'] = run_id
        if event_type == 'run:approval:resolved':
            approved = payload.get('approved')
            record['resolved'] = True
            record['approved'] = bool(approved) if approved is not None else None
            record['decision'] = record['approved']
            record['reason'] = str(payload.get('reason', '') or '')
            record['answer'] = str(payload.get('answer', '') or '')
            record['timed_out'] = bool(payload.get('timed_out', False))
        self._prune_approval_log()

    def _prune_approval_log(self) -> None:
        capacity = self.history_capacity
        while len(self._approval_log) > capacity:
            self._approval_log.pop(next(iter(self._approval_log)))

    def approvals(self, *, pending_only: bool = False, run_id: Optional[str] = None) -> List[dict]:
        '''The manual-confirmation pool (newest last), including decided ones.'''
        now = time.time()
        items = (orchify_broker.pending_approvals(run_id) if pending_only
                 else orchify_broker.approvals(run_id))
        records = [self._approval_record(item, now) for item in items]
        if pending_only:
            return records
        with self._lock:
            retained = [dict(record) for record in self._approval_log.values()]
        seen = {record['approval_id'] for record in records}
        for record in retained:
            if record['approval_id'] in seen:
                continue
            if run_id is not None and record.get('run_id') != run_id:
                continue
            record['age'] = _elapsed(record.get('created_at', now), now)
            records.append(record)
        return records

    def _approval_record(self, approval: Any, now: float) -> dict:
        created = getattr(approval, 'created_at', 0.0) or now
        return {
            'approval_id': str(getattr(approval, 'approval_id', '') or ''),
            'run_id': str(getattr(approval, 'run_id', '') or ''),
            'agent_name': str(getattr(approval, 'agent_name', '') or ''),
            'kind': str(getattr(approval, 'kind', '') or ''),
            'tool_id': str(getattr(approval, 'tool_id', '') or ''),
            'tool_name': str(getattr(approval, 'tool_name', '') or ''),
            'args': _jsonable(getattr(approval, 'args', None) or {}),
            'step': int(getattr(approval, 'step', 0) or 0),
            'timeout': getattr(approval, 'timeout', None),
            'created_at': created,
            'age': _elapsed(created, now),
            'resolved': bool(getattr(approval, 'resolved', False)),
            'approved': getattr(approval, 'approved', None),
            'decision': getattr(approval, 'decision', None),
            'reason': str(getattr(approval, 'reason', '') or ''),
            'answer': str(getattr(approval, 'answer', '') or ''),
            'timed_out': bool(getattr(approval, 'timed_out', False)),
        }

    def questions(self, *, pending_only: bool = False,
                  run_id: Optional[str] = None) -> List[dict]:
        '''Open-question entries of the manual pool (kind='question').

        Same records as ``approvals()``; ``pending_only=True`` keeps just the ones
        still waiting for an answer.

        Args:
            pending_only (bool): Return only unresolved questions.
            run_id (Optional[str]): Restrict to one execution.

        Returns:
            List[dict]: Question records, oldest first.
        '''
        records = [record for record in self.approvals(pending_only=pending_only, run_id=run_id)
                   if record.get('kind') == 'question']
        return records

    def hooks(self) -> dict:
        '''Registered hook bindings, grouped by owner and event type.'''
        bindings = list(orchify_broker.hooks)
        by_owner: Dict[str, int] = {}
        by_event_type: Dict[str, int] = {}
        items = []
        for binding in bindings:
            owner = binding.owner or '(global)'
            by_owner[owner] = by_owner.get(owner, 0) + 1
            for event_type in binding.event_type:
                by_event_type[event_type] = by_event_type.get(event_type, 0) + 1
            items.append({
                'owner': binding.owner,
                'event_type': list(binding.event_type),
                'run_id': binding.run_id,
                'mode': binding.mode,
                'count': binding.count,
                'disposed': bool(binding.disposed),
                'agent_name': binding.agent_name,
                'func': getattr(binding.func, '__qualname__', None) or repr(binding.func),
            })
        return {'total': len(items), 'by_owner': by_owner, 'by_event_type': by_event_type, 'bindings': items}

    def feedback(self) -> Dict[str, int]:
        '''Undrained feedback items queued per run (control messages on the way).'''
        return self._pending_feedback()

    def plugins(self) -> dict:
        '''Registered/loaded plugins (names only; no plugin objects).'''
        try:
            from .plugin_manager import orchify_plugins
        except Exception as exc:
            return {'error': str(exc), 'registered': [], 'loaded': []}
        registered = []
        for name, plugin in (orchify_plugins.registered() or {}).items():
            registered.append({
                'name': str(name),
                'loaded': bool(getattr(plugin, 'loaded', False)),
                'description': _truncate(getattr(plugin, 'description', '') or ''),
            })
        try:
            loaded = [str(name) for name in orchify_plugins.loaded]
        except Exception:
            loaded = [item['name'] for item in registered if item['loaded']]
        return {'registered': registered, 'loaded': loaded}

    def backend(self) -> dict:
        '''Health of the async backend loop and the shared tool executor.'''
        loop = getattr(orchify_web_backend, '_loop', None)
        thread = getattr(orchify_web_backend, '_thread', None)
        info: Dict[str, Any] = {
            'loop_running': bool(loop is not None and loop.is_running()),
            'thread_alive': bool(thread is not None and thread.is_alive()),
            'thread_name': str(getattr(thread, 'name', '') or ''),
            'http_timeout': _timeout_repr(getattr(orchify_web_backend, 'timeout', None)),
            'tool_executor': 'not started',
            'tool_workers': None,
            'tool_threads': None,
            'gates': gate_stats(),
        }
        try:
            import sys
            module = sys.modules.get('orchify.agent')
            if module is None:
                from . import agent as module
            executor = getattr(module, '_tool_executor', None)
            if executor is not None:
                info['tool_executor'] = 'running'
                info['tool_workers'] = getattr(executor, '_max_workers', None)
                info['tool_threads'] = len(getattr(executor, '_threads', ()) or ())
        except Exception as exc:
            info['tool_executor'] = 'unknown: {}'.format(exc)
        return info

    def events(self, *, limit: Optional[int] = 50, run_id: Optional[str] = None,
               types: Optional[Union[str, Sequence[str]]] = None) -> List[dict]:
        '''Recent events, oldest first: the last ``limit`` matching events.

        Args:
            limit (Optional[int]): Maximum number of events; None means all, 0
                returns an empty list.
            run_id (Optional[str]): Restrict to one execution.
            types (Optional[Union[str, Sequence[str]]]): One glob or a list of
                globs matched with ``fnmatch`` (the same syntax hooks use).

        Returns:
            List[dict]: Event records, oldest first.
        '''
        self._ensure_tap()
        with self._lock:
            items = list(self._events)
        if run_id:
            items = [item for item in items if item.get('run_id') == run_id]
        if types:
            patterns = [types] if isinstance(types, str) else list(types)
            items = [item for item in items
                     if any(fnmatchcase(item.get('event_type', ''), pattern) for pattern in patterns)]
        if limit is not None and limit >= 0:
            items = items[-limit:] if limit else []
        return [dict(item, payload=_jsonable(item.get('payload') or {})) for item in items]

    def totals(self) -> dict:
        '''Cumulative counters for the runs this manager has seen finish.'''
        with self._lock:
            totals = dict(self._totals)
            totals['history_size'] = len(self._history)
            totals['uptime'] = round(time.time() - self._created_at, 3)
            return totals


    def status(self) -> dict:
        '''Compact overview: capacity, counts, live runs, backend health.'''
        now = self._reconcile()
        runs = self.runs()
        by_status: Dict[str, int] = {status: 0 for status in RUN_STATUSES}
        for record in runs:
            by_status[record.get('status', RUNNING)] = by_status.get(record.get('status', RUNNING), 0) + 1
        agents = self.agents()
        groups = self.groups()
        pending = orchify_broker.pending_approvals()
        return {
            'enabled': self._enabled,
            'uptime': round(now - self._created_at, 3),
            'history_capacity': self.history_capacity,
            'history_size': len(self._history),
            'event_capacity': self.event_capacity,
            'event_size': len(self._events),
            'counts': {
                'runs': len(runs),
                'runs_by_status': by_status,
                'history': len(self._history),
                'agents': len(agents),
                'agents_running': sum(1 for item in agents if item.get('is_running')),
                'groups': len(groups),
                'groups_running': sum(1 for item in groups if item.get('is_running')),
                'pending_approvals': len(pending),
                'pending_questions': sum(1 for item in pending if item.kind == 'question'),
                'approvals': len(orchify_broker.approvals()),
                'hooks': len(orchify_broker.hooks),
                'events': len(self._events),
            },
            'runs': [
                {'run_id': item['run_id'], 'kind': item['kind'], 'name': item['name'],
                 'status': item['status'], 'step': item['step'],
                 'pending_approvals': item['pending_approvals']}
                for item in runs
            ],
            'totals': self.totals(),
            'backend': self.backend(),
            'gates': gate_stats(),
            'sink': self._sink_path or None,
            'sink_records': self._sink_written,
        }

    def snapshot(self, *, history_limit: int = 50, event_limit: int = 50) -> dict:
        '''Everything the manager knows, as one JSON-serialisable dict.'''
        return {
            'generated_at': time.time(),
            'status': self.status(),
            'runs': self.runs(),
            'history': self.history(limit=history_limit),
            'agents': self.agents(),
            'groups': self.groups(),
            'approvals': self.approvals(),
            'hooks': self.hooks(),
            'feedback': self.feedback(),
            'plugins': self.plugins(),
            'backend': self.backend(),
            'events': self.events(limit=event_limit),
            'totals': self.totals(),
        }

    def to_json(self, *, indent: int = 2, history_limit: int = 50, event_limit: int = 50) -> str:
        '''snapshot() as JSON text.'''
        return json.dumps(
            self.snapshot(history_limit=history_limit, event_limit=event_limit),
            indent=indent, ensure_ascii=False, default=str,
        )

    def report(self, *, history_limit: int = 10, event_limit: int = 10) -> str:
        '''A human-readable text report of the same data.'''
        status = self.status()
        counts = status['counts']
        lines: List[str] = []
        lines.append('Orchify manager - enabled={} uptime={}s'.format(
            'yes' if self._enabled else 'no', status['uptime']))
        lines.append('history {}/{}  events {}/{}'.format(
            counts['history'], self.history_capacity, counts['events'], self.event_capacity))
        lines.append('runs={} agents={} groups={} approvals={} (pending {}) hooks={}'.format(
            counts['runs'], counts['agents'], counts['groups'], counts['approvals'],
            counts['pending_approvals'], counts['hooks']))

        lines.append('')
        lines.append('RUNS')
        lines.append(_table(
            ['run_id', 'kind', 'name', 'status', 'step', 'appr', 'fb', 'elapsed'],
            [[_clip(item['run_id'], 30), item['kind'], _clip(item['name'], 16), item['status'],
              item['step'], item['pending_approvals'], item['feedback_depth'], item['elapsed']]
             for item in self.runs()],
        ))

        lines.append('')
        lines.append('AGENTS')
        lines.append(_table(
            ['key', 'model', 'running', 'last_run', 'status', 'tools', 'answer'],
            [[_clip(item['key'], 24), _clip(item['model'], 18), item['is_running'],
              _clip(item['last_run_id'], 24), item['status'], len(item['tools']),
              _clip(item['last_answer'], 24)]
             for item in self.agents()],
        ))

        lines.append('')
        lines.append('GROUPS')
        lines.append(_table(
            ['key', 'mode', 'members', 'running', 'status', 'winner', 'answer'],
            [[_clip(item['key'], 24), item['mode'], item['member_count'], item['is_running'],
              item['status'], _clip(item['last_winner'], 14), _clip(item['last_answer'], 24)]
             for item in self.groups()],
        ))

        lines.append('')
        lines.append('APPROVALS')
        lines.append(_table(
            ['approval_id', 'run_id', 'kind', 'tool', 'step', 'age', 'resolved', 'decision'],
            [[item['approval_id'], _clip(item['run_id'], 24), item['kind'],
              _clip(item['tool_name'], 16), item['step'], item['age'], item['resolved'],
              item['decision']]
             for item in self.approvals()],
        ))

        lines.append('')
        lines.append('HISTORY (last {})'.format(history_limit))
        lines.append(_table(
            ['run_id', 'kind', 'status', 'steps', 'elapsed', 'answer'],
            [[_clip(item['run_id'], 30), item['kind'], item['status'], item['total_steps'],
              item['elapsed'], _clip(item['answer'], 32)]
             for item in self.history(limit=history_limit)],
        ))

        lines.append('')
        lines.append('TOTALS')
        totals = self.totals()
        lines.append('  runs={runs} stopped={stopped} errors={errors} steps={steps} tokens={total_tokens}'.format(**totals))
        lines.append('  prompt={prompt_tokens} completion={completion_tokens} cache_hit={prompt_cache_hit_tokens} cache_miss={prompt_cache_miss_tokens}'.format(**totals))

        lines.append('')
        lines.append('BACKEND')
        for key, value in self.backend().items():
            lines.append('  {}={}'.format(key, value))

        lines.append('')
        lines.append('EVENTS (last {})'.format(event_limit))
        for item in self.events(limit=event_limit):
            lines.append('  [{:>8.3f}] {:<28} {}'.format(
                item['time'] % 100000, item['event_type'], _clip(item['run_id'], 30)))

        return '\n'.join(lines)


    def pause(self, run_id: str, *, payload: Optional[dict] = None) -> bool:
        '''Ask a run to park at its next checkpoint (False if it is not active).'''
        return orchify_broker.feedback('control:pause', run_id=run_id, payload=payload)

    def resume(self, run_id: str) -> bool:
        '''Release a run parked by pause().'''
        return orchify_broker.feedback('control:resume', run_id=run_id)

    def stop(self, run_id: str) -> bool:
        '''Ask a run to stop at its next checkpoint.'''
        return orchify_broker.feedback('control:stop', run_id=run_id)

    def stop_all(self) -> int:
        '''Stop every live run. Returns how many accepted the request.'''
        return sum(1 for run_id in orchify_broker.active_runs() if self.stop(run_id))

    def pause_all(self) -> int:
        '''Pause every live run. Returns how many accepted the request.'''
        return sum(1 for run_id in orchify_broker.active_runs() if self.pause(run_id))

    def approve(self, approval_id: str, *, reason: str = '') -> bool:
        '''Approve a pending approval (False when the id is unknown/already decided).'''
        return orchify_broker.approve(approval_id, reason=reason)

    def reject(self, approval_id: str, *, reason: str = 'denied') -> bool:
        '''Reject a pending approval (the parked tool call is denied, not the run).'''
        return orchify_broker.reject(approval_id, reason=reason)

    def answer(self, approval_id: str, text: str) -> bool:
        '''Deliver a free-text answer to a pending question (kind='question').

        Args:
            approval_id (str): The question entry to answer.
            text (str): The answer to hand back to the parked tool call.

        Returns:
            bool: True when the answer was accepted; False when the id is unknown
            or was already resolved. The parked run resumes with ``text`` as the
            tool result.
        '''
        return orchify_broker.answer_approval(approval_id, text)

    def wait(self, run_id: str, timeout: Optional[float] = None) -> bool:
        '''Block until a run finishes.

        Args:
            run_id (str): The run id.
            timeout (Optional[float]): Seconds to wait; None waits until the run
                finishes.

        Returns:
            bool: True when the run finished, False on timeout or for an unknown/
            non-active run.

        Raises:
            RuntimeError: When called from the orchestrator loop thread, where
                waiting would deadlock the run it is waiting for.
        '''
        loop_thread = getattr(orchify_web_backend, '_thread', None)
        if loop_thread is not None and threading.current_thread() is loop_thread:
            raise RuntimeError(
                'OrchifyManager.wait() would deadlock: it is being called from the '
                'orchestrator loop thread. Wait from your own thread, or hook '
                "'run:finish'/use run_done_event() instead."
            )
        return orchify_broker.wait_run(run_id, timeout=timeout)


def read_sink(path: str, *, limit: Optional[int] = None,
              run_id: Optional[str] = None, kinds: Optional[Sequence[str]] = None,
              types: Optional[Sequence[str]] = None) -> List[dict]:
    '''Read back a JSONL sink file written by ORCHIFY_MANAGER_SINK.

    Args:
        path (str): Path of the sink file.
        limit (Optional[int]): Keep only the LAST n records; 0 returns nothing
            and None returns everything.
        run_id (Optional[str]): Restrict to one execution.
        kinds (Optional[Sequence[str]]): Record kinds to keep; 'event' for
            tapped events, 'run' for finished-run summaries.
        types (Optional[Sequence[str]]): Event-type globs matched with
            ``fnmatch`` (for example 'agent:*').

    Returns:
        List[dict]: Records oldest-first; malformed lines are skipped and an
        empty or unreadable path yields an empty list.
    '''
    if not path:
        return []
    records: List[dict] = []
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if not isinstance(record, dict):
                    continue
                if kinds is not None and record.get('record') not in tuple(kinds):
                    continue
                if run_id is not None and record.get('run_id') != run_id:
                    continue
                if types is not None:
                    event_type = str(record.get('event_type', '') or '')
                    if not any(fnmatchcase(event_type, pattern) for pattern in types):
                        continue
                records.append(record)
    except OSError:
        return []
    if limit is None:
        return records
    limit = max(0, int(limit))
    if limit == 0:
        return []
    return records[-limit:]


orchify_manager = OrchifyManager()
