'''Durable run storage: event journal, checkpoints and agent snapshots.

Nothing here needs a database. Each run writes append-only JSONL files, which
survive a crash, are trivially tail-able and can be shipped to object storage by
the operator. Wire them up like this::

    from orchify import Agent, orchify_broker
    from orchify.persistence import CheckpointStore, EventStore

    events = EventStore('var/orchify')
    events.attach(orchify_broker)             # journal every event

    checkpoints = CheckpointStore('var/orchify')
    agent = Agent(name='assistant', llm=llm, checkpoint=checkpoints.callback)

    # ... later, after a crash:
    checkpoints.resume(agent, run_id)         # rewind to the last finished step
    agent.ask('carry on where you stopped')
'''

import json
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from orchify.utils import get_logger

logger = get_logger(__name__)

_SAFE = re.compile(r'[^A-Za-z0-9_-]+')


def _safe_name(value: str) -> str:
    '''Make a run id usable as a file name.

    Args:
        value (str): Raw identifier (a run id contains a ``#``).

    Returns:
        str: ``[A-Za-z0-9_-]`` only, never empty.
    '''
    name = _SAFE.sub('-', str(value or '')).strip('-')
    return name[:120] or 'run'


def _ensure_dir(path: str) -> str:
    '''Create ``path`` when missing and return it.

    Args:
        path (str): Directory to ensure.

    Returns:
        str: The same path.
    '''
    os.makedirs(path, exist_ok=True)
    return path


def _append_line(path: str, record: Dict[str, Any]) -> None:
    '''Append one JSON record to a file, creating it when needed.

    Args:
        path (str): File to append to.
        record (Dict[str, Any]): JSON-serialisable record.
    '''
    line = json.dumps(record, ensure_ascii=False, default=str)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def _read_lines(path: str) -> Iterator[Dict[str, Any]]:
    '''Yield the JSON records of a JSONL file, skipping damaged lines.

    Args:
        path (str): File to read.

    Yields:
        Dict[str, Any]: One decoded record per line.
    '''
    if not os.path.isfile(path):
        return
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                logger.warning('persistence: skipping a damaged line in %s', path)
                continue
            if isinstance(record, dict):
                yield record


def _run_ids(directory: str) -> List[str]:
    '''List the real run ids of the ``*.jsonl`` files in a directory.

    File names are sanitised versions of run ids (a run id carries ``#``), so the
    original id is read back from the first record; the file stem is the fallback
    for a file whose first line is damaged.

    Args:
        directory (str): Directory holding ``*.jsonl`` files.

    Returns:
        List[str]: Run ids, most recently modified file first.
    '''
    try:
        names = [name for name in os.listdir(directory) if name.endswith('.jsonl')]
    except OSError:
        return []
    found: List[tuple] = []
    for name in names:
        path = os.path.join(directory, name)
        run_id = name[:-6]
        for record in _read_lines(path):
            candidate = str(record.get('run_id') or '')
            if candidate:
                run_id = candidate
                break
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        found.append((mtime, run_id))
    found.sort(reverse=True)
    return [run_id for _, run_id in found]


class EventStore:
    '''Append-only journal of broker events, one JSONL file per run.

    Attributes:
        directory (str): Root directory; runs live under ``<directory>/runs``.
        run_id (str): Prefix of the files this instance owns.
    '''

    def __init__(self, directory: str = 'orchify-runs', *, flush_every: float = 1.0,
                 max_events: int = 20000):
        '''Configure the store.

        Args:
            directory (str): Root directory to write into (created when missing).
            flush_every (float): Seconds between flushes when a hook is attached;
                ``0`` flushes every record.
            max_events (int): Cap on events journalled per run (the oldest events
                stay on disk; this only bounds a single process' file).
        '''
        self.directory = os.path.abspath(directory)
        self.runs_dir = _ensure_dir(os.path.join(self.directory, 'runs'))
        self.flush_every = max(0.0, float(flush_every))
        self.max_events = max(1, int(max_events))
        self.run_id = ''
        self._binding: Any = None
        self._broker: Any = None
        self._buffer: List[Dict[str, Any]] = []
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self.written = 0

    def path_for(self, run_id: str) -> str:
        '''Return the journal path for one run.

        Args:
            run_id (str): Run identifier.

        Returns:
            str: Absolute path of the JSONL file.
        '''
        return os.path.join(self.runs_dir, f'{_safe_name(run_id)}.jsonl')

    def attach(self, broker=None, *, event_types: str = '*') -> Any:
        '''Journal every event the broker emits.

        Args:
            broker (Any): Broker to observe; defaults to ``orchify_broker``.
            event_types (str): Event filter (a glob, as for ``Broker.on``).

        Returns:
            Any: The hook binding, so it can be detached later.
        '''
        if broker is None:
            from orchify.broker import orchify_broker as broker
        self.detach()
        self._broker = broker
        self._binding = broker.register_hook(event_types, self.record, mode='background')
        return self._binding

    def detach(self) -> None:
        '''Stop journaling and flush what is buffered.

        Events already queued on the background dispatcher are drained first:
        removing the binding marks it disposed, and the dispatcher skips disposed
        bindings, so detaching without draining would silently drop the tail of
        a run (including ``run:finish``).
        '''
        if self._binding is not None:
            broker = self._broker
            try:
                if broker is not None:
                    broker.drain_hooks(timeout=5.0)
            except Exception:
                pass
            try:
                if broker is not None:
                    broker.remove_binding(self._binding)
                else:
                    self._binding.broker.remove_binding(self._binding)
            except Exception:
                try:
                    from orchify.broker import orchify_broker
                    orchify_broker.remove_binding(self._binding)
                except Exception:
                    pass
            self._binding = None
            self._broker = None
        self.flush()

    def record(self, event: Any) -> None:
        '''Journal one event (safe to call from any thread).

        Args:
            event (Any): The broker event.
        '''
        run_id = str(getattr(event, 'run_id', '') or '')
        record = self._event_record(event)
        with self._lock:
            count = self._counts.get(run_id, 0)
            if count >= self.max_events:
                return
            self._counts[run_id] = count + 1
            self._buffer.append(record)
            due = (time.time() - self._last_flush) >= self.flush_every or self.flush_every == 0
        if due:
            self.flush()

    @staticmethod
    def _event_record(event: Any) -> Dict[str, Any]:
        '''Build the JSON record for one event.

        Args:
            event (Any): The broker event.

        Returns:
            Dict[str, Any]: ``run_id``, ``type``, ``ts`` plus the payload.
        '''
        payload = getattr(event, 'payload', None) or {}
        record = {
            'ts': getattr(event, 'created_at', time.time()),
            'run_id': str(getattr(event, 'run_id', '') or ''),
            'type': str(getattr(event, 'event_type', '') or ''),
            'agent': str(getattr(event, 'agent_name', '') or ''),
            'turn': getattr(event, 'turn', 1),
            'payload': dict(payload) if isinstance(payload, dict) else {},
        }
        for extra in ('tool_name', 'tool_id'):
            value = getattr(event, extra, '')
            if value:
                record[extra] = value
        return record

    def flush(self) -> int:
        '''Write the buffer to disk.

        Returns:
            int: Number of records written by this call.
        '''
        with self._lock:
            if not self._buffer:
                self._last_flush = time.time()
                return 0
            buffered = self._buffer
            self._buffer = []
            self._last_flush = time.time()
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in buffered:
            grouped.setdefault(record['run_id'], []).append(record)
        written = 0
        for run_id, records in grouped.items():
            path = self.path_for(run_id or 'unknown')
            try:
                with open(path, 'a', encoding='utf-8') as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                        written += 1
            except OSError as exc:
                logger.warning('persistence: cannot write %s: %s', path, exc)
        self.written += written
        return written

    def events(self, run_id: str, *, event_type: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        '''Read one run's journal back.

        Args:
            run_id (str): Run identifier.
            event_type (Optional[str]): When given, only this event type.

        Yields:
            Dict[str, Any]: The recorded events in order.
        '''
        for record in _read_lines(self.path_for(run_id)):
            if event_type is None or record.get('type') == event_type:
                yield record

    def runs(self) -> List[str]:
        '''List the run ids that have a journal on disk.

        Returns:
            List[str]: Run ids, most recently modified first.
        '''
        return _run_ids(self.runs_dir)

    def close(self) -> None:
        '''Detach from the broker and flush the buffer.'''
        self.detach()
        self.flush()


class CheckpointStore:
    '''Conversation checkpoints, one JSONL file per run.

    A checkpoint is written by ``Agent(checkpoint=store.callback)`` after every
    finished step, so a crashed or stopped run can be rewound to its last
    complete step and continued with a new request.
    '''

    def __init__(self, directory: str = 'orchify-runs', *, keep: int = 200):
        '''Configure the store.

        Args:
            directory (str): Root directory (created when missing).
            keep (int): Number of trailing checkpoints kept per run on disk.
        '''
        self.directory = os.path.abspath(directory)
        self.checkpoints_dir = _ensure_dir(os.path.join(self.directory, 'checkpoints'))
        self.keep = max(1, int(keep))
        self._lock = threading.Lock()
        self.saved = 0

    def path_for(self, run_id: str) -> str:
        '''Return the checkpoint path for one run.

        Args:
            run_id (str): Run identifier.

        Returns:
            str: Absolute path of the JSONL file.
        '''
        return os.path.join(self.checkpoints_dir, f'{_safe_name(run_id)}.jsonl')

    @property
    def callback(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        '''Return a callable suitable for ``Agent(checkpoint=...)``.

        Returns:
            Callable[[Dict[str, Any]], Dict[str, Any]]: The bound ``save`` method.
        '''
        return self.save

    def save(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        '''Persist one checkpoint.

        Args:
            checkpoint (Dict[str, Any]): ``{'run_id', 'step', 'agent', 'model',
                'messages'}`` as handed over by the agent.

        Returns:
            Dict[str, Any]: The record written (with ``ts`` added).
        '''
        record = dict(checkpoint)
        record.setdefault('ts', time.time())
        run_id = str(record.get('run_id') or 'unknown')
        path = self.path_for(run_id)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            try:
                with open(path, 'a', encoding='utf-8') as handle:
                    handle.write(line + '\n')
                self.saved += 1
                self._trim(path)
            except OSError as exc:
                logger.warning('persistence: cannot write checkpoint %s: %s', path, exc)
        return record

    def _trim(self, path: str) -> None:
        '''Keep only the most recent ``keep`` checkpoints of a file.

        Args:
            path (str): Checkpoint file to trim.
        '''
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                lines = handle.readlines()
        except OSError:
            return
        if len(lines) <= self.keep:
            return
        kept = lines[-self.keep:]
        try:
            with open(path, 'w', encoding='utf-8') as handle:
                handle.writelines(kept)
        except OSError as exc:
            logger.warning('persistence: cannot trim %s: %s', path, exc)

    def latest(self, run_id: str) -> Optional[Dict[str, Any]]:
        '''Return the newest checkpoint of a run.

        Args:
            run_id (str): Run identifier.

        Returns:
            Optional[Dict[str, Any]]: The record, or None when the run has none.
        '''
        found: Optional[Dict[str, Any]] = None
        for record in _read_lines(self.path_for(run_id)):
            if isinstance(record.get('messages'), list):
                found = record
        return found

    def history(self, run_id: str) -> List[Dict[str, Any]]:
        '''Return every checkpoint of a run, oldest first.

        Args:
            run_id (str): Run identifier.

        Returns:
            List[Dict[str, Any]]: The checkpoint records.
        '''
        return [record for record in _read_lines(self.path_for(run_id))
                if isinstance(record.get('messages'), list)]

    def runs(self) -> List[str]:
        '''List the runs that have a checkpoint on disk.

        Returns:
            List[str]: Run ids, most recently modified first.
        '''
        return _run_ids(self.checkpoints_dir)

    def resume(self, agent: Any, run_id: Optional[str] = None,
               *, messages_mode: Optional[str] = None) -> Optional[Dict[str, Any]]:
        '''Rewind an agent to its last checkpointed conversation.

        Args:
            agent (Any): The Agent to restore into.
            run_id (Optional[str]): Run to resume; defaults to the agent's
                ``last_run_id``.
            messages_mode (Optional[str]): Optionally switch the agent to
                ``'run'`` while resuming so the restored list is not mutated by
                a later concurrent run.

        Returns:
            Optional[Dict[str, Any]]: The checkpoint used, or None when the run
            has no checkpoint (the agent is left untouched).
        '''
        target = run_id or getattr(agent, 'last_run_id', '')
        if not target:
            logger.warning('persistence: resume() needs a run id; nothing restored')
            return None
        record = self.latest(target)
        if record is None:
            logger.warning("persistence: no checkpoint for run '%s'", target)
            return None
        messages = [dict(message) for message in record.get('messages') or []]
        agent.messages = messages
        if messages_mode:
            agent.messages_mode = messages_mode
        logger.info("persistence: agent '%s' resumed run '%s' at step %s",
                    getattr(agent, 'name', '?'), target, record.get('step'))
        return record


def save_agent(agent: Any, path: str) -> str:
    '''Write an agent's history and identity to a JSON file.

    Args:
        agent (Any): The Agent to snapshot.
        path (str): Destination file.

    Returns:
        str: The path written.
    '''
    payload = {
        'ts': time.time(),
        'name': getattr(agent, 'name', ''),
        'model': getattr(agent, 'model', ''),
        'system_prompt': getattr(agent, 'system_prompt', ''),
        'messages_mode': getattr(agent, 'messages_mode', 'agent'),
        'messages': [dict(message) for message in getattr(agent, 'messages', [])],
        'last_run_id': getattr(agent, 'last_run_id', ''),
        'last_answer': getattr(agent, 'last_answer', ''),
    }
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        _ensure_dir(directory)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return path


def load_agent(agent: Any, path: str) -> Dict[str, Any]:
    '''Restore an agent's history from :func:`save_agent`.

    The agent's ``model`` is restored too; ``system_prompt`` is left alone so a
    redeployed prompt is not silently rolled back.

    Args:
        agent (Any): The Agent to restore into.
        path (str): File written by :func:`save_agent`.

    Returns:
        Dict[str, Any]: The decoded snapshot.
    '''
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    messages = payload.get('messages')
    if isinstance(messages, list):
        agent.messages = [dict(message) for message in messages]
    if payload.get('model'):
        agent.model = payload['model']
    if payload.get('messages_mode'):
        agent.messages_mode = payload['messages_mode']
    return payload
