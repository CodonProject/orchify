"""Durable run storage for single- and multi-replica deployments.

The manager keeps live state in memory and, by default, finished runs too. That is
fine for one process; a second replica (or a restart) sees nothing. A ``RunStore``
is the pluggable place finished records are written, and the place reads fall back
to::

    from orchify import OrchifyManager
    from orchify.store import SQLiteRunStore

    manager = OrchifyManager(store=SQLiteRunStore('var/runs.db'))

Three stores ship with Orchify:

* :class:`MemoryRunStore` — the default; bounded in-process cache.
* :class:`SQLiteRunStore` — one file, safe across processes and replicas on the
  same host (WAL mode, one writer at a time).
* :class:`RedisRunStore` — for replicas on different hosts; needs the optional
  ``redis`` package (``pip install redis``).

Every store presents the same tiny API: ``save(record)``, ``get(run_id)``,
``list(limit, kind, status)``, ``delete(run_id)``, ``count()``, ``clear()``,
``close()``. Records are the plain dicts the manager already produces, so a custom
store only has to persist JSON.
"""

import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from orchify.utils import get_logger

logger = get_logger(__name__)

#: Environment variable that selects a store: a filesystem path, ``sqlite:`` path,
#: ``redis://…`` URL, or ``memory``.
STORE_ENV = 'ORCHIFY_RUN_STORE'

#: Keys copied into the indexed columns of :class:`SQLiteRunStore`.
_INDEXED = ('run_id', 'kind', 'status', 'model', 'started_at', 'finished_at',
            'tokens', 'cost_usd', 'total_steps', 'name')


class RunStore:
    '''Interface implemented by every run store.

    Records are JSON-serialisable dicts with at least ``run_id``, ``kind``,
    ``status`` and ``started_at``.
    '''

    #: Store name used in reports.
    name = 'store'

    def save(self, record: Dict[str, Any]) -> None:
        '''Insert or replace one finished run record.

        Args:
            record (Dict[str, Any]): The record to persist.
        '''
        raise NotImplementedError('RunStore subclasses must implement save()')

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        '''Read one record.

        Args:
            run_id (str): Run identifier.

        Returns:
            Optional[Dict[str, Any]]: The record, or None.
        '''
        raise NotImplementedError('RunStore subclasses must implement get()')

    def list(self, limit: Optional[int] = 50, kind: Optional[str] = None,
             status: Optional[str] = None) -> List[Dict[str, Any]]:
        '''Recent records, newest first.

        Args:
            limit (Optional[int]): Maximum records; None means no limit.
            kind (Optional[str]): Only this kind ('agent', 'group', …).
            status (Optional[str]): Only this status ('done', 'error', …).

        Returns:
            List[Dict[str, Any]]: Matching records.
        '''
        raise NotImplementedError('RunStore subclasses must implement list()')

    def delete(self, run_id: str) -> bool:
        '''Remove one record.

        Args:
            run_id (str): Run identifier.

        Returns:
            bool: True when a record was removed.
        '''
        raise NotImplementedError('RunStore subclasses must implement delete()')

    def count(self) -> int:
        '''Number of stored records.'''
        raise NotImplementedError('RunStore subclasses must implement count()')

    def clear(self) -> int:
        '''Remove every record, returning how many were removed.'''
        raise NotImplementedError('RunStore subclasses must implement clear()')

    def close(self) -> None:
        '''Release resources; safe to call repeatedly.'''

    def snapshot(self) -> Dict[str, Any]:
        '''Store metadata for reports.'''
        return {'store': self.name, 'runs': self.count()}

    def __enter__(self) -> 'RunStore':
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class MemoryRunStore(RunStore):
    '''Bounded in-process store (the default).

    Args:
        max_runs (int): Capacity; the oldest record is dropped first.
    '''

    name = 'memory'

    def __init__(self, max_runs: int = 500):
        self.max_runs = max(1, int(max_runs))
        self._lock = threading.Lock()
        self._records: 'OrderedDict[str, Dict[str, Any]]' = OrderedDict()

    def save(self, record: Dict[str, Any]) -> None:
        '''Store one record, keeping insertion order for recency.'''
        run_id = str(record.get('run_id') or '')
        if not run_id:
            raise ValueError('run record needs a run_id')
        with self._lock:
            self._records.pop(run_id, None)
            self._records[run_id] = dict(record)
            while len(self._records) > self.max_runs:
                self._records.popitem(last=False)

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        '''Read one record from the cache.'''
        with self._lock:
            record = self._records.get(str(run_id or ''))
            return dict(record) if record is not None else None

    def list(self, limit: Optional[int] = 50, kind: Optional[str] = None,
             status: Optional[str] = None) -> List[Dict[str, Any]]:
        '''Recent records, newest first.'''
        with self._lock:
            items = [dict(record) for record in reversed(self._records.values())]
        if kind is not None:
            items = [item for item in items if item.get('kind') == kind]
        if status is not None:
            items = [item for item in items if item.get('status') == status]
        if limit is not None and limit >= 0:
            items = items[:limit]
        return items

    def delete(self, run_id: str) -> bool:
        '''Remove one record.'''
        with self._lock:
            return self._records.pop(str(run_id or ''), None) is not None

    def count(self) -> int:
        '''Number of cached records.'''
        with self._lock:
            return len(self._records)

    def clear(self) -> int:
        '''Drop every cached record.'''
        with self._lock:
            removed = len(self._records)
            self._records.clear()
            return removed

    def close(self) -> None:
        '''No resources to release.'''


class SQLiteRunStore(RunStore):
    '''Durable store in one SQLite file (WAL mode, one writer at a time).

    Multiple processes — including several replicas on the same host — can share
    the file: readers run concurrently, writers serialise on SQLite's own lock.

    Args:
        path (str): Database file; parent directories are created.
        max_runs (int): Trim to this many newest records after each save; 0 keeps all.
    '''

    name = 'sqlite'

    def __init__(self, path: str = 'orchify-runs/runs.db', *, max_runs: int = 0):
        self.path = str(path)
        self.max_runs = max(0, int(max_runs))
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = self._connect()

    def _connect(self) -> Any:
        import sqlite3

        conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute(
                'CREATE TABLE IF NOT EXISTS runs ('
                'run_id TEXT PRIMARY KEY, kind TEXT, status TEXT, model TEXT,'
                'name TEXT, started_at REAL, finished_at REAL, tokens INTEGER,'
                'cost_usd REAL, total_steps INTEGER, payload TEXT NOT NULL,'
                'updated_at REAL NOT NULL)'
            )
            conn.execute('CREATE INDEX IF NOT EXISTS runs_started '
                         'ON runs (started_at DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS runs_kind_status '
                         'ON runs (kind, status, started_at DESC)')
        return conn

    def save(self, record: Dict[str, Any]) -> None:
        '''Insert or replace one record.'''
        run_id = str(record.get('run_id') or '')
        if not run_id:
            raise ValueError('run record needs a run_id')
        payload = json.dumps(record, default=str)
        row = [run_id] + [_row_value(record, key) for key in _INDEXED[1:]] + \
              [payload, time.time()]
        with self._lock, self._conn as conn:
            conn.execute(
                'INSERT INTO runs (run_id, kind, status, model, name, started_at,'
                ' finished_at, tokens, cost_usd, total_steps, payload, updated_at)'
                ' VALUES (?,?,?,?,?,?,?,?,?,?,?,?)'
                ' ON CONFLICT(run_id) DO UPDATE SET kind=excluded.kind,'
                ' status=excluded.status, model=excluded.model, name=excluded.name,'
                ' started_at=excluded.started_at, finished_at=excluded.finished_at,'
                ' tokens=excluded.tokens, cost_usd=excluded.cost_usd,'
                ' total_steps=excluded.total_steps, payload=excluded.payload,'
                ' updated_at=excluded.updated_at',
                row,
            )
            if self.max_runs:
                conn.execute(
                    'DELETE FROM runs WHERE run_id IN ('
                    ' SELECT run_id FROM runs ORDER BY started_at DESC LIMIT -1 OFFSET ?)',
                    (self.max_runs,),
                )

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        '''Read one record.'''
        with self._lock:
            row = self._conn.execute('SELECT payload FROM runs WHERE run_id = ?',
                                     (str(run_id or ''),)).fetchone()
        return json.loads(row['payload']) if row is not None else None

    def list(self, limit: Optional[int] = 50, kind: Optional[str] = None,
             status: Optional[str] = None) -> List[Dict[str, Any]]:
        '''Recent records, newest first.'''
        clauses, params = [], []
        if kind is not None:
            clauses.append('kind = ?')
            params.append(kind)
        if status is not None:
            clauses.append('status = ?')
            params.append(status)
        sql = 'SELECT payload FROM runs'
        if clauses:
            sql += ' WHERE ' + ' AND '.join(clauses)
        sql += ' ORDER BY started_at DESC, run_id DESC'
        if limit is not None and limit >= 0:
            sql += ' LIMIT ?'
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(row['payload']) for row in rows]

    def delete(self, run_id: str) -> bool:
        '''Remove one record.'''
        with self._lock, self._conn as conn:
            cursor = conn.execute('DELETE FROM runs WHERE run_id = ?',
                                  (str(run_id or ''),))
            return cursor.rowcount > 0

    def count(self) -> int:
        '''Number of stored records.'''
        with self._lock:
            row = self._conn.execute('SELECT COUNT(*) AS n FROM runs').fetchone()
        return int(row['n'] if row is not None else 0)

    def clear(self) -> int:
        '''Delete every record.'''
        with self._lock, self._conn as conn:
            cursor = conn.execute('DELETE FROM runs')
            return int(cursor.rowcount or 0)

    def close(self) -> None:
        '''Close the database connection.'''
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - sqlite never raises on close
                pass

    def snapshot(self) -> Dict[str, Any]:
        '''Store metadata for reports.'''
        return {'store': self.name, 'runs': self.count(), 'path': self.path}


class RedisRunStore(RunStore):
    '''Shared store for replicas on different hosts (needs ``redis`` installed).

    Records live in one Redis hash plus a sorted set ordered by ``started_at``.
    The connection is lazy: constructing a store (or listing it in ``status()``)
    does not import ``redis`` or open a socket; the first real operation does.

    Args:
        url (Optional[str]): Redis URL; defaults to ``ORCHIFY_REDIS_URL`` or
            ``redis://127.0.0.1:6379/0``.
        prefix (str): Key prefix, so several deployments can share a server.
        client (Any): Pre-built ``redis`` client; skips the import and connect.
        max_runs (int): Trim to this many newest records after each save (per store);
            0 keeps all. Trimming is best-effort and only runs in-memory per process.
    '''

    name = 'redis'

    def __init__(self, url: Optional[str] = None, *, prefix: str = 'orchify:runs',
                 client: Any = None, max_runs: int = 0):
        self.url = url or os.environ.get('ORCHIFY_REDIS_URL') or 'redis://127.0.0.1:6379/0'
        self.prefix = prefix
        self.max_runs = max(0, int(max_runs))
        # The connection is lazy: building a store from a spec (or showing it in
        # ``status()``) must not need the redis package or a reachable server.
        self._client = client
        self._records_key = f'{prefix}:records'
        self._index_key = f'{prefix}:index'

    def _ensure(self) -> Any:
        '''Return a connected client, connecting on first use.'''
        if self._client is None:
            self._client = self._connect()
        return self._client

    def _connect(self) -> Any:
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                'RedisRunStore needs the redis package: pip install redis '
                '(or pass client=... with an existing connection).'
            ) from exc
        return redis.Redis.from_url(self.url, decode_responses=True)

    def save(self, record: Dict[str, Any]) -> None:
        '''Insert or replace one record.'''
        run_id = str(record.get('run_id') or '')
        if not run_id:
            raise ValueError('run record needs a run_id')
        score = float(record.get('started_at') or time.time())
        client = self._ensure()
        client.hset(self._records_key, run_id, json.dumps(record, default=str))
        client.zadd(self._index_key, {run_id: score})
        if self.max_runs:
            stale = client.zrange(self._index_key, 0, -self.max_runs - 1)
            if stale:
                client.hdel(self._records_key, *stale)
                client.zrem(self._index_key, *stale)

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        '''Read one record.'''
        raw = self._ensure().hget(self._records_key, str(run_id or ''))
        return json.loads(raw) if raw else None

    def list(self, limit: Optional[int] = 50, kind: Optional[str] = None,
             status: Optional[str] = None) -> List[Dict[str, Any]]:
        '''Recent records, newest first.'''
        stop = -1 if limit is None or limit < 0 else int(limit) - 1
        if stop == -2:
            return []
        client = self._ensure()
        ids = client.zrevrange(self._index_key, 0, stop)
        if not ids:
            return []
        rows = client.hmget(self._records_key, ids)
        items = [json.loads(raw) for raw in rows if raw]
        if kind is not None:
            items = [item for item in items if item.get('kind') == kind]
        if status is not None:
            items = [item for item in items if item.get('status') == status]
        return items

    def delete(self, run_id: str) -> bool:
        '''Remove one record.'''
        key = str(run_id or '')
        client = self._ensure()
        removed = int(client.hdel(self._records_key, key) or 0)
        client.zrem(self._index_key, key)
        return removed > 0

    def count(self) -> int:
        '''Number of stored records.'''
        return int(self._ensure().hlen(self._records_key) or 0)

    def clear(self) -> int:
        '''Delete every record.'''
        removed = self.count()
        self._ensure().delete(self._records_key, self._index_key)
        return removed

    def close(self) -> None:
        '''Close the client connection when it exposes ``close()``.'''
        client, self._client = self._client, None
        closer = getattr(client, 'close', None)
        if callable(closer):
            closer()

    def snapshot(self) -> Dict[str, Any]:
        '''Store metadata for reports (a missing redis package is reported, not raised).'''
        info: Dict[str, Any] = {'store': self.name, 'url': self.url,
                                'prefix': self.prefix}
        try:
            info['runs'] = self.count()
        except Exception as exc:  # noqa: BLE001 - status must never break a report
            info['runs'] = None
            info['error'] = str(exc)
        return info


def _row_value(record: Dict[str, Any], key: str) -> Any:
    '''Coerce one record field into a SQLite-friendly value.'''
    value = record.get(key)
    if value is None:
        return None
    if key in ('tokens', 'total_steps'):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if key in ('started_at', 'finished_at', 'cost_usd'):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return str(value)


def run_store_from_env(environ: Optional[Dict[str, str]] = None) -> RunStore:
    '''Build the store named by ``ORCHIFY_RUN_STORE``.

    Accepted values: ``memory`` (default), a filesystem path or ``sqlite:<path>``
    for :class:`SQLiteRunStore`, and ``redis://…``/``rediss://…`` for
    :class:`RedisRunStore`.

    Args:
        environ (Optional[Dict[str, str]]): Mapping to read; ``os.environ`` by default.

    Returns:
        RunStore: The configured store.
    '''
    env = os.environ if environ is None else environ
    spec = str(env.get(STORE_ENV) or '').strip()
    if not spec or spec.lower() in ('memory', 'none', 'off'):
        return MemoryRunStore()
    lowered = spec.lower()
    if lowered.startswith(('redis://', 'rediss://', 'redis+')):
        return RedisRunStore(spec)
    if lowered.startswith('sqlite:'):
        return SQLiteRunStore(spec.split(':', 1)[1])
    return SQLiteRunStore(spec)


__all__ = ['RunStore', 'MemoryRunStore', 'SQLiteRunStore', 'RedisRunStore',
           'run_store_from_env', 'STORE_ENV']
