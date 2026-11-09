import os
import shutil
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, orchify_broker
from orchify.manager import OrchifyManager
from orchify.store import (MemoryRunStore, RedisRunStore, SQLiteRunStore,
                           run_store_from_env)
from orchify.testing import FakeLLM, Recorder, final, wait_until

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')


def _directory():
    '''Create a fresh directory under the ignored tests/.data tree.'''
    path = os.path.join(_DATA, f'store-{uuid.uuid4().hex[:8]}')
    os.makedirs(path, exist_ok=True)
    return path


def _drop(path):
    '''Remove a test directory.'''
    shutil.rmtree(path, ignore_errors=True)


def _record(run_id, started, status='done', kind='agent', **extra):
    '''A finished-run record shaped like the manager's own.'''
    record = {'run_id': run_id, 'kind': kind, 'status': status, 'name': 'demo',
              'model': 'deepseek-chat', 'started_at': started, 'finished_at': started + 1,
              'tokens': 10, 'cost_usd': 0.5, 'total_steps': 1, 'answer': 'ok'}
    record.update(extra)
    return record


def test_memory_store_orders_filters_and_bounds():
    '''The memory store keeps recency order, filters and a capacity.'''
    store = MemoryRunStore(max_runs=3)
    for index in range(4):
        store.save(_record(f'run-{index}', started=100.0 + index))
    assert store.count() == 3
    assert store.get('run-0') is None, 'the oldest record is evicted'
    assert [item['run_id'] for item in store.list()] == ['run-3', 'run-2', 'run-1']
    assert [item['run_id'] for item in store.list(limit=1)] == ['run-3']
    assert store.list(kind='group') == []
    assert [item['run_id'] for item in store.list(status='done', limit=2)] == ['run-3', 'run-2']
    assert store.delete('run-2') is True
    assert store.delete('run-2') is False
    assert store.clear() == 2
    assert store.count() == 0

    try:
        store.save({'kind': 'agent'})
    except ValueError as exc:
        assert 'run_id' in str(exc)
    else:
        raise AssertionError('a record without a run_id must be rejected')


def test_sqlite_store_persists_across_reopen():
    '''Records survive closing and reopening the same database file.'''
    directory = _directory()
    path = os.path.join(directory, 'runs.db')
    try:
        store = SQLiteRunStore(path)
        store.save(_record('a#1', started=10.0))
        store.save(_record('b#2', started=20.0, kind='group', status='error'))
        assert store.count() == 2
        assert store.snapshot()['store'] == 'sqlite'
        store.close()

        reopened = SQLiteRunStore(path)
        assert reopened.get('a#1')['answer'] == 'ok'
        assert [item['run_id'] for item in reopened.list()] == ['b#2', 'a#1']
        assert [item['run_id'] for item in reopened.list(kind='agent')] == ['a#1']
        assert [item['run_id'] for item in reopened.list(status='error')] == ['b#2']
        assert reopened.get('missing') is None

        reopened.save(_record('a#1', started=30.0, answer='replaced'))
        assert reopened.get('a#1')['answer'] == 'replaced'
        assert reopened.delete('b#2') is True
        assert reopened.clear() == 1
        assert reopened.count() == 0
        reopened.close()
    finally:
        _drop(directory)


def test_sqlite_store_trims_to_max_runs():
    '''max_runs keeps only the newest records.'''
    directory = _directory()
    try:
        store = SQLiteRunStore(os.path.join(directory, 'trim.db'), max_runs=2)
        for index in range(4):
            store.save(_record(f'r{index}', started=100.0 + index))
        assert store.count() == 2
        assert [item['run_id'] for item in store.list()] == ['r3', 'r2']
        store.close()
    finally:
        _drop(directory)


class _FakeRedis:
    '''Minimal in-memory stand-in for the redis client methods we use.'''

    def __init__(self):
        self.hashes = {}
        self.scores = {}
        self.closed = False

    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hmget(self, key, fields):
        table = self.hashes.get(key, {})
        return [table.get(field) for field in fields]

    def hdel(self, key, *fields):
        table = self.hashes.setdefault(key, {})
        removed = 0
        for field in fields:
            removed += 1 if table.pop(field, None) is not None else 0
        return removed

    def hlen(self, key):
        return len(self.hashes.get(key, {}))

    def zadd(self, key, mapping):
        self.scores.setdefault(key, {}).update(mapping)

    def zrange(self, key, start, stop):
        ordered = sorted(self.scores.get(key, {}).items(), key=lambda item: item[1])
        ids = [item[0] for item in ordered]
        return ids[start:] if stop == -1 else ids[start:stop + 1]

    def zrevrange(self, key, start, stop):
        ordered = sorted(self.scores.get(key, {}).items(),
                         key=lambda item: item[1], reverse=True)
        ids = [item[0] for item in ordered]
        return ids[start:] if stop == -1 else ids[start:stop + 1]

    def zrem(self, key, *ids):
        table = self.scores.setdefault(key, {})
        removed = 0
        for run_id in ids:
            removed += 1 if table.pop(run_id, None) is not None else 0
        return removed

    def delete(self, *keys):
        for key in keys:
            self.hashes.pop(key, None)
            self.scores.pop(key, None)

    def close(self):
        self.closed = True


def test_redis_store_uses_hash_and_index():
    '''The Redis store reads newest-first and trims through its sorted set.'''
    client = _FakeRedis()
    store = RedisRunStore('redis://example/0', prefix='test', client=client)
    for index in range(3):
        store.save(_record(f'r{index}', started=100.0 + index))
    assert store.count() == 3
    assert [item['run_id'] for item in store.list()] == ['r2', 'r1', 'r0']
    assert [item['run_id'] for item in store.list(limit=2)] == ['r2', 'r1']
    assert store.get('r1')['answer'] == 'ok'
    assert store.list(status='error') == []
    assert store.delete('r0') is True
    assert store.snapshot()['prefix'] == 'test'
    assert store.clear() == 2
    assert store.count() == 0
    store.close()
    assert client.closed is True


def test_run_store_from_env_parses_specs():
    '''The environment selects memory, SQLite or Redis.'''
    assert run_store_from_env({}).name == 'memory'
    assert run_store_from_env({'ORCHIFY_RUN_STORE': 'memory'}).name == 'memory'

    directory = _directory()
    try:
        sqlite_store = run_store_from_env(
            {'ORCHIFY_RUN_STORE': os.path.join(directory, 'env.db')})
        assert sqlite_store.name == 'sqlite'
        assert sqlite_store.path == os.path.join(directory, 'env.db')
        sqlite_store.close()

        prefixed = run_store_from_env(
            {'ORCHIFY_RUN_STORE': 'sqlite:' + os.path.join(directory, 'prefixed.db')})
        assert prefixed.path == os.path.join(directory, 'prefixed.db')
        prefixed.close()
    finally:
        _drop(directory)

    redis_store = run_store_from_env(
        {'ORCHIFY_RUN_STORE': 'redis://localhost:6379/2'})
    try:
        assert redis_store.name == 'redis'
        assert redis_store.url == 'redis://localhost:6379/2'
    finally:
        redis_store.close()


def test_redis_store_connects_lazily(monkeypatch):
    '''Building a Redis store must not need the redis package until first use.'''
    monkeypatch.setitem(sys.modules, 'redis', None)   # import redis -> ImportError
    store = RedisRunStore('redis://localhost:6379/2')
    assert store.name == 'redis'
    with pytest.raises(RuntimeError) as caught:
        store.count()
    assert 'pip install redis' in str(caught.value)
    snapshot = store.snapshot()
    assert snapshot['runs'] is None and 'pip install redis' in snapshot['error']
    store.close()


def test_manager_writes_finished_runs_to_the_store():
    '''A finished run is persisted and readable by a second manager instance.'''
    store = MemoryRunStore()
    manager = OrchifyManager(store=store)
    other = None
    rec = Recorder()
    try:
        agent = Agent(name='stored', llm=FakeLLM([[final('persisted')]]))
        assert agent.ask('store this', max_steps=1) == 'persisted'
        # The manager persists on its tap thread, so wait for the record itself
        # rather than racing it.
        wait_until(lambda: bool(manager.history(limit=5)), timeout=5.0)
        run_id = manager.history(limit=1)[0]['run_id']

        assert manager.store is store
        record = store.get(run_id)
        assert record is not None, manager.history(limit=5)
        assert record['answer'] == 'persisted'
        assert record['status'] == 'done'
        assert manager.history(limit=1)[0]['run_id'] == run_id

        # A second replica that shares the store sees the run without its own tap.
        other = OrchifyManager(enabled=False, store=store)
        assert other.run(run_id)['answer'] == 'persisted'
        assert other.history(limit=5)[0]['run_id'] == run_id
        assert manager.status()['store']['store'] == 'memory'
    finally:
        rec.close()
        orchify_broker.drain_hooks(timeout=5.0)
        manager.close()
        if other is not None:
            other.close()
