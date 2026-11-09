import asyncio
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional


def _read_int_env(name: str, default: int = 0) -> int:
    '''A non-negative int from the environment, else `default`.'''
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _poll_interval() -> float:
    '''Seconds between gate polls (env ORCHIFY_POLL_INTERVAL, default 0.05).'''
    raw = os.getenv('ORCHIFY_POLL_INTERVAL')
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return 0.05


class LimitGate:
    '''At most `limit` holders at a time (limit 0 = unlimited, no bookkeeping).

    Waiters park on an ``asyncio.Event`` that ``release()`` sets, so a freed slot
    is handed over immediately instead of after a poll interval. Waiting is FIFO:
    an arriving caller may not jump the queue. When a ``stop`` callable is given
    it is re-checked at most every ``poll`` seconds (the wait itself is still
    event driven, so a release wakes it at once).
    '''

    def __init__(self, limit: int = 0, name: str = 'gate') -> None:
        '''Create a gate.

        Args:
            limit (int): Maximum number of concurrent holders; 0 means unlimited
                and skips all bookkeeping.
            name (str): Label used in logs and in ``gate_stats()``.
        '''
        self.name = name
        self.limit = max(0, int(limit or 0))
        self._lock = threading.Lock()
        self._in_use = 0
        self._acquired = 0
        self._waited = 0
        self._waiters: deque = deque()

    @property
    def enabled(self) -> bool:
        '''Whether the gate limits anything at all.

        Returns:
            bool: True when a positive limit was configured.
        '''
        return self.limit > 0

    @property
    def in_use(self) -> int:
        '''Number of slots currently held by callers.

        Returns:
            int: Slots in use.
        '''
        with self._lock:
            return self._in_use

    def would_block(self) -> bool:
        '''True when a caller arriving now would have to wait.'''
        with self._lock:
            return self.limit > 0 and (self._in_use >= self.limit or bool(self._waiters))

    def _drop_waiter_locked(self, waiter: list) -> bool:
        '''Remove `waiter` from the queue; True when it was still queued.'''
        for index, current in enumerate(self._waiters):
            if current is waiter:
                del self._waiters[index]
                return True
        return False

    def _wake_next_locked(self) -> None:
        '''Wake the oldest waiter so it can take a free slot (caller holds the lock).'''
        while self._waiters:
            loop, event = self._waiters[0]
            try:
                loop.call_soon_threadsafe(event.set)
                return
            except RuntimeError:
                self._waiters.popleft()

    async def acquire(
        self,
        stop: Optional[Callable[[], bool]] = None,
        poll: Optional[float] = None,
    ) -> bool:
        '''Take one slot, waiting while the gate is full.

        Args:
            stop (Optional[Callable[[], bool]]): Polled while waiting; returning True
                gives up without taking a slot. It is not called when a slot is free.
            poll (Optional[float]): How often ``stop`` may be re-checked while
                waiting; defaults to the configured poll interval.

        Returns:
            bool: True when a slot was taken, False when ``stop`` asked to give up.
        '''
        if not self.enabled:
            return True
        interval = poll if poll is not None and poll > 0 else _poll_interval()
        loop = asyncio.get_running_loop()
        waiter: Optional[list] = None
        waited = False
        try:
            while True:
                with self._lock:
                    if waiter is None:
                        waiter = [loop, asyncio.Event()]
                        self._waiters.append(waiter)
                    # Clear before checking so a wake-up cannot be lost between
                    # the check and the wait below.
                    waiter[1].clear()
                    if self._in_use < self.limit and self._waiters[0] is waiter:
                        self._drop_waiter_locked(waiter)
                        self._in_use += 1
                        self._acquired += 1
                        if waited:
                            self._waited += 1
                        return True
                waited = True
                if stop is None:
                    await waiter[1].wait()
                else:
                    try:
                        await asyncio.wait_for(waiter[1].wait(), timeout=interval)
                    except asyncio.TimeoutError:
                        pass
                    if stop():
                        return False
        finally:
            if waiter is not None:
                with self._lock:
                    self._drop_waiter_locked(waiter)
                    if self._in_use < self.limit:
                        # We left the queue (with or without a slot) and there is
                        # room left, so the next waiter must not wait for a
                        # release() that may never come.
                        self._wake_next_locked()

    def release(self) -> None:
        '''Release a slot taken by acquire() (never drops below zero).'''
        with self._lock:
            if self._in_use > 0:
                self._in_use -= 1
            if self._in_use < self.limit:
                self._wake_next_locked()

    def stats(self) -> Dict[str, Any]:
        '''Snapshot for the manager / status output.'''
        with self._lock:
            return {
                'name': self.name,
                'enabled': self.limit > 0,
                'limit': self.limit,
                'in_use': self._in_use,
                'acquired': self._acquired,
                'waited': self._waited,
            }


class RateLimiter:
    '''At most `per_minute` acquisitions in a rolling 60 s window (0 = off).'''

    def __init__(self, per_minute: int = 0, name: str = 'rpm') -> None:
        '''Create a rate limiter.

        Args:
            per_minute (int): Maximum acquisitions in a rolling 60 s window; 0
                disables the limiter.
            name (str): Label used in logs and in ``gate_stats()``.
        '''
        self.name = name
        self.per_minute = max(0, int(per_minute or 0))
        self._times: deque = deque()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        '''Whether the limiter limits anything at all.

        Returns:
            bool: True when a positive per-minute rate was configured.
        '''
        return self.per_minute > 0

    def _trim(self, now: float) -> None:
        cutoff = now - 60.0
        while self._times and self._times[0] <= cutoff:
            self._times.popleft()

    def _take(self) -> bool:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            if len(self._times) < self.per_minute:
                self._times.append(now)
                return True
        return False

    def _wait_time(self) -> float:
        '''Seconds until the rolling window has room again (0 when it already has).'''
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            if len(self._times) < self.per_minute:
                return 0.0
            return max(0.0, self._times[0] + 60.0 - now)

    async def acquire(
        self,
        stop: Optional[Callable[[], bool]] = None,
        poll: Optional[float] = None,
    ) -> bool:
        '''Wait for a free slot in the window; False when `stop()` said to give up.

        The wait is timed to the moment the oldest request leaves the window
        instead of re-checking every poll interval; ``poll`` only bounds how
        often ``stop`` is looked at.
        '''
        if not self.enabled:
            return True
        interval = poll if poll is not None and poll > 0 else _poll_interval()
        while True:
            if self._take():
                return True
            if stop is not None and stop():
                return False
            delay = self._wait_time()
            if delay <= 0:
                delay = interval
            elif stop is not None:
                delay = min(delay, interval)
            await asyncio.sleep(delay)

    def stats(self) -> Dict[str, Any]:
        '''Snapshot for the manager / status output.'''
        with self._lock:
            self._trim(time.monotonic())
            used = len(self._times)
        return {
            'name': self.name,
            'enabled': self.per_minute > 0,
            'per_minute': self.per_minute,
            'used_last_minute': used,
        }


_gates_lock = threading.Lock()
_run_gate: Optional[LimitGate] = None
_llm_gate: Optional[LimitGate] = None
_llm_rpm: Optional[RateLimiter] = None


def get_run_gate() -> LimitGate:
    '''Gate on concurrent runs (env ORCHIFY_MAX_CONCURRENT_RUNS).'''
    global _run_gate
    if _run_gate is None:
        with _gates_lock:
            if _run_gate is None:
                _run_gate = LimitGate(_read_int_env('ORCHIFY_MAX_CONCURRENT_RUNS', 0), 'runs')
    return _run_gate


def get_llm_gate() -> LimitGate:
    '''Gate on in-flight LLM requests (env ORCHIFY_MAX_CONCURRENT_LLM_CALLS).'''
    global _llm_gate
    if _llm_gate is None:
        with _gates_lock:
            if _llm_gate is None:
                _llm_gate = LimitGate(_read_int_env('ORCHIFY_MAX_CONCURRENT_LLM_CALLS', 0), 'llm')
    return _llm_gate


def get_llm_rpm() -> RateLimiter:
    '''Rolling-minute limit on LLM requests (env ORCHIFY_LLM_RPM).'''
    global _llm_rpm
    if _llm_rpm is None:
        with _gates_lock:
            if _llm_rpm is None:
                _llm_rpm = RateLimiter(_read_int_env('ORCHIFY_LLM_RPM', 0), 'llm_rpm')
    return _llm_rpm


def reset_gates() -> None:
    '''Drop the cached gates so the next getter rebuilds them from the env (tests).'''
    global _run_gate, _llm_gate, _llm_rpm
    with _gates_lock:
        _run_gate = None
        _llm_gate = None
        _llm_rpm = None


def gate_stats() -> Dict[str, Any]:
    '''Snapshot of every gate (creating them from the current env).'''
    return {
        'runs': get_run_gate().stats(),
        'llm': get_llm_gate().stats(),
        'llm_rpm': get_llm_rpm().stats(),
    }
