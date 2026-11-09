from typing import Callable, Any, Dict, Optional, List, Union
from collections import OrderedDict

import asyncio
import concurrent.futures as _futures
import inspect
import json
import os
import threading
import time

from .utils import analyze_tool_function, get_logger


logger = get_logger(__name__)


DEFAULT_CACHE_SIZE = 256


def _read_int_env(name: str, default: int, minimum: int = 0) -> int:
    '''Read an int from the environment, falling back when missing/invalid.'''
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _read_tool_timeout() -> Optional[float]:
    '''Read the async-tool timeout from the environment.

    Only awaitable results can be bounded (a synchronous function cannot be
    interrupted from here).

    Returns:
        Optional[float]: ``ORCHIFY_TOOL_TIMEOUT`` in seconds, or None to wait for
        as long as the coroutine takes.
    '''
    raw = os.getenv('ORCHIFY_TOOL_TIMEOUT')
    if raw is None or raw.strip() == '':
        return None
    try:
        value = float(raw.strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _read_cache_default_ttl() -> Optional[float]:
    '''Read the default TTL applied to ``cache=True`` tools.

    ``cache=True`` keeps its documented meaning (cache forever) unless
    ``ORCHIFY_TOOL_CACHE_TTL`` is set, in which case that many seconds bound every
    entry so a long-lived tool cannot serve permanently stale results.

    Returns:
        Optional[float]: TTL in seconds, or None when unset/invalid/<= 0.
    '''
    raw = os.getenv('ORCHIFY_TOOL_CACHE_TTL')
    if raw is None or raw.strip() == '':
        return None
    try:
        value = float(raw.strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


class ToolCache:
    '''Thread-safe LRU result store that several tools may share.

    Pass one instance as ``cache_store`` to every ``Tool`` that should share
    cached results. Keys are namespaced by the tool name, so tools sharing a
    store never collide; single-flight also coalesces across them.

    Attributes:
        lock (threading.RLock): Guards ``data`` and ``inflight``.
        data (OrderedDict): ``key -> (expires_at, value)`` entries.
        inflight (Dict[str, list]): Single-flight slots, ``key -> [lock, waiters]``.
    '''

    def __init__(self) -> None:
        '''Create an empty shared store.'''
        self.lock = threading.RLock()
        self.data: 'OrderedDict[str, tuple]' = OrderedDict()
        self.inflight: Dict[str, list] = {}

    def clear(self) -> None:
        '''Drop every cached value and in-flight slot.'''
        with self.lock:
            self.data.clear()
            self.inflight.clear()


class Tool:
    '''A function exposed to the model as a callable tool.

    A Tool wraps a Python callable: its signature and docstring become the
    JSON-schema description sent to the LLM, and each call runs the callable
    with the model-provided arguments.

    Attributes:
        func (Callable): The wrapped callable.
        name (str): Tool name advertised to the model.
        description (str): Tool description advertised to the model.
        parameters (List[dict]): Parameter schema; inferred from the signature
            and docstring when not supplied.
        cache (Union[None, bool, float]): Caching policy: ``None``, ``False`` and
            ``0`` disable caching, ``True`` caches forever, a number is a TTL in
            seconds.
        cache_size (int): Maximum number of cached results (LRU eviction);
            0 means unbounded.
        timeout (Optional[float]): Seconds to wait for an async result.
        require_approval (bool): Park every call for manual approval.
        ask_question (bool): Treat every call as an open question to the human
            operator instead of work.
        is_async (bool): True when the wrapped callable is a coroutine function.
        last_cached (bool): Whether the most recent execution was served from
            cache.
    '''

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
        parameters_schema: Optional[Dict[str, Any]] = None,
        cache: Optional[Any] = None,
        require_approval: bool = False,
        cache_size: Optional[int] = None,
        timeout: Optional[float] = None,
        ask_question: bool = False,
        cache_store: Optional['ToolCache'] = None,
        retry: int = 0,
        retry_on: Optional[Union[type, tuple]] = None,
        backoff: float = 0.5,
        max_backoff: float = 8.0,
        on_error: str = 'raise',
    ):
        '''Build a tool around ``func``.

        Args:
            func (Callable): The callable to expose to the model.
            name (Optional[str]): Tool name; defaults to the function name.
            description (Optional[str]): Description for the model; defaults to the
                function docstring.
            parameters (Optional[List[Dict[str, Any]]]): Explicit parameter schema;
                inferred from the signature and docstring when omitted.
            parameters_schema (Optional[Dict[str, Any]]): Raw JSON Schema for the
                arguments (``{"type": "object", "properties": {...}}``), used
                verbatim instead of ``parameters``. Needed for schemas the flat
                ``parameters`` list cannot express (nested objects, enums) — for
                example tools imported from an MCP server.
            cache (Union[None, bool, float]): Caching policy: ``None``, ``False`` and
                ``0`` disable caching, ``True`` caches forever, a number is a TTL in
                seconds.
            require_approval (bool): Park each call for manual approval before it
                executes.
            cache_size (Optional[int]): Maximum number of cached results (LRU
                eviction); defaults to ``ORCHIFY_TOOL_CACHE_SIZE`` (256), where 0
                means unbounded.
            timeout (Optional[float]): Seconds to wait for an async (``async def``)
                result; defaults to ``ORCHIFY_TOOL_TIMEOUT``, where None waits
                forever. Synchronous tools cannot be interrupted, so the value does
                not apply to them.
            ask_question (bool): When True the tool is an open question to the human
                operator rather than work: the owning Agent parks the call in the
                approval pool with ``kind='question'``, emits
                ``run:approval:required`` and feeds the free-text answer back as the
                tool result. The tool function itself is never executed.
            cache_store (Optional[ToolCache]): Shared store to use instead of a
                private one; every tool given the same instance shares cached
                results (and single-flight) while keys remain namespaced per tool.
            retry (int): Extra attempts after a failure (0 means try once).
            retry_on (Optional[Union[type, tuple]]): Exception classes that may be
                retried; None retries any ``Exception``.
            backoff (float): Seconds to wait before the first retry; doubles per
                attempt up to ``max_backoff``.
            max_backoff (float): Ceiling for the exponential backoff.
            on_error (str): What to do when the attempts are exhausted:
                ``'raise'`` (default, the Agent reports the error to the model) or
                ``'return'`` (return ``'[tool error] ...'`` as the result so the
                model can react).

        Note:
            A tool whose function returns an awaitable has its result resolved on
            the shared backend event loop, so async tools behave like sync ones for
            the Agent. Calling such a tool from that loop thread raises RuntimeError.
        '''
        self.func = func

        analysis = analyze_tool_function(func)

        self.name: str = name or func.__name__
        self.description: str = description or analysis.get('docstring', 'No description provided.')
        self.parameters: List[Dict[str, Any]] = parameters or analysis.get('parameters', [])
        self.parameters_schema: Optional[Dict[str, Any]] = (
            dict(parameters_schema) if parameters_schema else None
        )
        self.cache = cache
        self.require_approval: bool = bool(require_approval)
        self.ask_question: bool = bool(ask_question)
        self.cache_size: int = (
            cache_size if cache_size is not None
            else _read_int_env('ORCHIFY_TOOL_CACHE_SIZE', DEFAULT_CACHE_SIZE, minimum=0)
        )
        self.timeout: Optional[float] = timeout if timeout is not None else _read_tool_timeout()
        self.retry: int = max(0, int(retry or 0))
        self.retry_on = retry_on
        self.backoff: float = max(0.0, float(backoff or 0.0))
        self.max_backoff: float = max(self.backoff, float(max_backoff or 0.0))
        self.on_error: str = str(on_error or 'raise').strip().lower() or 'raise'
        self.is_async: bool = inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)
        if cache is True:
            self.cache_ttl: Optional[float] = _read_cache_default_ttl()
        elif isinstance(cache, (int, float)) and cache:
            self.cache_ttl = float(cache)
        else:
            self.cache_ttl = None

        self._store = cache_store if cache_store is not None else ToolCache()
        self._cache: 'OrderedDict[str, tuple]' = self._store.data
        self._inflight: Dict[str, list] = self._store.inflight
        self._cache_lock = self._store.lock
        self._hits: int = 0
        self._misses: int = 0
        self._coalesced: int = 0
        self._evictions: int = 0
        self.last_cached: bool = False
        self._info = self.build_info()

    @property
    def needs_approval(self) -> bool:
        '''True when this tool must be manually confirmed before it runs.'''
        return bool(self.require_approval)

    @property
    def info(self) -> Dict[str, Any]:
        '''Return the pre-built tool description.

        Returns:
            Dict[str, Any]: The OpenAI-compatible description dict (see
            ``build_info``).
        '''
        return self._info

    def build_info(self) -> Dict[str, Any]:
        '''Build the OpenAI function-calling description.

        Returns:
            Dict[str, Any]: A description dict that can be serialised to JSON and
            sent to the LLM API.
        '''
        if self.parameters_schema:
            schema = json.loads(json.dumps(self.parameters_schema))
            schema.setdefault('type', 'object')
            schema.setdefault('properties', {})
            return {
                'type': 'function',
                'function': {
                    'name': self.name,
                    'description': self.description,
                    'parameters': schema,
                },
            }

        json_schema_properties = {}
        required_params = []
        py_to_json_type_map = {
            'str': 'string',
            'int': 'integer',
            'float': 'number',
            'bool': 'boolean',
            'list': 'array',
            'dict': 'object'
        }
        for param in self.parameters:
            param_name = param['name']

            param_type = py_to_json_type_map.get(param.get('annotation', 'str'), 'string')

            json_schema_properties[param_name] = {
                'type': param_type,
                'description': param.get('description', '')
            }

            if param.get('required', False):
                required_params.append(param_name)

        if not self.description.startswith('A tool:') and \
           not self.description.startswith('An Agent:') and \
           not self.description.startswith('A Group:'):
            self.description = f'A tool:\n{self.description}'

        tool_info = {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': {
                    'type': 'object',
                    'properties': json_schema_properties,
                }
            }
        }

        if required_params:
            tool_info['function']['parameters']['required'] = required_params

        return tool_info

    def __call__(self, **kwargs):
        '''Allows the tool instance to be called like a function.'''
        return self.execute(**kwargs)

    def _call(self, kwargs: Dict[str, Any]) -> Any:
        '''Invoke the wrapped function, resolving an awaitable result.'''
        result = self.func(**kwargs)
        if inspect.isawaitable(result):
            result = self._resolve_awaitable(result)
        return result

    def _resolve_awaitable(self, awaitable: Any) -> Any:
        '''Run an async tool's coroutine on the backend loop.

        The backend loop is the one Agent runs execute on, so an async tool can use
        the same httpx client and any other loop-bound resource the library holds.

        Args:
            awaitable (Any): The coroutine (or other awaitable) to resolve.

        Returns:
            Any: The value the awaitable produced.
        '''
        from .backend import orchify_web_backend

        if threading.current_thread() is orchify_web_backend._thread:
            close = getattr(awaitable, 'close', None)
            if callable(close):
                close()
            raise RuntimeError(
                f"async tool '{self.name}' cannot be resolved on the backend event-loop "
                'thread; await the coroutine instead of calling Tool.execute()'
            )

        async def _await_result():
            return await awaitable

        future = asyncio.run_coroutine_threadsafe(_await_result(), orchify_web_backend._loop)

        if self.timeout is None:
            return future.result()
        try:
            return future.result(timeout=self.timeout)
        except (TimeoutError, _futures.TimeoutError) as exc:
            future.cancel()
            raise TimeoutError(
                f"async tool '{self.name}' timed out after {self.timeout}s"
            ) from exc

    def _cache_lookup(self, key: str) -> tuple:
        '''Return (found, value) for a cache key, updating hit/miss counters.'''
        now = time.time()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return False, None
            expires_at, value = entry
            if expires_at is not None and expires_at <= now:
                self._cache.pop(key, None)
                self._misses += 1
                return False, None
            self._cache.move_to_end(key)
            self._hits += 1
            return True, value

    def _cache_store(self, key: str, value: Any) -> None:
        '''Store a result, evicting the least recently used entry over the cap.'''
        with self._cache_lock:
            expires_at = None if self.cache_ttl is None else time.time() + float(self.cache_ttl)
            self._cache[key] = (expires_at, value)
            self._cache.move_to_end(key)
            if self.cache_size:
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
                    self._evictions += 1

    def _acquire_slot(self, key: str) -> tuple:
        '''Enter the single-flight gate for a cache key.

        The first caller for a key computes the result while the others wait for
        it on an event (so they all wake at once instead of serialising on a
        lock; each blocked follower costs one thread either way, but a failed
        leader lets exactly one of them take over).

        Args:
            key (str): Serialised call arguments.

        Returns:
            tuple: ``(slot, leader)``; every caller must hand the slot back to
            ``_release_slot()`` so the last one out clears the in-flight entry.
        '''
        with self._cache_lock:
            slot = self._inflight.get(key)
            if slot is None:
                slot = [threading.Event(), 0]
                self._inflight[key] = slot
                leader = True
            else:
                leader = False
            slot[1] += 1
        return slot, leader

    def _release_slot(self, key: str, slot: list) -> None:
        with self._cache_lock:
            slot[1] -= 1
            if slot[1] <= 0:
                self._inflight.pop(key, None)
            slot[0].set()

    def execute(self, **kwargs: Any) -> Any:
        '''Execute the tool.

        Args:
            **kwargs (Any): Parameters supplied by the Agent, keyed by parameter
                name.

        Returns:
            Any: The result of the tool's function.
        '''
        self.last_cached = False

        if not self.cache:
            return self._call_with_retry(kwargs)

        key = f'{self.name}\x00' + json.dumps(kwargs, sort_keys=True, default=str)

        found, value = self._cache_lookup(key)
        if found:
            self.last_cached = True
            return value

        retried = False
        while True:
            slot, leader = self._acquire_slot(key)
            try:
                if leader:
                    if retried:
                        found, value = self._cache_lookup(key)
                        if found:
                            self.last_cached = True
                            return value
                    value = self._call_with_retry(kwargs)
                    self._cache_store(key, value)
                    return value
                slot[0].wait()
            finally:
                self._release_slot(key, slot)
            found, value = self._cache_lookup(key)
            if found:
                self._coalesced += 1
                self.last_cached = True
                return value
            retried = True

    def _call_with_retry(self, kwargs: Dict[str, Any]) -> Any:
        '''Call the tool, retrying transient failures per the configured policy.

        Args:
            kwargs (Dict[str, Any]): Tool arguments.

        Returns:
            Any: The tool's result, or an error string when ``on_error='return'``.

        Raises:
            BaseException: The last failure when ``on_error='raise'`` (the default)
                and every attempt failed.
        '''
        attempts = self.retry + 1
        delay = self.backoff
        last_exc: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                return self._call(kwargs)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                last_exc = exc
                if attempt + 1 >= attempts or not self._retryable(exc):
                    break
                if delay > 0:
                    time.sleep(delay)
                delay = min(delay * 2, self.max_backoff) if delay else 0.0
        if self.on_error == 'return':
            return f'[tool error] {type(last_exc).__name__}: {last_exc}'
        raise last_exc  # type: ignore[misc]

    def _retryable(self, exc: BaseException) -> bool:
        '''Whether an exception matches ``retry_on``.

        Args:
            exc (BaseException): The failure raised by the tool.

        Returns:
            bool: True when the call may be retried.
        '''
        if self.retry_on is None:
            return True
        return isinstance(exc, self.retry_on)

    def cache_info(self) -> Dict[str, Any]:
        '''Snapshot of this tool's cache: size, bound, hit/miss and eviction counters.'''
        with self._cache_lock:
            return {
                'enabled': bool(self.cache),
                'ttl': self.cache_ttl,
                'size': len(self._cache),
                'maxsize': self.cache_size or None,
                'hits': self._hits,
                'misses': self._misses,
                'coalesced': self._coalesced,
                'evictions': self._evictions,
                'inflight': len(self._inflight),
            }

    def clear_cache(self) -> None:
        '''Drop every cached result this tool can reach (a shared store clears for all).'''
        self._store.clear()

    def reset_cache_stats(self) -> None:
        '''Zero the hit/miss/coalesced/eviction counters (keeps cached values).'''
        with self._cache_lock:
            self._hits = 0
            self._misses = 0
            self._coalesced = 0
            self._evictions = 0

    def __repr__(self) -> str:
        '''Provides a string representation of the Tool instance.'''
        return f"Tool(name='{self.name}')"


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[List[Dict[str, Any]]] = None,
    cache: Optional[Any] = None,
    require_approval: bool = False,
    cache_size: Optional[int] = None,
    timeout: Optional[float] = None,
    ask_question: bool = False,
    cache_store: Optional[ToolCache] = None,
    retry: int = 0,
    retry_on: Optional[Union[type, tuple]] = None,
    backoff: float = 0.5,
    max_backoff: float = 8.0,
    on_error: str = 'raise',
):
    '''Turn a function into a Tool, usable as a decorator.

    Args:
        name (Optional[str]): Tool name; defaults to the function name.
        description (Optional[str]): Description for the model; defaults to the
            function docstring.
        parameters (Optional[List[Dict[str, Any]]]): Explicit parameter schema;
            inferred from the signature and docstring when omitted.
        cache (Union[None, bool, float]): Caching policy (see :class:`Tool`).
        require_approval (bool): Park each call for manual approval.
        cache_size (Optional[int]): Maximum cached results; 0 means unbounded.
        timeout (Optional[float]): Seconds to wait for an async result.
        ask_question (bool): Treat the call as an open question to the operator.
        cache_store (Optional[ToolCache]): Shared store to use (see :class:`Tool`).
        retry (int): Extra attempts after a failure (see :class:`Tool`).
        retry_on (Optional[Union[type, tuple]]): Exception classes allowed to retry.
        backoff (float): Initial retry delay in seconds.
        max_backoff (float): Ceiling for the exponential backoff.
        on_error (str): ``'raise'`` (default) or ``'return'`` an error string.

    Returns:
        Callable: A decorator that wraps the function in a :class:`Tool`.
    '''
    def decorator(func: Callable) -> Tool:
        '''Wrap ``func`` in a Tool.

        Args:
            func (Callable): The function to expose.

        Returns:
            Tool: The tool wrapping ``func``.
        '''
        return Tool(
            func=func,
            name=name,
            description=description,
            parameters=parameters,
            cache=cache,
            require_approval=require_approval,
            cache_size=cache_size,
            timeout=timeout,
            ask_question=ask_question,
            cache_store=cache_store,
            retry=retry,
            retry_on=retry_on,
            backoff=backoff,
            max_backoff=max_backoff,
            on_error=on_error,
        )
    return decorator
