from typing import Callable, Any, Dict, Optional, List
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
        cache: Optional[Any] = None,
        require_approval: bool = False,
        cache_size: Optional[int] = None,
        timeout: Optional[float] = None,
        ask_question: bool = False,
    ):
        '''Build a tool around ``func``.

        Args:
            func (Callable): The callable to expose to the model.
            name (Optional[str]): Tool name; defaults to the function name.
            description (Optional[str]): Description for the model; defaults to the
                function docstring.
            parameters (Optional[List[Dict[str, Any]]]): Explicit parameter schema;
                inferred from the signature and docstring when omitted.
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
        self.cache = cache
        self.require_approval: bool = bool(require_approval)
        self.ask_question: bool = bool(ask_question)
        self.cache_size: int = (
            cache_size if cache_size is not None
            else _read_int_env('ORCHIFY_TOOL_CACHE_SIZE', DEFAULT_CACHE_SIZE, minimum=0)
        )
        self.timeout: Optional[float] = timeout if timeout is not None else _read_tool_timeout()
        self.is_async: bool = inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)

        self._cache: 'OrderedDict[str, tuple]' = OrderedDict()
        self._inflight: Dict[str, list] = {}
        self._cache_lock = threading.RLock()
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
            expires_at = None if self.cache is True else time.time() + float(self.cache)
            self._cache[key] = (expires_at, value)
            self._cache.move_to_end(key)
            if self.cache_size:
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
                    self._evictions += 1

    def _acquire_slot(self, key: str) -> tuple:
        '''Enter the single-flight gate for a cache key.

        The first caller for a key computes the result while the others wait for it.

        Args:
            key (str): Serialised call arguments.

        Returns:
            tuple: ``(lock, leader)``; every caller must hand the lock back to
            ``_release_slot()`` so the last one out clears the in-flight entry.
        '''
        with self._cache_lock:
            slot = self._inflight.get(key)
            if slot is None:
                slot = [threading.Lock(), 0]
                self._inflight[key] = slot
                leader = True
            else:
                leader = False
            slot[1] += 1
            lock = slot[0]
        lock.acquire()
        return lock, leader

    def _release_slot(self, key: str, lock: threading.Lock) -> None:
        try:
            lock.release()
        except RuntimeError:
            pass
        with self._cache_lock:
            slot = self._inflight.get(key)
            if slot is None:
                return
            slot[1] -= 1
            if slot[1] <= 0:
                self._inflight.pop(key, None)

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
            return self._call(kwargs)

        key = json.dumps(kwargs, sort_keys=True, default=str)

        found, value = self._cache_lookup(key)
        if found:
            self.last_cached = True
            return value

        lock, leader = self._acquire_slot(key)
        try:
            if not leader:
                found, value = self._cache_lookup(key)
                if found:
                    self._coalesced += 1
                    self.last_cached = True
                    return value

            value = self._call(kwargs)
            self._cache_store(key, value)
            return value
        finally:
            self._release_slot(key, lock)

    def cache_info(self) -> Dict[str, Any]:
        '''Snapshot of this tool's cache: size, bound, hit/miss and eviction counters.'''
        with self._cache_lock:
            return {
                'enabled': bool(self.cache),
                'ttl': None if self.cache is True else (float(self.cache) if self.cache else None),
                'size': len(self._cache),
                'maxsize': self.cache_size or None,
                'hits': self._hits,
                'misses': self._misses,
                'coalesced': self._coalesced,
                'evictions': self._evictions,
                'inflight': len(self._inflight),
            }

    def clear_cache(self) -> None:
        '''Drop every cached result of this tool.'''
        with self._cache_lock:
            self._cache.clear()

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
        )
    return decorator
