from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

import os
import sys

from .backend import orchify_web_backend, _read_http_timeout
from .gate import reset_gates
from .broker import _read_poll_interval, _read_run_outcome_limit, orchify_broker
from .utils import configure_logging


def _as_bool(value: Any) -> bool:
    '''Coerce a setting to a bool, treating 0/false/no/off/'' as False.

    Args:
        value (Any): Raw value from a caller or the environment.

    Returns:
        bool: The coerced flag.
    '''
    if isinstance(value, str):
        return value.strip().lower() not in ('', '0', 'false', 'no', 'off', 'none')
    return bool(value)


@dataclass
class OrchifyConfig:
    '''Every environment variable orchify reads, as one object.

    A field left as None means "not set": the reader falls back to its built-in
    default. ``configure()`` writes the fields you pass to ``os.environ`` and
    refreshes the values that are read at call time, so most changes take effect
    without re-creating agents.

    Attributes:
        api_key (Optional[str]): API key (``API_KEY`` / ``OPENAI_API_KEY``).
        base_url (Optional[str]): Provider base URL (``BASE_URL``).
        model_id (Optional[str]): Default model (``MODEL_ID``).
        poll_interval (Optional[float]): Seconds between control polls.
        http_timeout (Optional[float]): Per-request HTTP read timeout; 0 disables.
        log_level (Optional[str]): Logger level name.
        log_echo (Optional[bool]): Echo log records to stdout.
        tool_workers (Optional[int]): Threads in the shared tool executor.
        tool_timeout (Optional[float]): Default per-tool timeout.
        tool_cache_size (Optional[int]): Entries per tool result cache.
        pause_timeout (Optional[float]): Default pause deadline per agent.
        approval_timeout (Optional[float]): Default approval/question deadline.
        run_timeout (Optional[float]): Default ``wait()`` deadline.
        max_call_depth (Optional[int]): Maximum agent-to-agent call depth.
        budget_tokens (Optional[int]): Default per-agent token budget.
        budget_usd (Optional[float]): Default per-agent spend budget.
        budget_seconds (Optional[float]): Default per-agent wall-clock budget.
        price_table (Optional[str]): Model price table for cost tracking.
        vote_timeout (Optional[float]): Seconds a voting group waits for votes.
        max_concurrent_runs (Optional[int]): Global in-flight run limit.
        max_concurrent_llm_calls (Optional[int]): Global in-flight LLM limit.
        llm_rpm (Optional[int]): Provider requests per minute.
        manager_enabled (Optional[bool]): Collect events in the global manager.
        manager_history (Optional[int]): Finished runs the manager keeps.
        manager_events (Optional[int]): Events the manager keeps in memory.
        manager_sink (Optional[str]): JSONL path the manager appends to.
        llm_max_retries (Optional[int]): Provider retries per request.
        llm_retry_backoff (Optional[float]): Base seconds between retries.
        run_outcome_limit (Optional[int]): Finished-run outcomes the broker keeps.
    '''

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_id: Optional[str] = None
    poll_interval: Optional[float] = None
    http_timeout: Optional[float] = None
    log_level: Optional[str] = None
    log_echo: Optional[bool] = None
    tool_workers: Optional[int] = None
    tool_timeout: Optional[float] = None
    tool_cache_size: Optional[int] = None
    pause_timeout: Optional[float] = None
    approval_timeout: Optional[float] = None
    run_timeout: Optional[float] = None
    max_call_depth: Optional[int] = None
    budget_tokens: Optional[int] = None
    budget_usd: Optional[float] = None
    budget_seconds: Optional[float] = None
    price_table: Optional[str] = None
    vote_timeout: Optional[float] = None
    max_concurrent_runs: Optional[int] = None
    max_concurrent_llm_calls: Optional[int] = None
    llm_rpm: Optional[int] = None
    manager_enabled: Optional[bool] = None
    manager_history: Optional[int] = None
    manager_events: Optional[int] = None
    manager_sink: Optional[str] = None
    llm_max_retries: Optional[int] = None
    llm_retry_backoff: Optional[float] = None
    run_outcome_limit: Optional[int] = None

    ENV_NAMES: ClassVar[Dict[str, str]] = {}
    INT_FIELDS: ClassVar[tuple] = (
        'tool_workers', 'tool_cache_size', 'max_call_depth', 'budget_tokens',
        'max_concurrent_runs', 'max_concurrent_llm_calls', 'llm_rpm',
        'manager_history', 'manager_events', 'llm_max_retries',
        'run_outcome_limit',
    )
    FLOAT_FIELDS: ClassVar[tuple] = (
        'poll_interval', 'http_timeout', 'tool_timeout', 'pause_timeout',
        'approval_timeout', 'run_timeout', 'budget_usd', 'budget_seconds',
        'vote_timeout', 'llm_retry_backoff',
    )
    BOOL_FIELDS: ClassVar[tuple] = ('log_echo', 'manager_enabled')

    @classmethod
    def from_env(cls) -> 'OrchifyConfig':
        '''Read the current environment into a config.

        Returns:
            OrchifyConfig: One field per setting; None where the environment
            variable is unset.
        '''
        config = cls()
        for name, env_name in cls.ENV_NAMES.items():
            raw = os.getenv(env_name)
            if raw is None or (isinstance(raw, str) and raw.strip() == ''):
                continue
            setattr(config, name, cls._coerce(name, raw))
        return config

    @classmethod
    def _coerce(cls, name: str, value: Any) -> Any:
        '''Coerce one setting to the type its field expects.

        Args:
            name (str): Field name.
            value (Any): Raw value.

        Returns:
            Any: The coerced value.

        Raises:
            ValueError: When the value does not fit the field's type.
        '''
        if value is None:
            return None
        if name in cls.INT_FIELDS:
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{name} expects an integer, got {value!r}') from exc
        if name in cls.FLOAT_FIELDS:
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{name} expects a number, got {value!r}') from exc
        if name in cls.BOOL_FIELDS:
            return _as_bool(value)
        return str(value)

    def apply(self, *, refresh: bool = True) -> 'OrchifyConfig':
        '''Write this config to the environment and refresh runtime caches.

        A None field removes its environment variable, so the reader falls back
        to its built-in default.

        Args:
            refresh (bool): Also refresh the settings that are read at call time
                (poll interval, gates, logging, HTTP timeout, manager switch).

        Returns:
            OrchifyConfig: This config, for chaining.
        '''
        for name, env_name in self.ENV_NAMES.items():
            value = getattr(self, name)
            if value is None:
                os.environ.pop(env_name, None)
            elif name in self.BOOL_FIELDS:
                os.environ[env_name] = '1' if _as_bool(value) else '0'
            elif name in self.INT_FIELDS:
                os.environ[env_name] = str(int(value))
            elif name in self.FLOAT_FIELDS:
                os.environ[env_name] = repr(float(value))
            else:
                os.environ[env_name] = str(value)
        if refresh:
            _refresh_runtime(self)
        return self

    def as_dict(self) -> Dict[str, Any]:
        '''Return the set fields, keyed by setting name.

        Returns:
            Dict[str, Any]: Only the fields that hold a value.
        '''
        return {name: getattr(self, name) for name in self.ENV_NAMES
                if getattr(self, name) is not None}

    def resolved(self) -> Dict[str, Any]:
        '''Return every field with its effective value and environment name.

        Returns:
            Dict[str, Any]: One entry per setting with ``value`` (the resolved
            value, or the reader's default when unset) and ``env`` (the
            environment variable that holds it).
        '''
        timeout = _read_http_timeout()
        defaults = {
            'model_id': 'gpt-4o',
            'poll_interval': _read_poll_interval(),
            'http_timeout': None if timeout is None else {
                'connect': timeout.connect,
                'read': timeout.read,
                'write': timeout.write,
                'pool': timeout.pool,
            },
            'log_level': 'INFO',
            'log_echo': True,
            'tool_workers': 32,
            'tool_cache_size': 256,
            'max_call_depth': 8,
            'vote_timeout': 300.0,
            'manager_enabled': True,
            'manager_history': 200,
            'manager_events': 500,
            'llm_max_retries': 2,
            'llm_retry_backoff': 0.5,
            'run_outcome_limit': 256,
        }
        resolved: Dict[str, Any] = {}
        for name, env_name in self.ENV_NAMES.items():
            value = getattr(self, name)
            if value is None:
                value = defaults.get(name)
            resolved[name] = {'value': value, 'env': env_name}
        return resolved


OrchifyConfig.ENV_NAMES = {
    'api_key': 'API_KEY',
    'base_url': 'BASE_URL',
    'model_id': 'MODEL_ID',
    'poll_interval': 'ORCHIFY_POLL_INTERVAL',
    'http_timeout': 'ORCHIFY_HTTP_TIMEOUT',
    'log_level': 'ORCHIFY_LOG_LEVEL',
    'log_echo': 'ORCHIFY_LOG_ECHO',
    'tool_workers': 'ORCHIFY_TOOL_WORKERS',
    'tool_timeout': 'ORCHIFY_TOOL_TIMEOUT',
    'tool_cache_size': 'ORCHIFY_TOOL_CACHE_SIZE',
    'pause_timeout': 'ORCHIFY_PAUSE_TIMEOUT',
    'approval_timeout': 'ORCHIFY_APPROVAL_TIMEOUT',
    'run_timeout': 'ORCHIFY_RUN_TIMEOUT',
    'max_call_depth': 'ORCHIFY_MAX_CALL_DEPTH',
    'budget_tokens': 'ORCHIFY_BUDGET_TOKENS',
    'budget_usd': 'ORCHIFY_BUDGET_USD',
    'budget_seconds': 'ORCHIFY_BUDGET_SECONDS',
    'price_table': 'ORCHIFY_PRICE_TABLE',
    'vote_timeout': 'ORCHIFY_VOTE_TIMEOUT',
    'max_concurrent_runs': 'ORCHIFY_MAX_CONCURRENT_RUNS',
    'max_concurrent_llm_calls': 'ORCHIFY_MAX_CONCURRENT_LLM_CALLS',
    'llm_rpm': 'ORCHIFY_LLM_RPM',
    'manager_enabled': 'ORCHIFY_MANAGER_ENABLED',
    'manager_history': 'ORCHIFY_MANAGER_HISTORY',
    'manager_events': 'ORCHIFY_MANAGER_EVENTS',
    'manager_sink': 'ORCHIFY_MANAGER_SINK',
    'llm_max_retries': 'ORCHIFY_LLM_MAX_RETRIES',
    'llm_retry_backoff': 'ORCHIFY_LLM_RETRY_BACKOFF',
    'run_outcome_limit': 'ORCHIFY_RUN_OUTCOME_LIMIT',
}


def _refresh_runtime(config: 'OrchifyConfig') -> None:
    '''Refresh the caches that hold values read at call time.

    Args:
        config (OrchifyConfig): The config that was just applied.

    Returns:
        None: Nothing to return.
    '''
    interval = _read_poll_interval()
    for module_name in ('orchify', 'orchify.broker', 'orchify.agent'):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, 'POLL_INTERVAL'):
            setattr(module, 'POLL_INTERVAL', interval)
    orchify_broker._run_outcome_limit = _read_run_outcome_limit()
    reset_gates()
    configure_logging()
    timeout = _read_http_timeout()
    orchify_web_backend.timeout = timeout
    client = getattr(orchify_web_backend, '_client', None)
    if client is not None:
        try:
            client.timeout = timeout
        except Exception:
            pass
    try:
        from .manager import orchify_manager
    except Exception:
        return
    if config.manager_enabled is not None and bool(orchify_manager.enabled) != bool(config.manager_enabled):
        orchify_manager.set_enabled(config.manager_enabled)
    if config.manager_sink is not None and orchify_manager.sink_path != config.manager_sink:
        orchify_manager.set_sink(config.manager_sink)


def config() -> 'OrchifyConfig':
    '''Return the settings currently in the environment.

    Returns:
        OrchifyConfig: Snapshot of every orchify setting; None means unset.
    '''
    return OrchifyConfig.from_env()


def configure(**values: Any) -> 'OrchifyConfig':
    '''Set orchify settings from Python instead of the environment.

    Values accepted here are the field names of :class:`OrchifyConfig`, and a
    None removes that setting so the reader falls back to its default. Settings
    read at call time (filters, gates, logging, manager switch) are refreshed
    immediately; settings read when an object is built (timeouts, budgets, tool
    cache size, vote timeout) apply to the objects created afterwards.

    Args:
        **values (Any): Setting name to new value.

    Returns:
        OrchifyConfig: The applied config.

    Raises:
        TypeError: When a name is not a known setting.
        ValueError: When a value does not fit the setting's type.
    '''
    known = set(OrchifyConfig.ENV_NAMES)
    unknown = sorted(name for name in values if name not in known)
    if unknown:
        raise TypeError(
            f"Unknown orchify setting(s): {', '.join(unknown)}. "
            f"Known settings: {', '.join(sorted(known))}."
        )
    current = OrchifyConfig.from_env()
    for name, value in values.items():
        setattr(current, name, OrchifyConfig._coerce(name, value))
    return current.apply()


__all__ = ['OrchifyConfig', 'config', 'configure']
