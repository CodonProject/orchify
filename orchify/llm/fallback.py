"""Model fallback chain: keep answering when the primary provider fails.

``FallbackLLM`` wraps several :class:`orchify.llm.base.LLMInterface` instances and
tries them in order. A provider that fails with a transient error (rate limit,
5xx, timeout, transport error) is skipped for the rest of the request and then
cooled down for ``cooldown`` seconds, so a dead endpoint does not cost one retry
per call.

Usage::

    from orchify import Agent
    from orchify.llm.fallback import FallbackLLM
    from orchify.llm.openai import OpenAICompat

    llm = FallbackLLM([
        OpenAICompat(model='deepseek-flash'),
        OpenAICompat(model='gpt-4o-mini'),
    ])
    agent = Agent(llm=llm, model='deepseek-flash')

A provider is only abandoned *before* it produced output: once a chunk reached
the agent, replaying the request would duplicate the answer, so the error is
raised instead.
"""

import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from orchify.llm.base import LLMInterface, Response
from orchify.llm.retry import is_retryable
from orchify.utils import get_logger

logger = get_logger(__name__)


def _provider_name(provider: Any, index: int) -> str:
    '''Human readable provider name: its ``name`` attribute or the class name.'''
    name = getattr(provider, 'name', None)
    if name:
        return str(name)
    return type(provider).__name__ or str(index)


class FallbackLLM(LLMInterface):
    '''Try several providers in order, cooling down the ones that fail.

    Args:
        providers (List[LLMInterface]): Providers in preference order.
        max_failures (int): Failures before a provider is cooled down.
        cooldown (float): Seconds a failed provider is skipped.
    '''

    name = 'fallback'

    def __init__(self, providers: Optional[List[LLMInterface]] = None, *,
                 max_failures: int = 1, cooldown: float = 60.0):
        super().__init__()
        self.providers: List[LLMInterface] = []
        self.max_failures = max(1, int(max_failures))
        self.cooldown = max(0.0, float(cooldown))
        self._failures: List[int] = []
        self._down_until: List[float] = []
        self.attempts = 0
        self.failures = 0
        self.switches = 0
        self.last_provider = ''
        self.last_error = ''
        for provider in providers or []:
            self.add(provider)

    def add(self, provider: LLMInterface) -> LLMInterface:
        '''Append a provider to the chain.

        Args:
            provider (LLMInterface): Provider to append.

        Returns:
            LLMInterface: The provider, so calls can be chained.
        '''
        if not hasattr(provider, '_dispatch'):
            raise TypeError(f'FallbackLLM needs LLMInterface instances, got {type(provider)!r}')
        self.providers.append(provider)
        self._failures.append(0)
        self._down_until.append(0.0)
        return provider

    def __len__(self) -> int:
        '''Number of providers in the chain.'''
        return len(self.providers)

    def available(self) -> List[int]:
        '''Indices of the providers that are not cooling down right now.'''
        now = time.monotonic()
        return [index for index in range(len(self.providers)) if self._down_until[index] <= now]

    def reset(self) -> None:
        '''Clear failure counters and cooldowns.'''
        self._failures = [0] * len(self.providers)
        self._down_until = [0.0] * len(self.providers)

    def snapshot(self) -> Dict[str, Any]:
        '''Return the chain state: providers, failures, cooldowns and counters.'''
        now = time.monotonic()
        return {
            'providers': [_provider_name(provider, index) for index, provider in enumerate(self.providers)],
            'available': self.available(),
            'failures': list(self._failures),
            'cooldown_left': [round(max(0.0, until - now), 3) for until in self._down_until],
            'attempts': self.attempts,
            'switches': self.switches,
            'last_provider': self.last_provider,
            'last_error': self.last_error,
        }

    def _record_failure(self, index: int, exc: BaseException) -> None:
        '''Count a failure and cool the provider down once it reaches the limit.'''
        self._failures[index] += 1
        self.failures += 1
        self.last_error = f'{type(exc).__name__}: {exc}'
        if self._failures[index] >= self.max_failures:
            self._down_until[index] = time.monotonic() + self.cooldown
            logger.warning(
                'fallback: provider %d (%s) disabled for %.1fs after %d failure(s): %s',
                index, _provider_name(self.providers[index], index),
                self.cooldown, self._failures[index], self.last_error,
            )

    async def _core_request(self, **kwargs: Any) -> AsyncGenerator[Response, None]:
        '''Stream from the first provider that answers.

        Args:
            **kwargs (Any): The standard ``_core_request`` request parameters.

        Yields:
            Response: The streamed responses of the winning provider.

        Raises:
            BaseException: The last provider error when no provider answers.
        '''
        order = self.available()
        if not order:
            # Everything is cooling down: still try the chain rather than fail
            # outright, but keep the preference order.
            order = list(range(len(self.providers)))
        if not order:
            raise RuntimeError('FallbackLLM has no providers configured')
        first = order[0]
        last_exc: Optional[BaseException] = None
        reasons: List[str] = []
        for index in order:
            provider = self.providers[index]
            reason = provider.missing_credentials() if hasattr(provider, 'missing_credentials') else None
            if reason:
                self._down_until[index] = float('inf')
                reasons.append(f'{_provider_name(provider, index)}: {reason}')
                continue
            self.attempts += 1
            emitted = False
            try:
                async for response in provider._dispatch(dict(kwargs), scope='*'):
                    emitted = True
                    name = _provider_name(provider, index)
                    if self.last_provider != name:
                        if self.last_provider:
                            self.switches += 1
                        self.last_provider = name
                    yield response
            except Exception as exc:
                if emitted or not is_retryable(exc):
                    raise
                self._record_failure(index, exc)
                last_exc = exc
                reasons.append(f'{_provider_name(provider, index)}: {type(exc).__name__}: {exc}')
                continue
            else:
                self._failures[index] = 0
                self._down_until[index] = 0.0
                if index != first:
                    self.switches += 1
                return
        if last_exc is not None:
            raise last_exc
        raise RuntimeError('No LLM provider is available: ' + '; '.join(reasons))


__all__ = ['FallbackLLM']
