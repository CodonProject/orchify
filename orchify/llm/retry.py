"""Provider-agnostic retry helpers shared by the LLM adapters.

Every adapter talks to its provider through
:meth:`orchify.backend.WebBackend.post_stream_async`, which raises the same
``httpx`` exception types whatever the provider is. Keeping the retry decision in
one place means the OpenAI, Anthropic and Gemini adapters agree on which failures
are transient.
"""

import asyncio
from typing import AsyncGenerator, Callable, Dict, Optional

import httpx

#: HTTP status codes worth retrying: rate limits, timeouts and server errors.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Upper bound for one exponential backoff delay, in seconds.
MAX_RETRY_DELAY = 30.0


def is_retryable(exc: BaseException) -> bool:
    '''Decide whether a provider failure is transient.

    Args:
        exc (BaseException): The exception raised by the HTTP layer.

    Returns:
        bool: True for rate limits, server errors, timeouts and transport errors.
    '''
    if isinstance(exc, httpx.HTTPStatusError):
        response = getattr(exc, 'response', None)
        return response is not None and response.status_code in RETRYABLE_STATUS
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException, TimeoutError, ConnectionError)):
        return True
    return False


def retry_delay(attempt: int, backoff: float) -> float:
    '''Exponential backoff for the given zero-based attempt.

    Args:
        attempt (int): Zero-based attempt index.
        backoff (float): Base delay in seconds.

    Returns:
        float: The delay to sleep, capped at :data:`MAX_RETRY_DELAY`.
    '''
    if backoff <= 0:
        return 0.0
    return min(backoff * (2 ** attempt), MAX_RETRY_DELAY)


async def stream_sse(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, object],
    *,
    max_retries: int = 2,
    backoff: float = 0.5,
    label: str = 'llm',
) -> AsyncGenerator[str, None]:
    '''Stream SSE lines from a provider, retrying only before the first line.

    A failure after the first line is raised instead of retried: the caller has
    already emitted part of the answer, so replaying the request would duplicate
    output.

    Args:
        url (str): Streaming endpoint.
        headers (Dict[str, str]): Request headers.
        payload (Dict[str, object]): JSON request body.
        max_retries (int): Extra attempts after the first failure.
        backoff (float): Base delay for the exponential backoff.
        label (str): Name used in log messages.

    Yields:
        str: Lines from the response stream.
    '''
    from orchify.backend import orchify_web_backend

    attempt = 0
    started = False
    while True:
        try:
            async for line in orchify_web_backend.post_stream_async(url, headers, payload):
                started = True
                yield line
            return
        except Exception as exc:
            if started or attempt >= max(0, int(max_retries)) or not is_retryable(exc):
                raise
            delay = retry_delay(attempt, backoff)
            attempt += 1
            if delay:
                await asyncio.sleep(delay)
