import asyncio
import os
import threading
from typing import AsyncGenerator, Any, Dict, Optional
import httpx


# Request timeout for the shared HTTP client. Without one, a stalled provider
# hangs the whole run forever and the only escape is a control:stop.
# ORCHIFY_HTTP_TIMEOUT accepts seconds (applied to every phase), or 0 / none /
# off to disable the timeout entirely.
DEFAULT_HTTP_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0)


def _read_http_timeout():
    '''Resolve ORCHIFY_HTTP_TIMEOUT into an httpx timeout (or None = no timeout).'''
    raw = os.getenv('ORCHIFY_HTTP_TIMEOUT')
    if not raw or raw.strip() == '':
        return DEFAULT_HTTP_TIMEOUT
    value = raw.strip().lower()
    if value in ('0', 'none', 'off', 'false', '-1', 'inf', 'infinite'):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return DEFAULT_HTTP_TIMEOUT
    if seconds <= 0:
        return None
    return httpx.Timeout(seconds)


class WebBackend:
    '''
    WEB Backend running an asyncio event loop in a dedicated background thread.

    Provides a thread-safe way to make asynchronous HTTP requests with streaming
    capabilities using a shared httpx.AsyncClient.
    '''

    def __init__(self) -> None:
        '''
        Initializes the WebBackend and starts the background loop.
        '''
        self._loop = asyncio.new_event_loop()
        self._client: Optional[httpx.AsyncClient] = None
        self.timeout = _read_http_timeout()
        self._thread = threading.Thread(
            target=self._run_loop,
            name='OrchifyWebBackend',
            daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        '''
        Target function for the background thread running the event loop.
        '''
        asyncio.set_event_loop(self._loop)
        self._client = httpx.AsyncClient(timeout=self.timeout)
        self._loop.run_forever()

    async def _cleanup(self) -> None:
        '''
        Closes the AsyncClient.
        '''
        if self._client:
            await self._client.aclose()

    def shutdown(self) -> None:
        '''
        Stops the event loop and joins the background thread.
        '''
        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._cleanup(), self._loop)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()

    async def post_stream_async(
        self,
        url: str,
        headers: Dict[str, str],
        json_data: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        '''
        Makes an asynchronous streaming POST request.

        Args:
            url (str): Target HTTP URL.
            headers (Dict[str, str]): Request headers.
            json_data (Dict[str, Any]): Request payload.

        Yields:
            str: Lines of the response stream.
        '''
        async with self._client.stream('POST', url, headers=headers, json=json_data) as response:
            if response.status_code >= 400:
                # Read the error body so raise_for_status() details survive the
                # stream context (and so callers can see Retry-After / messages).
                await response.aread()
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line


orchify_web_backend = WebBackend()
