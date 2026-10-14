import asyncio
import os
import threading
import time
from typing import AsyncGenerator, Any, Dict, Optional
import httpx


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
    '''Web backend owning an asyncio event loop on a background thread.

    Agent runs, HTTP requests and plugin coroutines are submitted to
    ``_loop``, so the library can stream responses without blocking the caller.
    One shared ``httpx.AsyncClient`` is reused for every request.
    '''

    def __init__(self) -> None:
        '''Initialize the backend and start its background thread.'''
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
        '''Run the private event loop until ``shutdown()`` stops it.'''
        asyncio.set_event_loop(self._loop)
        self._client = httpx.AsyncClient(timeout=self.timeout)
        try:
            self._loop.run_forever()
        finally:
            client, self._client = self._client, None
            if client is not None:
                try:
                    self._loop.run_until_complete(client.aclose())
                except Exception:
                    pass

    async def _cleanup(self) -> None:
        '''Close the shared ``httpx.AsyncClient``.'''
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    def shutdown(self) -> None:
        '''Stop the event loop and join the background thread.

        The shared client is closed before the loop is stopped, so the cleanup
        coroutine always runs to completion instead of being destroyed pending.
        Safe to call repeatedly and right after construction.
        '''
        loop = self._loop
        if not loop.is_running() and self._thread.is_alive():
            deadline = time.monotonic() + 5.0
            while (not loop.is_running() and self._thread.is_alive()
                   and time.monotonic() < deadline):
                time.sleep(0.01)
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._cleanup(), loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    async def post_stream_async(
        self,
        url: str,
        headers: Dict[str, str],
        json_data: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        '''Open a streaming POST request.

        Args:
            url (str): Target HTTP URL.
            headers (Dict[str, str]): Request headers.
            json_data (Dict[str, Any]): Request payload.

        Yields:
            str: Lines of the response stream.
        '''
        async with self._client.stream('POST', url, headers=headers, json=json_data) as response:
            if response.status_code >= 400:
                await response.aread()
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line


orchify_web_backend = WebBackend()
