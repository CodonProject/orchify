import asyncio
import threading
import queue
from typing import Generator, AsyncGenerator, Any, Dict, Optional
import httpx


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
        self._client = httpx.AsyncClient(timeout=None)
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

    async def _post_stream_coro(
        self,
        url: str,
        headers: Dict[str, str],
        json_data: Dict[str, Any],
        q: queue.Queue
    ) -> None:
        '''
        Coroutine to make the async POST request and stream chunks into a Queue.

        Args:
            url (str): Target HTTP URL.
            headers (Dict[str, str]): Request headers.
            json_data (Dict[str, Any]): Request payload.
            q (queue.Queue): Thread-safe queue to put received lines/exceptions.
        '''
        try:
            async with self._client.stream('POST', url, headers=headers, json=json_data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    q.put(('line', line))
            q.put(('done', None))
        except Exception as e:
            q.put(('error', e))

    def post_stream(
        self,
        url: str,
        headers: Dict[str, str],
        json_data: Dict[str, Any]
    ) -> Generator[str, None, None]:
        '''
        Makes a streaming POST request via the background asyncio loop. (Synchronous wrapper)

        Args:
            url (str): Target HTTP URL.
            headers (Dict[str, str]): Request headers.
            json_data (Dict[str, Any]): Request payload.

        Returns:
            Generator[str, None, None]: Generator yielding lines of the response stream.
        '''
        q = queue.Queue()
        future = asyncio.run_coroutine_threadsafe(
            self._post_stream_coro(url, headers, json_data, q),
            self._loop
        )

        while True:
            try:
                msg_type, val = q.get(timeout=0.1)
                if msg_type == 'line':
                    yield val
                elif msg_type == 'done':
                    break
                elif msg_type == 'error':
                    raise val
            except queue.Empty:
                if future.done():
                    exc = future.exception()
                    if exc:
                        raise exc
                    break

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
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line


orchify_web_backend = WebBackend()
