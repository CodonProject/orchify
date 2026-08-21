import asyncio
import threading
from typing import AsyncGenerator, Any, Dict, Optional
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
