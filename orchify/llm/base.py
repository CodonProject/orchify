from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Literal, Callable, Dict, Any
from functools import partial


@dataclass
class Chunk:
    is_cot: bool
    content: str

    is_assembly_tool: bool = field(default=False)
    assembly_chunk: str    = field(default='')

    total_content: str          = field(default='')
    total_cot_content: str      = field(default='')
    total_tool_call: list[dict] = field(default_factory=list)

    is_cot_end: bool = field(default=False)


@dataclass
class FinalStatus:
    content: str
    reasoning: str
    tool_calls: list[dict] = field(default_factory=list)

    completion_tokens: int = field(default=0)
    prompt_tokens: int = field(default=0)
    prompt_cache_hit_tokens: int = field(default=0)
    prompt_cache_miss_tokens: int = field(default=0)
    total_tokens: int = field(default=0)

    is_abort: bool = field(default=False)

    @property
    def completion_token(self) -> int:
        return self.completion_tokens

    @property
    def prompt_token(self) -> int:
        return self.prompt_tokens

    @property
    def total_token(self) -> int:
        return self.total_tokens


@dataclass
class Response:
    current_chunk: Optional[Chunk]
    final_status: Optional[FinalStatus] = field(default=None)

    is_final: bool = field(default=False)


NextRequest = Callable[..., AsyncGenerator['Response', None]]


class Middleware:
    '''
    LLM request middleware (onion model).

    Implement __call__ as an async generator. Before starting the underlying
    stream you may mutate `kwargs` (messages/model/tools/temperature/top_p/
    json_format/max_tokens/thinking/effort/extra_data). The async generator
    returned wraps the Response stream, allowing observation or transformation
    of each Response (or the final_status), plus error handling/retries around
    the stream.

        class MyMiddleware(Middleware):
            async def __call__(self, kwargs, next_request):
                kwargs['thinking'] = 'disabled'
                try:
                    async for resp in next_request(kwargs):
                        yield resp
                except httpx.HTTPStatusError:
                    ...retry...

    First registered middleware is the outermost.
    '''
    name: str = ''

    def __init__(self, name: str = ''):
        self.name = name or self.__class__.__name__

    async def __call__(self, kwargs: Dict[str, Any], next_request: NextRequest) -> AsyncGenerator[Response, None]:
        raise NotImplementedError('Middleware subclasses must implement __call__.')


class LLMInterface:
    def __init__(self):
        self.middlewares: list['Middleware'] = []

    def use(self, middleware: 'Middleware | list[Middleware] | tuple[Middleware, ...]') -> 'Middleware | list[Middleware]':
        '''
        Register one middleware or a list/tuple of middlewares.
        The first registered middleware is the outermost.
        Returns the input(s), so it can be used as a decorator either way.
        '''
        items = middleware if isinstance(middleware, (list, tuple)) else [middleware]
        for mw in items:
            if not isinstance(mw, Middleware):
                raise TypeError(f'Middleware must be a Middleware instance, got {type(mw)}')
            self.middlewares.append(mw)
        return middleware

    def _dispatch(self, kwargs: Dict[str, Any]) -> AsyncGenerator[Response, None]:
        def core_dispatch(k: Dict[str, Any]) -> AsyncGenerator[Response, None]:
            return self._core_request(**k)

        chain: Callable = core_dispatch
        for mw in reversed(self.middlewares):
            chain = partial(mw, next_request=chain)
        return chain(kwargs)

    async def _core_request(self, **kwargs: Any) -> AsyncGenerator[Response, None]:
        raise NotImplementedError('Subclasses must implement _core_request.')

    async def request(
        self,
        messages: list[dict],
        model: str = 'gpt-4o',
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        json_format: bool = False,
        max_tokens: Optional[int] = None,
        thinking: Literal['disabled', 'enabled', 'auto'] = 'auto',
        effort: Literal['minimal', 'low', 'medium', 'high', 'xhigh'] = 'medium',
        extra_data: Optional[dict] = None,
    ) -> AsyncGenerator[Response, None]:
        kwargs: Dict[str, Any] = {
            'messages': messages,
            'model': model,
            'tools': tools,
            'temperature': temperature,
            'top_p': top_p,
            'json_format': json_format,
            'max_tokens': max_tokens,
            'thinking': thinking,
            'effort': effort,
            'extra_data': extra_data,
        }
        if not self.middlewares:
            async for resp in self._core_request(**kwargs):
                yield resp
            return
        async for resp in self._dispatch(kwargs):
            yield resp