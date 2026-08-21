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


@dataclass
class Response:
    current_chunk: Optional[Chunk]
    final_status: Optional[FinalStatus] = field(default=None)

    is_final: bool = field(default=False)


NextRequest = Callable[..., AsyncGenerator['Response', None]]


def scope_matches(scope, agent_name) -> bool:
    '''
    Whether a plugin scope matches an agent name.

    Acceptable forms for `scope` (or list of them):
      - None / '*':  matches anything (global).
      - 'agent':     matches any named agent.
      - 'agent:foo': matches the agent named `foo`.
      - callable:    called with the agent name string; return truthy to match.
    '''
    if scope is None or scope == '*':
        return True
    items = scope if isinstance(scope, (list, tuple, set)) else [scope]
    agent_name = agent_name or ''
    for item in items:
        if callable(item):
            try:
                if item(agent_name):
                    return True
            except Exception:
                pass
            continue
        if item in ('*', 'agent'):
            return True
        if item == f'agent:{agent_name}':
            return True
    return False


class Middleware:
    '''
    LLM request middleware (onion model).

    Implement __call__ as an async generator. Before starting the underlying
    stream you may mutate `kwargs` (messages/model/tools/temperature/top_p/
    json_format/max_tokens/thinking/effort/extra_data/scope). The async generator
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

    `scope` (None/'*' for global) restricts the middleware to requests whose
    `scope` matches (see scope_matches) — usually the agent name.
    '''
    name: str = ''
    scope: Any = None

    def __init__(self, name: str = '', scope=None):
        self.name = name or self.__class__.__name__
        self.scope = scope

    def matches(self, agent_name) -> bool:
        return scope_matches(self.scope, agent_name)

    async def __call__(self, kwargs: Dict[str, Any], next_request: NextRequest) -> AsyncGenerator[Response, None]:
        raise NotImplementedError('Middleware subclasses must implement __call__.')


class _FuncMiddleware(Middleware):
    '''Wraps a plain async-callable (async generator of Response) as a Middleware.'''

    def __init__(self, func: NextRequest):
        super().__init__(name=getattr(func, '__name__', 'middleware'))
        self._func = func

    async def __call__(self, kwargs: Dict[str, Any], next_request: NextRequest) -> AsyncGenerator[Response, None]:
        async for resp in self._func(kwargs, next_request):
            yield resp


class LLMInterface:
    def __init__(self):
        self.middlewares: list['Middleware'] = []

    def use(self, middleware, scope=None):
        '''
        Register one middleware or a list/tuple of middlewares. Accepts
        Middleware instances as well as plain callables with the signature
        (kwargs, next_request) — plain callables are wrapped automatically.
        The first registered middleware is the outermost.
        If `scope` is provided it is assigned to the middleware (see Middleware.scope).
        Returns the input(s), so it can be used as a decorator either way.
        '''
        items = middleware if isinstance(middleware, (list, tuple)) else [middleware]
        for mw in items:
            if isinstance(mw, Middleware):
                pass
            elif callable(mw):
                mw = _FuncMiddleware(mw)
            else:
                raise TypeError(f'Middleware must be a Middleware instance or callable, got {type(mw)}')
            if scope is not None:
                mw.scope = scope
            self.middlewares.append(mw)
        return middleware

    def _dispatch(self, kwargs: Dict[str, Any], scope: str = '*') -> AsyncGenerator[Response, None]:
        enabled_plugins = kwargs.pop('_enabled_plugins', None)

        def core_dispatch(k: Dict[str, Any]) -> AsyncGenerator[Response, None]:
            k = {kk: vv for kk, vv in k.items() if kk != 'scope'}
            return self._core_request(**k)

        chain: Callable = core_dispatch
        for mw in reversed(self.middlewares):
            if mw.matches(scope):
                # filter by enabled plugins: skip middleware owned by disabled plugins
                if enabled_plugins is not None and hasattr(mw, 'owner') and mw.owner and mw.owner not in enabled_plugins:
                    continue
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
        scope: str = '*',
        _enabled_plugins=None,
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
            'scope': scope,
            '_enabled_plugins': _enabled_plugins,
        }
        async for resp in self._dispatch(kwargs, scope=scope):
            yield resp