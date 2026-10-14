import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchify.env import req_key, req_model, req_url
from orchify.llm.base import (
    Chunk,
    FinalStatus,
    LLMInterface,
    Middleware,
    Response,
    _FuncMiddleware,
    scope_matches,
)


class _Env:
    '''Context manager that sets and restores environment variables.'''

    def __init__(self, **values):
        '''Record the variables to set.

        Args:
            **values: Variable names and their new values; None removes a variable.
        '''
        self.values = values
        self.previous = {}

    def __enter__(self):
        '''Apply the variables and remember the old ones.'''
        for name, value in self.values.items():
            self.previous[name] = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return self

    def __exit__(self, exc_type, exc, tb):
        '''Restore the previous environment.'''
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return False


class RecordingLLM(LLMInterface):
    '''LLM backend that records the kwargs of every core request.

    Attributes:
        reply (str): Answer text streamed by the core request.
        calls (list[dict]): Kwargs of each core request, in order.
    '''

    def __init__(self, reply='answer'):
        '''Create the backend.

        Args:
            reply (str): Answer text to stream.
        '''
        super().__init__()
        self.reply = reply
        self.calls = []

    async def _core_request(self, **kwargs):
        '''Record the request and stream one chunk plus a final status.

        Args:
            **kwargs: Request parameters assembled by the chain.

        Yields:
            Response: A content chunk, then the final response.
        '''
        self.calls.append(kwargs)
        yield Response(current_chunk=Chunk(is_cot=False, content=self.reply))
        yield Response(current_chunk=None, final_status=FinalStatus(content=self.reply, reasoning=''), is_final=True)


class TagMiddleware(Middleware):
    '''Middleware that appends its tag to extra_data['order'].

    Attributes:
        tag (str): Label appended to the order list.
        seen (list[Response]): Responses observed while streaming.
    '''

    def __init__(self, tag='mw', scope=None):
        '''Create the middleware.

        Args:
            tag (str): Label appended to the order list.
            scope (Any): Optional scope filter, see ``Middleware``.
        '''
        super().__init__(name=tag, scope=scope)
        self.tag = tag
        self.seen = []

    async def __call__(self, kwargs, next_request):
        '''Tag the request, then stream the inner responses.

        Args:
            kwargs (dict): Request kwargs handed to the chain.
            next_request (Callable): Callable returning the inner stream.

        Yields:
            Response: Every response produced by the inner stream.
        '''
        extra = kwargs.get('extra_data')
        if not isinstance(extra, dict):
            extra = {}
            kwargs['extra_data'] = extra
        extra.setdefault('order', []).append(self.tag)
        async for resp in next_request(kwargs):
            self.seen.append(resp)
            yield resp


def _drain(llm, **kwargs):
    '''Run one request against the backend and collect its responses.

    Args:
        llm (LLMInterface): Backend to drive.
        **kwargs: Extra ``request`` keyword arguments.

    Returns:
        list[Response]: Every streamed response.
    '''

    async def run():
        return [resp async for resp in llm.request(messages=[{'role': 'user', 'content': 'hi'}], **kwargs)]

    return asyncio.run(run())


def _order(llm):
    '''Read the middleware order recorded in the core request.

    Args:
        llm (RecordingLLM): Backend whose last request is inspected.

    Returns:
        list[str]: Middleware tags in call order.
    '''
    extra = llm.calls[-1].get('extra_data') or {}
    return list(extra.get('order') or [])


def test_request_reaches_the_core_with_defaults():
    '''A bare request forwards its defaults and yields the core stream.'''

    llm = RecordingLLM(reply='hello')
    responses = _drain(llm)
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call['model'] == 'gpt-4o'
    assert call['temperature'] == 0.7
    assert call['thinking'] == 'auto'
    assert call['messages'] == [{'role': 'user', 'content': 'hi'}]
    assert 'scope' not in call
    assert '_enabled_plugins' not in call
    assert llm.middlewares == []
    assert [resp.is_final for resp in responses] == [False, True]
    assert responses[0].current_chunk.content == 'hello'
    assert responses[-1].final_status.content == 'hello'


def test_first_registered_middleware_is_outermost():
    '''Registration order decides the onion order of the chain.'''

    llm = RecordingLLM()
    outer, inner = TagMiddleware('outer'), TagMiddleware('inner')
    llm.use(outer)
    llm.use(inner)
    responses = _drain(llm)
    assert _order(llm) == ['outer', 'inner']
    assert len(outer.seen) == len(responses) == 2
    assert outer.seen[-1].is_final is True
    assert llm.middlewares == [outer, inner]


def test_middleware_mutates_the_outgoing_request():
    '''Middleware kwargs changes reach the core request.'''

    class Tweak(TagMiddleware):
        '''Middleware that rewrites two request parameters.'''

        async def __call__(self, kwargs, next_request):
            kwargs['temperature'] = 0.1
            kwargs['max_tokens'] = 7
            async for resp in next_request(kwargs):
                yield resp

    llm = RecordingLLM()
    llm.use(Tweak('tweak'))
    _drain(llm, temperature=0.9, max_tokens=99)
    assert llm.calls[0]['temperature'] == 0.1
    assert llm.calls[0]['max_tokens'] == 7


def test_use_wraps_callables_and_lists_and_rejects_junk():
    '''use() accepts callables and lists, and rejects everything else.'''

    async def plain(kwargs, next_request):
        async for resp in next_request(kwargs):
            yield resp

    llm = RecordingLLM()
    assert llm.use(plain) is plain
    assert isinstance(llm.middlewares[0], _FuncMiddleware)
    assert llm.middlewares[0].name == 'plain'

    tag = TagMiddleware('tagged')
    assert llm.use([tag, plain]) == [tag, plain]
    assert llm.middlewares[1] is tag
    assert len(llm.middlewares) == 3

    try:
        llm.use(object())
    except TypeError as exc:
        assert 'Middleware must be a Middleware instance or callable' in str(exc)
    else:
        raise AssertionError('non-callable middleware was accepted')
    assert len(llm.middlewares) == 3

    scoped = _FuncMiddleware(plain)
    assert scoped.name == 'plain'
    assert Middleware().name == 'Middleware'
    assert Middleware(name='named').name == 'named'

    responses = _drain(llm)
    assert len(responses) == 2
    assert len(llm.calls) == 1
    assert _order(llm) == ['tagged']


def test_scope_filters_middleware_by_agent_name():
    '''A scoped middleware only wraps matching requests.'''

    llm = RecordingLLM()
    scoped = TagMiddleware('scoped', scope='agent:alpha')
    global_mw = TagMiddleware('global')
    llm.use(scoped)
    llm.use(global_mw)

    _drain(llm, scope='alpha')
    assert _order(llm) == ['scoped', 'global']
    _drain(llm, scope='beta')
    assert _order(llm) == ['global']
    assert scoped.matches('alpha') is True
    assert scoped.matches('beta') is False
    assert global_mw.matches('anything') is True

    llm.use(TagMiddleware('listed'), scope=['agent:beta', 'agent:alpha'])
    _drain(llm, scope='beta')
    assert _order(llm) == ['global', 'listed']
    _drain(llm, scope='gamma')
    assert _order(llm) == ['global']


def test_enabled_plugins_filter_by_owner():
    '''Middleware with an owner is skipped unless that plugin is enabled.'''

    llm = RecordingLLM()
    mw = TagMiddleware('plugin-mw')
    mw.owner = 'plug-a'
    llm.use(mw)
    llm.use(TagMiddleware('app'))

    _drain(llm, _enabled_plugins={'plug-a'})
    assert _order(llm) == ['plugin-mw', 'app']
    assert '_enabled_plugins' not in llm.calls[-1]
    assert 'scope' not in llm.calls[-1]

    _drain(llm, _enabled_plugins={'plug-b'})
    assert _order(llm) == ['app']

    _drain(llm, _enabled_plugins=None)
    assert _order(llm) == ['plugin-mw', 'app']


def test_unowned_middleware_survives_the_plugin_filter():
    '''Middleware without an owner is never filtered out.'''

    llm = RecordingLLM()
    llm.use(TagMiddleware('app'))
    _drain(llm, _enabled_plugins=set())
    assert _order(llm) == ['app']


def test_scope_matches_helper():
    '''scope_matches understands the scope spellings.'''

    assert scope_matches(None, 'alpha') is True
    assert scope_matches('*', 'alpha') is True
    assert scope_matches('agent', 'alpha') is True
    assert scope_matches('agent:alpha', 'alpha') is True
    assert scope_matches('agent:alpha', 'beta') is False
    assert scope_matches('agent:alpha', '') is False
    assert scope_matches(['agent:beta', 'agent:alpha'], 'alpha') is True
    assert scope_matches(('agent:beta',), 'alpha') is False
    assert scope_matches({'agent:alpha'}, 'alpha') is True
    assert scope_matches(lambda name: name == 'alpha', 'alpha') is True
    assert scope_matches(lambda name: name == 'alpha', 'beta') is False

    def boom(name):
        '''Callable scope that always raises.'''
        raise RuntimeError('boom')

    assert scope_matches(boom, 'alpha') is False


def test_base_classes_raise_not_implemented():
    '''The base middleware and the base core request refuse to run.'''

    pending = Middleware()({'messages': []}, next_request=None)
    try:
        asyncio.run(pending)
    except NotImplementedError as exc:
        assert 'must implement __call__' in str(exc)
    else:
        raise AssertionError('base Middleware did not raise')

    pending_core = LLMInterface()._core_request(messages=[])
    try:
        asyncio.run(pending_core)
    except NotImplementedError as exc:
        assert 'must implement _core_request' in str(exc)
    else:
        raise AssertionError('base LLMInterface did not raise')


def test_env_helpers():
    '''Environment helpers honour both key spellings and their fallbacks.'''

    with _Env(API_KEY='primary', OPENAI_API_KEY='secondary', BASE_URL='https://example.test/v1', MODEL_ID='demo-model'):
        assert req_key() == 'primary'
        assert req_url() == 'https://example.test/v1'
        assert req_url('fallback') == 'https://example.test/v1'
        assert req_model() == 'demo-model'
        assert req_model('fallback') == 'demo-model'

    with _Env(API_KEY=None, OPENAI_API_KEY='secondary', BASE_URL=None, MODEL_ID=None):
        assert req_key() == 'secondary'
        assert req_url() == ''
        assert req_url('https://fallback.test/v1') == 'https://fallback.test/v1'
        assert req_model() == ''
        assert req_model('fallback-model') == 'fallback-model'

    with _Env(API_KEY=None, OPENAI_API_KEY=None):
        assert req_key() == ''


def main():
    '''Run every test in this module and report a summary.'''
    tests = [
        test_request_reaches_the_core_with_defaults,
        test_first_registered_middleware_is_outermost,
        test_middleware_mutates_the_outgoing_request,
        test_use_wraps_callables_and_lists_and_rejects_junk,
        test_scope_filters_middleware_by_agent_name,
        test_enabled_plugins_filter_by_owner,
        test_unowned_middleware_survives_the_plugin_filter,
        test_scope_matches_helper,
        test_base_classes_raise_not_implemented,
        test_env_helpers,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            import traceback
            failures.append(fn.__name__)
            print(f'FAIL {fn.__name__}: {exc}')
            traceback.print_exc()
    print()
    if failures:
        print(f'{len(failures)} failed: {failures}')
        return 1
    print(f'all {len(tests)} tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
