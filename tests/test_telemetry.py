"""Telemetry: what the Prometheus metrics and the tracer record.

The collector is fed from broker events, so these tests drive a real run with
:class:`~orchify.testing.FakeLLM` and then inspect the counters, the rendered
exposition text, and the spans.
"""

import time

from orchify import Agent, orchify_broker
from orchify.telemetry import Metrics, Tracer, setup_telemetry
from orchify.testing import FakeLLM, final


def _usage_final(content='ok', prompt=100, completion=20):
    '''Build a final step that reports token usage.

    Args:
        content (str): Answer text.
        prompt (int): Prompt tokens.
        completion (int): Completion tokens.

    Returns:
        dict: A :class:`~orchify.testing.FakeLLM` plan step.
    '''
    return {'final': {
        'content': content,
        'reasoning': '',
        'tool_calls': [],
        'prompt_tokens': prompt,
        'completion_tokens': completion,
        'total_tokens': prompt + completion,
    }}


def _wait(predicate, timeout=8.0):
    '''Spin until ``predicate()`` is true.

    Args:
        predicate (callable): Condition to poll.
        timeout (float): Seconds to wait.

    Returns:
        bool: Whether the condition became true.
    '''
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_metrics_count_a_finished_run():
    '''Counters cover events, runs, steps, tokens and durations.'''
    metrics = Metrics('orchify')
    metrics.attach(orchify_broker)
    try:
        agent = Agent(name='metrics-run', llm=FakeLLM([[_usage_final('counted')]]))
        assert agent.ask('measure me') == 'counted'
        assert _wait(lambda: metrics.counters.get(
            ('runs_finished_total', (('status', 'ok'),)), 0) >= 1)

        def counter(name, **labels):
            key = (name, tuple(sorted((str(k), str(v)) for k, v in labels.items())))
            return metrics.counters.get(key, 0)

        assert counter('runs_started_total') == 1
        assert counter('runs_finished_total', status='ok') == 1
        assert counter('steps_total') == 1
        assert counter('llm_tokens_total', kind='prompt_tokens') == 100
        assert counter('llm_tokens_total', kind='completion_tokens') == 20
        assert counter('llm_tokens_total', kind='total_tokens') == 120
        assert counter('events_total', type='run:start') == 1
        assert counter('events_total', type='agent:finish') == 1
        observations = [item for item in metrics.snapshot()['observations']
                        if item['name'] == 'run_duration_seconds']
        assert observations and observations[0]['count'] == 1
        assert metrics.gauges[(('runs_active'), ())] == 0.0
    finally:
        metrics.detach()


def test_render_is_prometheus_text():
    '''The exposition text has TYPE lines, labels and escaped values.'''
    metrics = Metrics('orchify')
    metrics.inc('events_total', 3, type='run:start')
    metrics.inc('tool_calls_total', tool='quote"me')
    metrics.observe('run_duration_seconds', 1.5)
    metrics.observe('run_duration_seconds', 0.5)
    metrics.set('runs_active', 2)

    text = metrics.render()
    assert '# TYPE orchify_events_total counter' in text
    assert 'orchify_events_total{type="run:start"} 3' in text
    assert 'orchify_tool_calls_total{tool="quote\\"me"} 1' in text
    assert '# TYPE orchify_run_duration_seconds summary' in text
    assert 'orchify_run_duration_seconds_sum 2' in text
    assert 'orchify_run_duration_seconds_count 2' in text
    assert 'orchify_runs_active 2' in text


def test_tool_labels_are_bounded():
    '''Only a bounded number of distinct tool labels are exported.'''
    metrics = Metrics()
    for index in range(80):
        metrics.observe_event(_event('tool:call:finish', tool_name=f'tool_{index}'))
    labels = {name for (name, _pairs) in metrics.counters
              if name == 'tool_calls_total'}
    assert labels == {'tool_calls_total'}
    tools = {dict(pairs).get('tool') for (name, pairs) in metrics.counters
             if name == 'tool_calls_total'}
    assert 'other' in tools
    assert len(tools) <= 65
    assert metrics.dropped_tool_labels > 0


class _event:
    '''Minimal broker event stand-in.

    Attributes:
        event_type (str): Event type.
        run_id (str): Run id.
        payload (dict): Payload.
        tool_name (str): Tool name for tool events.
    '''

    def __init__(self, event_type, run_id='run-1', payload=None, tool_name='',
                 agent_name=''):
        '''Build the event.

        Args:
            event_type (str): Event type.
            run_id (str): Run id.
            payload (dict): Payload.
            tool_name (str): Tool name.
            agent_name (str): Agent name.
        '''
        self.event_type = event_type
        self.run_id = run_id
        self.payload = payload or {}
        self.tool_name = tool_name
        self.agent_name = agent_name
        self.turn = 1


def test_tracer_records_run_spans():
    '''A finished run becomes a span with status and token attributes.'''
    tracer = Tracer('tests')
    tracer.attach(orchify_broker)
    try:
        agent = Agent(name='traced-run', llm=FakeLLM([[_usage_final('traced')]]))
        assert agent.ask('trace me') == 'traced'
        assert _wait(lambda: any(s['status'] in ('ok', 'stopped') for s in tracer.spans)
                     or tracer.backend == 'opentelemetry')
        if tracer.backend == 'memory':
            span = [s for s in tracer.spans if s['status'] == 'ok'][-1]
            assert span['attributes']['orchify.tokens'] == 120
            assert span['attributes']['orchify.steps'] == 1
            assert span['duration'] >= 0
    finally:
        tracer.detach()


def test_span_context_manager_records_work():
    '''``Tracer.span`` traces an arbitrary block when no OTel API is present.'''
    tracer = Tracer('tests')
    tracer._otel_tracer = None  # force the dependency-free backend
    assert tracer.backend == 'memory'
    before = len(tracer.spans)
    with tracer.span('custom-work', step=3):
        pass
    assert len(tracer.spans) == before + 1
    assert tracer.spans[-1]['name'] == 'custom-work'
    assert tracer.spans[-1]['attributes'] == {'step': 3}


def test_setup_telemetry_attaches_both():
    '''``setup_telemetry`` returns an attached collector and tracer.'''
    metrics, tracer = setup_telemetry(orchify_broker, prefix='checks', service_name='checks')
    try:
        agent = Agent(name='checks-run', llm=FakeLLM([[_usage_final('done')]]))
        assert agent.ask('check') == 'done'
        assert _wait(lambda: metrics.counters.get(
            ('runs_finished_total', (('status', 'ok'),)), 0) >= 1)
        assert 'checks_runs_started_total' in metrics.render()
        assert tracer.backend in ('memory', 'opentelemetry')
    finally:
        metrics.detach()
        tracer.detach()


def test_tokens_and_cost_are_labelled_by_model():
    '''``run:finish`` exports per-model tokens and the priced cost.'''
    metrics = Metrics('orchify')
    metrics.attach(orchify_broker)
    try:
        agent = Agent(name='priced-run', model='gpt-4o-mini',
                      llm=FakeLLM([[_usage_final('priced')]]))
        assert agent.ask('price me') == 'priced'
        assert _wait(lambda: metrics.counters.get(
            ('runs_finished_total', (('status', 'ok'),)), 0) >= 1)

        def counter(name, **labels):
            key = (name, tuple(sorted((str(k), str(v)) for k, v in labels.items())))
            return metrics.counters.get(key, 0)

        assert counter('llm_tokens_by_model_total', model='gpt-4o-mini',
                       kind='prompt_tokens') == 100
        assert counter('llm_tokens_by_model_total', model='gpt-4o-mini',
                       kind='total_tokens') == 120
        # gpt-4o-mini: 0.15 USD / 1M prompt (cache miss) and 0.60 USD / 1M completion
        expected = (100 * 0.15 + 20 * 0.60) / 1e6
        assert counter('llm_cost_usd_total', model='gpt-4o-mini') == \
            __import__('pytest').approx(expected)
    finally:
        metrics.detach()


def test_instrumentation_spans_each_request():
    '''``LLMInstrumentation`` records a span and request metrics per call.'''
    import asyncio

    from orchify.telemetry import instrument_llm

    metrics = Metrics('req')
    tracer = Tracer('req')
    tracer._otel_tracer = None  # force the dependency-free backend
    llm = FakeLLM([[_usage_final('instrumented')]])
    middleware = instrument_llm(llm, metrics, tracer)
    assert middleware.name == 'llm.request'

    async def consume():
        collected = []
        async for response in llm.request(messages=[{'role': 'user', 'content': 'hi'}],
                                          model='gpt-4o-mini', scope='probe'):
            collected.append(response)
        return collected

    collected = asyncio.run(consume())
    assert collected and collected[-1].is_final

    span = [item for item in tracer.spans if item['name'] == 'llm.request'][-1]
    assert span['status'] == 'ok'
    assert span['attributes']['orchify.model'] == 'gpt-4o-mini'
    assert span['attributes']['orchify.tokens'] == 120
    assert span['attributes']['orchify.prompt_tokens'] == 100
    assert span['attributes']['orchify.completion_tokens'] == 20

    assert metrics.counters[('llm_requests_total',
                             (('model', 'gpt-4o-mini'), ('status', 'ok')))] == 1
    observations = [item for item in metrics.snapshot()['observations']
                    if item['name'] == 'llm_request_seconds']
    assert observations and observations[0]['count'] == 1
    assert observations[0]['labels'] == {'model': 'gpt-4o-mini'}


def test_instrumentation_marks_failed_requests():
    '''An exception closes the span with ``status='error'``.'''
    import asyncio

    from orchify.llm.base import LLMInterface
    from orchify.telemetry import LLMInstrumentation

    class Broken(LLMInterface):
        '''Fails every request.'''

        async def _core_request(self, **kwargs):
            if False:  # pragma: no cover - makes this an async generator
                yield None

    tracer = Tracer('req')
    tracer._otel_tracer = None
    middleware = LLMInstrumentation(metrics=None, tracer=tracer)
    middleware.scope = None

    async def consume():
        async def next_request(_kwargs):
            raise RuntimeError('provider exploded')
            yield None
        async for _ in middleware({'model': 'broken-model'}, next_request):
            pass

    try:
        asyncio.run(consume())
    except RuntimeError as exc:
        assert 'provider exploded' in str(exc)
    span = [item for item in tracer.spans if item['name'] == 'llm.request'][-1]
    assert span['status'] == 'error'
    assert 'provider exploded' in span['attributes']['orchify.error']


def test_errors_are_counted_separately():
    '''``run:error`` increments the error status instead of the ok status.'''
    metrics = Metrics()
    metrics.observe_event(_event('run:finish', payload={'stopped': True, 'usage': {}}))
    metrics.observe_event(_event('run:error'))
    assert metrics.counters[('runs_finished_total', (('status', 'stopped'),))] == 1
    assert metrics.counters[('runs_finished_total', (('status', 'error'),))] == 1
    assert ('runs_finished_total', (('status', 'ok'),)) not in metrics.counters
