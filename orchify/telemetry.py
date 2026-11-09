'''Telemetry for orchify: Prometheus metrics and optional OpenTelemetry spans.

Both pieces are fed from the broker event stream, so enabling them instruments
every run, group and tool call without touching application code::

    from orchify import orchify_broker
    from orchify.telemetry import Metrics, Tracer

    metrics = Metrics().attach(orchify_broker)
    Tracer(service_name='my-app').attach(orchify_broker)

    # expose the scrape endpoint from any web framework
    text = metrics.render()

The metrics renderer implements the Prometheus text exposition format directly,
so no ``prometheus_client`` dependency is required. ``Tracer`` uses the real
OpenTelemetry API when ``opentelemetry-api`` is installed and otherwise records
spans in memory, which keeps local development and tests dependency free.

Run-level numbers come from the broker hooks; request-level numbers (one per
step) come from ``LLMInstrumentation``, which is a normal LLM middleware::

    from orchify.telemetry import LLMInstrumentation

    agent.llm.use(LLMInstrumentation(metrics, tracer))
'''

import contextlib
import re
import threading
import time
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Tuple

from orchify.budget import cost_of, price_for
from orchify.llm.base import Middleware
from orchify.utils import get_logger

logger = get_logger(__name__)

_LABEL_SAFE = re.compile(r'[^A-Za-z0-9_.:-]+')
_MAX_TOOL_LABELS = 64


def _escape(value: Any) -> str:
    '''Escape one Prometheus label value.

    Args:
        value (Any): Raw label value.

    Returns:
        str: ``\\`` and ``"`` escaped, newlines flattened.
    '''
    text = str(value if value is not None else '').replace('\\', '\\\\').replace('"', '\\"')
    return text.replace('\n', ' ')


@lru_cache(maxsize=256)
def _metric_name(name: str) -> str:
    '''Normalise a metric name to ``[a-zA-Z_:][a-zA-Z0-9_:]*``.

    Cached: the same handful of names is normalised on every event, and the
    regex used to cost more than the metric update itself.

    Args:
        name (str): Raw name.

    Returns:
        str: A valid Prometheus metric name.
    '''
    cleaned = re.sub(r'[^A-Za-z0-9_:]', '_', str(name or 'metric'))
    return cleaned if not cleaned[0].isdigit() else f'_{cleaned}'


def _label_key(labels: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    '''Sorted label pairs, the stable half of a metric key.'''
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


class Metrics:
    '''Counters, gauges and durations derived from broker events.

    Attributes:
        prefix (str): Metric name prefix (``'orchify'`` by default).
        counters (Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float]): Every
            counter value, keyed by name and label pairs.
    '''

    def __init__(self, prefix: str = 'orchify'):
        '''Configure the collector.

        Args:
            prefix (str): Prefix for every exported metric name.
        '''
        self.prefix = _metric_name(prefix)
        self._lock = threading.Lock()
        self.counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self.gauges: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self.sums: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self.counts: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], int] = {}
        self._run_started: Dict[str, float] = {}
        self._run_models: Dict[str, str] = {}
        self._tool_labels: set = set()
        self._binding: Any = None
        self.dropped_tool_labels = 0

    # ------------------------------------------------------------------ updates

    # The ``_locked`` helpers assume ``self._lock`` is already held. They exist
    # so one event can update several metrics under a single acquisition instead
    # of taking the lock once per metric.

    def _inc_locked(self, key: tuple, value: float = 1.0) -> None:
        self.counters[key] = self.counters.get(key, 0.0) + float(value)

    def _set_locked(self, key: tuple, value: float) -> None:
        self.gauges[key] = float(value)

    def _bump_locked(self, key: tuple, delta: float) -> None:
        self.gauges[key] = self.gauges.get(key, 0.0) + float(delta)

    def _observe_locked(self, key: tuple, value: float) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + float(value)
        self.counts[key] = self.counts.get(key, 0) + 1

    def inc(self, name: str, value: float = 1.0, **labels: Any) -> None:
        '''Increment a counter.

        Args:
            name (str): Metric name without the prefix.
            value (float): Amount to add.
            **labels (Any): Label pairs.
        '''
        key = (_metric_name(name), _label_key(labels))
        with self._lock:
            self._inc_locked(key, value)

    def set(self, name: str, value: float, **labels: Any) -> None:
        '''Set a gauge.

        Args:
            name (str): Metric name without the prefix.
            value (float): Gauge value.
            **labels (Any): Label pairs.
        '''
        key = (_metric_name(name), _label_key(labels))
        with self._lock:
            self._set_locked(key, value)

    def observe(self, name: str, value: float, **labels: Any) -> None:
        '''Record one observation of a duration or size.

        Args:
            name (str): Metric name without the prefix.
            value (float): Observed value.
            **labels (Any): Label pairs.
        '''
        key = (_metric_name(name), _label_key(labels))
        with self._lock:
            self._observe_locked(key, value)

    def observe_request(self, model: str, seconds: float, *, status: str = 'ok') -> None:
        '''Record one LLM request (count and duration, labelled by model).

        Request-level numbers come from ``LLMInstrumentation`` instead of the
        event stream, because a single run issues one request per step.

        Args:
            model (str): Model that served the request.
            seconds (float): Wall-clock duration of the request.
            status (str): ``'ok'`` or ``'error'``.
        '''
        label = str(model or 'unknown')
        with self._lock:
            self._inc_locked((_metric_name('llm_requests_total'),
                              _label_key({'model': label, 'status': status})))
            self._observe_locked((_metric_name('llm_request_seconds'),
                                  _label_key({'model': label})), float(seconds))

    # ------------------------------------------------------------------- events

    def attach(self, broker: Any = None, *, event_types: str = '*',
               mode: str = 'background') -> 'Metrics':
        '''Subscribe to the broker and start collecting.

        Args:
            broker (Any): Broker to observe; defaults to ``orchify_broker``.
            event_types (str): Event filter.
            mode (str): ``'background'`` (default) keeps the event loop free.

        Returns:
            Metrics: ``self``.
        '''
        if broker is None:
            from orchify.broker import orchify_broker as broker
        self.detach()
        self._binding = broker.register_hook(event_types, self.observe_event, mode=mode)
        return self

    def detach(self) -> None:
        '''Stop collecting; already collected values stay available.'''
        if self._binding is None:
            return
        try:
            self._binding.broker.remove_binding(self._binding)
        except Exception:
            try:
                from orchify.broker import orchify_broker
                orchify_broker.remove_binding(self._binding)
            except Exception:
                pass
        self._binding = None

    def _tool_label(self, tool_name: str) -> str:
        '''Bound the label cardinality of tool names.

        Args:
            tool_name (str): Raw tool name.

        Returns:
            str: The sanitised name, or ``'other'`` past the cap.
        '''
        with self._lock:
            return self._tool_label_locked(tool_name)

    def _tool_label_locked(self, tool_name: str) -> str:
        ''' :meth:`_tool_label` for callers that already hold ``self._lock``.'''
        name = _LABEL_SAFE.sub('_', str(tool_name or 'unknown'))[:64] or 'unknown'
        if name in self._tool_labels:
            return name
        if len(self._tool_labels) >= _MAX_TOOL_LABELS:
            self.dropped_tool_labels += 1
            return 'other'
        self._tool_labels.add(name)
        return name

    def observe_event(self, event: Any) -> None:
        '''Update every metric that one broker event implies.

        One lock acquisition per event, not one per metric.

        Args:
            event (Any): The broker event.
        '''
        event_type = str(getattr(event, 'event_type', '') or '')
        run_id = str(getattr(event, 'run_id', '') or '')
        payload = getattr(event, 'payload', None) or {}
        if not isinstance(payload, dict):
            payload = {}
        active_key = (_metric_name('runs_active'), ())
        with self._lock:
            self._inc_locked((_metric_name('events_total'), _label_key({'type': event_type})))
            if event_type == 'run:start':
                self._inc_locked((_metric_name('runs_started_total'), ()))
                self._bump_locked(active_key, 1.0)
                if run_id:
                    self._run_started[run_id] = time.time()
                    model = str(payload.get('model') or '')
                    if model:
                        self._run_models[run_id] = model
            elif event_type == 'run:finish':
                status = 'stopped' if payload.get('stopped') else 'ok'
                self._inc_locked((_metric_name('runs_finished_total'), _label_key({'status': status})))
                steps = payload.get('total_steps') or 0
                if steps:
                    self._inc_locked((_metric_name('steps_total'), ()), float(steps))
                usage = payload.get('usage') or {}
                model = self._run_models.pop(run_id, '') if run_id else ''
                if isinstance(usage, dict) and usage:
                    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                        value = usage.get(key)
                        if value:
                            self._inc_locked(
                                (_metric_name('llm_tokens_total'), _label_key({'kind': key})),
                                float(value),
                            )
                            if model:
                                self._inc_locked(
                                    (_metric_name('llm_tokens_by_model_total'),
                                     _label_key({'model': model, 'kind': key})),
                                    float(value),
                                )
                    if model:
                        price = price_for(model)
                        if price is not None:
                            cost = cost_of(usage, price)
                            if cost:
                                self._inc_locked(
                                    (_metric_name('llm_cost_usd_total'),
                                     _label_key({'model': model})),
                                    float(cost),
                                )
                if run_id:
                    started = self._run_started.pop(run_id, None)
                    if started is not None:
                        self._observe_locked(
                            (_metric_name('run_duration_seconds'), ()),
                            time.time() - started,
                        )
                self.gauges[active_key] = max(0.0, self.gauges.get(active_key, 0.0) - 1.0)
            elif event_type == 'run:error':
                self._inc_locked((_metric_name('runs_finished_total'), _label_key({'status': 'error'})))
            elif event_type == 'tool:call:finish':
                tool = self._tool_label_locked(getattr(event, 'tool_name', ''))
                self._inc_locked((_metric_name('tool_calls_total'), _label_key({'tool': tool})))
                if payload.get('cached'):
                    self._inc_locked((_metric_name('tool_cache_hits_total'), ()))
            elif event_type == 'tool:call:error':
                tool = self._tool_label_locked(getattr(event, 'tool_name', ''))
                self._inc_locked((_metric_name('tool_errors_total'), _label_key({'tool': tool})))
            elif event_type == 'run:approval:resolved':
                decision = 'approved' if payload.get('approved') else 'denied'
                self._inc_locked((_metric_name('approvals_total'), _label_key({'decision': decision})))
            elif event_type == 'run:budget:exceeded':
                self._inc_locked((_metric_name('budget_exceeded_total'), ()))
            elif event_type == 'run:llm:retry':
                self._inc_locked((_metric_name('llm_retries_total'), ()))

    # ------------------------------------------------------------------ output

    def snapshot(self) -> Dict[str, Any]:
        '''Return the collected values as plain data (handy for JSON).

        Returns:
            Dict[str, Any]: Counters, gauges and observations.
        '''
        with self._lock:
            return {
                'counters': [{'name': name, 'labels': dict(labels), 'value': value}
                             for (name, labels), value in self.counters.items()],
                'gauges': [{'name': name, 'labels': dict(labels), 'value': value}
                           for (name, labels), value in self.gauges.items()],
                'observations': [{'name': name, 'labels': dict(labels),
                                  'sum': self.sums[(name, labels)],
                                  'count': self.counts[(name, labels)]}
                                 for (name, labels) in self.sums],
            }

    def render(self) -> str:
        '''Render every metric in the Prometheus text exposition format.

        Returns:
            str: Text payload for a ``/metrics`` endpoint.
        '''
        lines: List[str] = []
        with self._lock:
            counter_items = sorted(self.counters.items())
            gauge_items = sorted(self.gauges.items())
            sum_items = sorted(self.sums.items())
            count_items = dict(self.counts)

        for (name, labels), value in counter_items:
            full = f'{self.prefix}_{name}'
            lines.append(f'# TYPE {full} counter')
            lines.append(f'{full}{_render_labels(labels)} {_render_value(value)}')
        for (name, labels), value in gauge_items:
            full = f'{self.prefix}_{name}'
            lines.append(f'# TYPE {full} gauge')
            lines.append(f'{full}{_render_labels(labels)} {_render_value(value)}')
        for (name, labels), value in sum_items:
            full = f'{self.prefix}_{name}'
            lines.append(f'# TYPE {full} summary')
            lines.append(f'{full}_sum{_render_labels(labels)} {_render_value(value)}')
            lines.append(f'{full}_count{_render_labels(labels)} '
                         f'{count_items.get((name, labels), 0)}')
        return '\n'.join(lines) + ('\n' if lines else '')


def _render_labels(labels: Tuple[Tuple[str, str], ...]) -> str:
    '''Render Prometheus label pairs.

    Args:
        labels (Tuple[Tuple[str, str], ...]): Sorted ``(key, value)`` pairs.

    Returns:
        str: ``{k="v",...}`` or an empty string.
    '''
    if not labels:
        return ''
    body = ','.join(f'{key}="{_escape(value)}"' for key, value in labels)
    return '{' + body + '}'


def _render_value(value: float) -> str:
    '''Render a numeric sample value.

    Args:
        value (float): The value.

    Returns:
        str: Integers without a decimal point, floats with up to six decimals.
    '''
    if float(value).is_integer():
        return str(int(value))
    return f'{float(value):.6f}'


class Tracer:
    '''Run-level tracing, exported to OpenTelemetry when it is installed.

    Attributes:
        service_name (str): Service name reported to the tracer.
        spans (List[Dict[str, Any]]): Finished spans (most recent last) when the
            OpenTelemetry API is not installed, so traces stay inspectable.
    '''

    def __init__(self, service_name: str = 'orchify'):
        '''Configure the tracer.

        Args:
            service_name (str): Service name for spans.
        '''
        self.service_name = service_name
        self.spans: List[Dict[str, Any]] = []
        self._active: Dict[str, Any] = {}
        self._binding: Any = None
        self._otel_tracer = None
        try:  # pragma: no cover - depends on the environment
            from opentelemetry import trace as otel_trace
            self._otel_tracer = otel_trace.get_tracer(service_name)
        except Exception:
            self._otel_tracer = None

    @property
    def backend(self) -> str:
        '''Name of the span exporter in use.

        Returns:
            str: ``'opentelemetry'`` or ``'memory'``.
        '''
        return 'opentelemetry' if self._otel_tracer is not None else 'memory'

    def attach(self, broker: Any = None, *, event_types: str = '*',
               mode: str = 'background') -> 'Tracer':
        '''Subscribe to the broker and trace runs.

        Args:
            broker (Any): Broker to observe; defaults to ``orchify_broker``.
            event_types (str): Event filter.
            mode (str): Hook mode (``'background'`` by default).

        Returns:
            Tracer: ``self``.
        '''
        if broker is None:
            from orchify.broker import orchify_broker as broker
        self.detach()
        self._binding = broker.register_hook(event_types, self.observe_event, mode=mode)
        return self

    def detach(self) -> None:
        '''Stop tracing and close any span still open.'''
        for run_id in list(self._active):
            self.end_run(run_id, 'detached')
        if self._binding is None:
            return
        try:
            self._binding.broker.remove_binding(self._binding)
        except Exception:
            try:
                from orchify.broker import orchify_broker
                orchify_broker.remove_binding(self._binding)
            except Exception:
                pass
        self._binding = None

    def open_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
        '''Open a span that outlives a single block (e.g. an async stream).

        Args:
            name (str): Span name.
            attributes (Optional[Dict[str, Any]]): Initial attributes.

        Returns:
            Any: An OpenTelemetry span when available, else a dict handle to pass
            to :meth:`close_span`.
        '''
        attributes = dict(attributes or {})
        if self._otel_tracer is not None:  # pragma: no cover - depends on the environment
            try:
                span = self._otel_tracer.start_span(name)
                for key, value in attributes.items():
                    span.set_attribute(key, value)
                return span
            except Exception:
                pass
        return {'name': name, 'started_at': time.time(), 'attributes': attributes}

    def close_span(self, span: Any, status: str = 'ok',
                   attributes: Optional[Dict[str, Any]] = None) -> None:
        '''Close a span opened with :meth:`open_span`.

        Args:
            span (Any): Handle returned by :meth:`open_span`; None is ignored.
            status (str): Span status recorded as ``orchify.status``.
            attributes (Optional[Dict[str, Any]]): Extra attributes.
        '''
        if span is None:
            return
        attributes = dict(attributes or {})
        if not isinstance(span, dict):  # pragma: no cover - depends on the environment
            try:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
                span.set_attribute('orchify.status', status)
                span.end()
                return
            except Exception:
                pass
        started = span.get('started_at', time.time()) if isinstance(span, dict) else time.time()
        self.spans.append({
            'name': span.get('name', 'span') if isinstance(span, dict) else 'span',
            'status': status,
            'service': self.service_name,
            'started_at': started,
            'duration': max(0.0, time.time() - started),
            'attributes': {**(span.get('attributes') if isinstance(span, dict) else {}),
                           **attributes},
        })
        if len(self.spans) > 1000:
            del self.spans[:200]

    def start_run(self, run_id: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        '''Open the span for one run.

        Args:
            run_id (str): Run identifier.
            attributes (Optional[Dict[str, Any]]): Span attributes.
        '''
        attributes = dict(attributes or {})
        if self._otel_tracer is not None:  # pragma: no cover - depends on the environment
            try:
                span = self._otel_tracer.start_span(f'run {run_id}')
                for key, value in attributes.items():
                    span.set_attribute(key, value)
                self._active[run_id] = span
                return
            except Exception:
                pass
        self._active[run_id] = {'started_at': time.time(), 'attributes': attributes}

    def end_run(self, run_id: str, status: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        '''Close the span for one run.

        Args:
            run_id (str): Run identifier.
            status (str): ``'ok'``, ``'error'``, ``'stopped'`` ...
            attributes (Optional[Dict[str, Any]]): Extra span attributes.
        '''
        span = self._active.pop(run_id, None)
        if span is None:
            return
        attributes = dict(attributes or {})
        if self._otel_tracer is not None and not isinstance(span, dict):  # pragma: no cover
            try:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
                span.set_attribute('orchify.status', status)
                span.end()
                return
            except Exception:
                pass
        record = {
            'name': f'run {run_id}',
            'status': status,
            'service': self.service_name,
            'started_at': span.get('started_at', time.time()) if isinstance(span, dict) else 0.0,
            'duration': time.time() - span.get('started_at', time.time())
            if isinstance(span, dict) else 0.0,
            'attributes': {**(span.get('attributes') if isinstance(span, dict) else {}),
                           **attributes},
        }
        self.spans.append(record)
        if len(self.spans) > 1000:
            del self.spans[:200]

    def observe_event(self, event: Any) -> None:
        '''React to one broker event (start or finish a run span).

        Args:
            event (Any): The broker event.
        '''
        event_type = str(getattr(event, 'event_type', '') or '')
        run_id = str(getattr(event, 'run_id', '') or '')
        if not run_id:
            return
        payload = getattr(event, 'payload', None) or {}
        if event_type == 'run:start':
            self.start_run(run_id, {
                'orchify.agent': str(getattr(event, 'agent_name', '') or ''),
                'orchify.task': str(payload.get('task_name') or ''),
                'orchify.model': str(payload.get('model') or ''),
            })
        elif event_type == 'run:finish':
            usage = payload.get('usage') or {}
            self.end_run(run_id, 'stopped' if payload.get('stopped') else 'ok', {
                'orchify.steps': payload.get('total_steps') or 0,
                'orchify.tokens': (usage or {}).get('total_tokens', 0),
            })
        elif event_type == 'run:error':
            self.end_run(run_id, 'error', {'orchify.error': str(payload.get('error') or '')[:500]})

    @contextlib.contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        '''Trace an arbitrary block of work.

        Args:
            name (str): Span name.
            **attributes (Any): Span attributes.

        Yields:
            None: The block runs inside the span.
        '''
        started = time.time()
        span = None
        if self._otel_tracer is not None:  # pragma: no cover - depends on the environment
            try:
                span = self._otel_tracer.start_span(name)
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            except Exception:
                span = None
        try:
            yield
        finally:
            if span is not None:  # pragma: no cover
                try:
                    span.end()
                except Exception:
                    pass
            else:
                self.spans.append({
                    'name': name,
                    'status': 'ok',
                    'service': self.service_name,
                    'started_at': started,
                    'duration': time.time() - started,
                    'attributes': dict(attributes),
                })


class LLMInstrumentation(Middleware):
    '''Record one span and timing metrics per LLM request.

    Register it as a middleware on any ``LLMInterface``::

        from orchify.telemetry import LLMInstrumentation

        agent.llm.use(LLMInstrumentation(metrics, tracer))

    Run-level counters come from the broker event stream; this middleware adds
    the per-request layer (``llm_requests_total`` and ``llm_request_seconds``)
    plus an ``llm.request`` span carrying the model, message count and token
    totals of that call.

    Attributes:
        metrics (Optional[Metrics]): Collector to update.
        tracer (Optional[Tracer]): Tracer that owns the spans.
    '''

    def __init__(self, metrics: Optional[Metrics] = None, tracer: Optional[Tracer] = None,
                 name: str = 'llm.request', scope: Any = None):
        '''Configure the middleware.

        Args:
            metrics (Optional[Metrics]): Collector to update; may be None.
            tracer (Optional[Tracer]): Tracer to record spans on; may be None.
            name (str): Span / middleware name.
            scope (Any): Agent scope filter, as for any middleware.
        '''
        super().__init__(name=name, scope=scope)
        self.metrics = metrics
        self.tracer = tracer

    async def __call__(self, kwargs: Dict[str, Any], next_request) -> AsyncGenerator[Any, None]:
        '''Wrap one LLM request.

        Args:
            kwargs (Dict[str, Any]): Request kwargs.
            next_request (Any): Callable returning the inner Response stream.

        Yields:
            Response: Every response of the inner stream, unchanged.
        '''
        model = str(kwargs.get('model') or 'unknown')
        span = None
        if self.tracer is not None:
            span = self.tracer.open_span(self.name, {
                'orchify.model': model,
                'orchify.messages': len(kwargs.get('messages') or []),
                'orchify.tools': len(kwargs.get('tools') or []),
                'orchify.run_id': str(kwargs.get('run_id') or ''),
            })
        started = time.perf_counter()
        status = 'ok'
        error = ''
        final = None
        try:
            async for response in next_request(kwargs):
                if getattr(response, 'final_status', None) is not None:
                    final = response.final_status
                yield response
        except Exception as exc:
            status = 'error'
            error = f'{type(exc).__name__}: {exc}'[:300]
            raise
        finally:
            seconds = time.perf_counter() - started
            attributes: Dict[str, Any] = {
                'orchify.duration': round(seconds, 6),
                'orchify.status': status,
            }
            if final is not None:
                attributes.update({
                    'orchify.tokens': int(getattr(final, 'total_tokens', 0) or 0),
                    'orchify.prompt_tokens': int(getattr(final, 'prompt_tokens', 0) or 0),
                    'orchify.completion_tokens': int(getattr(final, 'completion_tokens', 0) or 0),
                    'orchify.finish_reason': str(getattr(final, 'finish_reason', '') or ''),
                })
            if error:
                attributes['orchify.error'] = error
            if self.tracer is not None:
                self.tracer.close_span(span, status, attributes)
            if self.metrics is not None:
                self.metrics.observe_request(model, seconds, status=status)


def instrument_llm(llm: Any, metrics: Optional[Metrics] = None,
                   tracer: Optional[Tracer] = None, *, scope: Any = None) -> LLMInstrumentation:
    '''Install request-level instrumentation on an LLM.

    Args:
        llm (Any): LLM instance exposing ``use(middleware, scope=...)``.
        metrics (Optional[Metrics]): Collector to update.
        tracer (Optional[Tracer]): Tracer to record spans on.
        scope (Any): Agent scope filter.

    Returns:
        LLMInstrumentation: The installed middleware.
    '''
    middleware = LLMInstrumentation(metrics, tracer, scope=scope)
    llm.use(middleware, scope=scope)
    return middleware


def setup_telemetry(broker: Any = None, *, prefix: str = 'orchify',
                    service_name: str = 'orchify',
                    metrics: Optional[Metrics] = None,
                    tracer: Optional[Tracer] = None) -> Tuple[Metrics, Tracer]:
    '''Attach a metrics collector and a tracer to the broker.

    Args:
        broker (Any): Broker to observe; defaults to ``orchify_broker``.
        prefix (str): Metric name prefix.
        service_name (str): Service name for spans.
        metrics (Optional[Metrics]): Existing collector to reuse.
        tracer (Optional[Tracer]): Existing tracer to reuse.

    Returns:
        Tuple[Metrics, Tracer]: The collector and tracer in use.
    '''
    metrics = metrics or Metrics(prefix)
    tracer = tracer or Tracer(service_name)
    metrics.attach(broker)
    tracer.attach(broker)
    logger.info('telemetry: metrics=%s tracer=%s', prefix, tracer.backend)
    return metrics, tracer
