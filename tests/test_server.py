"""The FastAPI application: auth, runs, polling, SSE, quotas, metrics.

FastAPI is an optional extra, so the whole module is skipped when it is absent.
"""

import threading
import time

import pytest

pytest.importorskip('fastapi')

from fastapi.testclient import TestClient  # noqa: E402

from orchify import Agent  # noqa: E402
from orchify.server import (  # noqa: E402
    EventLog,
    Quota,
    create_app,
    norm_id,
    url_id,
    with_url_id,
)
from orchify.telemetry import Metrics  # noqa: E402
from orchify.testing import FakeLLM, final, wait_until  # noqa: E402


def _app(**kwargs):
    '''Build an app whose sessions answer ``'ok'`` immediately.

    Args:
        **kwargs (Any): Forwarded to :func:`create_app`.

    Returns:
        Any: The FastAPI application.
    '''
    kwargs.setdefault('agent_factory',
                      lambda session: Agent(name=session, llm=FakeLLM([[final('ok')]])))
    return create_app(**kwargs)


def _start(client, text='hello', headers=None):
    '''Start a run and return its ``run_id``.

    Args:
        client (Any): Test client.
        text (str): User input.
        headers (dict): Optional request headers.

    Returns:
        str: The run id.
    '''
    response = client.post('/api/chat', json={'text': text}, headers=headers or {})
    assert response.status_code == 200, response.text
    return response.json()['run_id']


def _finished(client, run_id):
    '''Poll a run until it ends.

    Args:
        client (Any): Test client.
        run_id (str): The run id.

    Returns:
        bool: Whether the run reached a terminal status.
    '''
    def check():
        record = client.get(f'/api/runs/{url_id(run_id)}').json()
        return record.get('status') in ('done', 'stopped', 'error')
    return wait_until(check, timeout=10.0)


def test_ids_round_trip():
    '''``#`` and ``!`` map to each other and records carry the URL form.'''
    assert url_id('demo#abc12345') == 'demo!abc12345'
    assert norm_id('demo!abc12345') == 'demo#abc12345'
    assert with_url_id({'run_id': 'demo#abc'})['run_id_url'] == 'demo!abc'


def test_open_mode_health():
    '''Without keys every endpoint is public and health describes the server.'''
    with TestClient(_app()) as client:
        body = client.get('/api/health').json()
        assert body['ok'] is True
        assert body['auth'] == 'open'
        assert body['sessions'] == 0
        assert body['events']['events_kept'] == 0


def test_api_key_protects_every_route():
    '''A configured key is required on data routes and accepted in both headers.'''
    with TestClient(_app(api_keys=['secret'])) as client:
        assert client.get('/api/health').status_code == 200
        assert client.get('/api/runs').status_code == 401
        assert client.get('/api/runs', headers={'Authorization': 'Bearer wrong'}).status_code == 401
        assert client.get('/api/runs',
                          headers={'X-API-Key': 'secret'}).status_code == 200
        assert client.get('/api/runs',
                          headers={'Authorization': 'Bearer secret'}).status_code == 200
        assert client.post('/api/chat', json={'text': 'hi'}).status_code == 401
        assert client.post('/api/chat', json={'text': 'hi'},
                           headers={'X-API-Key': 'secret'}).status_code == 200


def test_chat_records_cost_fields_and_events():
    '''A run is pollable, ends with ``run:finish`` and exposes usage/cost.'''
    with TestClient(_app()) as client:
        run_id = _start(client)
        assert _finished(client, run_id)
        record = client.get(f'/api/runs/{url_id(run_id)}').json()
        assert record['answer'] == 'ok'
        assert record['tokens'] == 0
        assert record['cost_usd'] == 0.0
        assert record['model']  # the run context carries the model name

        events = client.get(f'/api/runs/{url_id(run_id)}/events').json()
        types = [item['type'] for item in events['items']]
        assert 'run:start' in types
        assert 'run:finish' in types
        assert events['finished'] is True
        assert events['last_seq'] >= len(events['items'])

        tail = client.get(f"/api/runs/{url_id(run_id)}/events?since={events['last_seq']}").json()
        assert tail['items'] == []


def test_unknown_run_is_404():
    '''Unknown run ids are rejected on both read and control routes.'''
    with TestClient(_app()) as client:
        assert client.get('/api/runs/nope!1').status_code == 404
        assert client.get('/api/runs/nope!1/events').json()['items'] == []
        assert client.post('/api/runs/nope!1/control/stop').status_code == 404
        assert client.post('/api/runs/nope!1/control/explode').status_code == 400


def test_sse_stream_replays_and_finishes():
    '''The SSE endpoint sends buffered events and a terminal ``done`` event.'''
    with TestClient(_app()) as client:
        run_id = _start(client)
        assert _finished(client, run_id)
        lines = []
        with client.stream('GET',
                           f'/api/runs/{url_id(run_id)}/events/stream') as response:
            assert response.status_code == 200
            assert response.headers['content-type'].startswith('text/event-stream')
            for line in response.iter_lines():
                if not line:
                    continue
                lines.append(line)
                if line.startswith('event: done'):
                    break
        payloads = [line[6:] for line in lines if line.startswith('data: ')]
        assert payloads, lines
        assert any('run:finish' in payload for payload in payloads)
        assert '"status"' in payloads[-1]


def test_stop_control_ends_the_run():
    '''``POST /api/runs/{id}/control/stop`` stops a streaming run.'''
    app = _app(agent_factory=lambda session: Agent(
        name=session,
        llm=FakeLLM([[{'content': 'think', 'delay': 0.05},
                      {'delay': 0.4},
                      final('too late')]]),
    ))
    with TestClient(app) as client:
        run_id = _start(client)
        assert wait_until(lambda: client.get(f'/api/runs/{url_id(run_id)}')
                          .json().get('status') == 'running', timeout=10.0)
        body = client.post(f'/api/runs/{url_id(run_id)}/control/stop').json()
        assert body['ok'] is True
        assert body['accepted'] is True
        assert _finished(client, run_id)
        record = client.get(f'/api/runs/{url_id(run_id)}').json()
        assert record['status'] == 'stopped'


def test_quota_rejects_a_second_run():
    '''A per-session window quota answers 429 instead of queueing.'''
    with TestClient(_app(quota=Quota(max_per_window=1, window=60))) as client:
        _start(client)
        second = client.post('/api/chat', json={'text': 'again'})
        assert second.status_code == 429
        assert 'exceeded' in second.json()['detail']


def test_quota_charges_tokens_and_cost():
    '''Usage accumulates per session and blocks the next run once over a cap.'''
    quota = Quota(max_tokens=100, max_cost=1.0)
    quota.acquire('s1')
    quota.charge('s1', tokens=60, cost=0.5)
    assert quota.check('s1') is None
    quota.charge('s1', tokens=50, cost=0.6)
    assert 'tokens' in quota.check('s1')
    usage = quota.usage('s1')
    assert usage['tokens'] == 110 and usage['runs'] == 1
    assert usage['cost'] == pytest.approx(1.1)
    assert quota.snapshot()['max_tokens'] == 100
    quota.reset('s1')
    assert quota.check('s1') is None


def test_token_quota_blocks_the_next_run():
    '''A finished run's tokens are charged to its session by the app hook.'''

    def factory(session):
        plan = [{'final': {'content': 'ok', 'reasoning': '', 'tool_calls': [],
                           'prompt_tokens': 100, 'completion_tokens': 20,
                           'total_tokens': 120}}]
        return Agent(name=session, llm=FakeLLM([plan, plan]))

    quota = Quota(max_tokens=50)
    with TestClient(_app(agent_factory=factory, quota=quota)) as client:
        run_id = _start(client)
        assert _finished(client, run_id)
        from orchify import orchify_broker
        orchify_broker.drain_hooks(timeout=5.0)
        second = client.post('/api/chat', json={'text': 'again'})
        assert second.status_code == 429
        assert 'tokens' in second.json()['detail']
        assert quota.usage('default')['tokens'] == 120


def test_sessions_endpoint_lists_and_drops():
    '''Sessions are visible and can be forgotten without touching the run.'''
    with TestClient(_app()) as client:
        _start(client, 'hi')
        assert _finished(client, client.get('/api/runs').json()['items'][0]['run_id'])
        sessions = client.get('/api/sessions').json()
        assert sessions['items'] == ['default']
        assert client.delete('/api/sessions/default').json()['dropped'] is True
        assert client.get('/api/sessions').json()['total'] == 0


def test_metrics_endpoint_exposes_counters():
    '''``/metrics`` renders the collector attached by the app lifespan.'''
    metrics = Metrics('orchify')
    with TestClient(_app(metrics=metrics)) as client:
        run_id = _start(client)
        assert _finished(client, run_id)
        assert wait_until(lambda: metrics.counters.get(
            ('runs_finished_total', (('status', 'ok'),)), 0) >= 1, timeout=10.0)
        text = client.get('/metrics').text
        assert '# TYPE orchify_runs_started_total counter' in text
        assert 'orchify_runs_started_total 1' in text
        assert '# TYPE orchify_runs_active gauge' in text


def _logged(event_type, run_id='r1', **kwargs):
    '''Minimal stand-in for a broker event, as ``EventLog`` reads them.'''

    class _Event:
        pass

    event = _Event()
    event.event_type = event_type
    event.run_id = run_id
    event.payload = {}
    event.turn = 1
    event.created_at = time.time()
    for key, value in kwargs.items():
        setattr(event, key, value)
    return event


def test_event_log_wakes_only_the_run_being_watched():
    '''A stream waits on its own run's condition and stops when it finishes.'''
    log = EventLog(per_run=5)
    try:
        assert log.wait_for('r1', 0, timeout=0.05) == ([], False)

        def publish():
            time.sleep(0.05)
            log._on_event(_logged('run:start', run_id='r2'))
            log._on_event(_logged('run:start', run_id='r1'))

        thread = threading.Thread(target=publish, daemon=True)
        thread.start()
        started = time.monotonic()
        items, done = log.wait_for('r1', 0, timeout=5.0)
        elapsed = time.monotonic() - started
        thread.join(2.0)
        assert [item['type'] for item in items] == ['run:start'], items
        assert items[0]['run_id'] == 'r1'
        assert done is False and elapsed < 2.0, elapsed
        assert log.since('r1', items[-1]['seq']) == []

        log._on_event(_logged('run:finish', run_id='r1'))
        assert log.finished('r1') is True
        assert 'r1' not in log._conds, sorted(log._conds)
        assert [item['type'] for item in log.since('r1', items[-1]['seq'])] == ['run:finish']
    finally:
        log.stop()


def test_event_log_drops_noise_and_merges_reasoning():
    '''Token fragments are dropped, reasoning chunks are merged.'''
    class _Event:
        def __init__(self, event_type, **kwargs):
            self.event_type = event_type
            self.run_id = 'r1'
            self.payload = {}
            self.turn = 1
            self.created_at = time.time()
            for key, value in kwargs.items():
                setattr(self, key, value)

    log = EventLog(per_run=5)
    try:
        log._on_event(_Event('agent:reason:step', content='think'))
        log._on_event(_Event('agent:reason:step', content='ing'))
        log._on_event(_Event('agent:answer', content='tok'))
        log._on_event(_Event('tool:assembly:step', content='{'))
        log._on_event(_Event('agent:reason:finish'))
        items = log.since('r1')
        assert [item['type'] for item in items] == ['agent:reason:finish']
        assert items[0]['detail'] == 'thinking'
        assert log.dropped == 2
        assert log.finished('r1') is False
    finally:
        log.stop()
