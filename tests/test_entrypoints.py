"""Entry point discovery for plugins (``pip install orchify-plugin-<name>``)."""
import json

import pytest

from orchify import cli
from orchify.agent import Agent
from orchify.broker import orchify_broker
from orchify.plugin import Plugin
from orchify.plugin_manager import ENTRY_POINT_GROUP, PluginManager, orchify_plugins
from orchify.testing import FakeLLM, final


class EntryPlugin(Plugin):
    '''Plugin advertised through a fake entry point.'''

    name = 'entry_demo'
    version = '1.0.0'
    description = 'entry point demo plugin'

    def on_load(self):
        self.seen = []

    @Plugin.hook('run:start')
    def on_run_start(self, event):
        self.seen.append(event.run_id)


class FactoryPlugin(Plugin):
    '''Plugin constructed by a factory entry point.'''

    name = 'entry_factory'


class _StubEntryPoint:
    '''Minimal stand-in for ``importlib.metadata.EntryPoint``.'''

    def __init__(self, name, target=None, error=None):
        self.name = name
        self.value = f'orchify_entry_demo:{name}'
        self.group = ENTRY_POINT_GROUP
        self._target = target
        self._error = error
        self.loaded = False

    def load(self):
        self.loaded = True
        if self._error is not None:
            raise self._error
        return self._target


@pytest.fixture
def stub_entry_points(monkeypatch):
    '''Install a fake entry point table and reset the discovery cache.'''
    store = {'entries': []}

    def fake(group):
        return list(store['entries']) if group == ENTRY_POINT_GROUP else []

    monkeypatch.setattr('orchify.plugin_manager._entry_points', fake)
    orchify_plugins._entry_points_loaded.discard(ENTRY_POINT_GROUP)
    yield store
    for name in ('entry_demo', 'entry_factory'):
        if name in orchify_plugins.registered():
            orchify_plugins.unregister(name)
    orchify_plugins._entry_points_loaded.discard(ENTRY_POINT_GROUP)


def test_entry_points_are_listed_without_importing(stub_entry_points):
    entry = _StubEntryPoint('entry_demo', EntryPlugin)
    stub_entry_points['entries'] = [entry]

    listed = orchify_plugins.entry_points()

    assert listed == [{'name': 'entry_demo', 'value': entry.value, 'group': ENTRY_POINT_GROUP}]
    assert entry.loaded is False


def test_entry_point_is_registered_and_loaded(stub_entry_points):
    stub_entry_points['entries'] = [_StubEntryPoint('entry_demo', EntryPlugin)]

    loaded = orchify_plugins.load_entry_points()

    assert [plugin.id for plugin in loaded] == ['entry_demo']
    plugin = orchify_plugins.get('entry_demo')
    assert plugin.loaded is True
    assert 'entry_demo' in orchify_plugins.registered()

    orchify_broker.emit(orchify_broker.event('run:start', run_id='entry-run'))
    assert plugin.seen == ['entry-run']


def test_entry_point_may_point_at_a_factory(stub_entry_points):
    stub_entry_points['entries'] = [_StubEntryPoint('entry_factory', FactoryPlugin)]

    orchify_plugins.load_entry_points()

    assert orchify_plugins.get('entry_factory').loaded is True


def test_broken_entry_point_is_reported_not_raised(stub_entry_points):
    stub_entry_points['entries'] = [_StubEntryPoint('broken', error=ImportError('no such module'))]
    errors = []
    binding = orchify_broker.register_hook(
        'plugin:error', lambda event: errors.append(dict(event.payload))
    )
    try:
        assert orchify_plugins.load_entry_points() == []
    finally:
        orchify_broker.remove_binding(binding)

    assert errors and errors[0]['plugin'] == 'broken'
    assert 'no such module' in errors[0]['error']


def test_agent_enabling_an_id_loads_it_from_entry_points(stub_entry_points):
    stub_entry_points['entries'] = [_StubEntryPoint('entry_demo', EntryPlugin)]
    agent = Agent(name='entry-agent', llm=FakeLLM([[final('ok')]]), plugins=['entry_demo'])

    assert orchify_plugins.get('entry_demo').loaded is True
    assert agent in orchify_plugins.get('entry_demo')._attached_agents

    agent.set_plugins([])
    orchify_plugins.unregister('entry_demo')


def test_ensure_entry_points_runs_only_once(stub_entry_points):
    stub_entry_points['entries'] = [_StubEntryPoint('entry_demo', EntryPlugin)]

    first = orchify_plugins.ensure_entry_points()
    second = orchify_plugins.ensure_entry_points()

    assert first == ['entry_demo']
    assert second == []


def test_cli_plugins_reports_entry_points_and_registry(stub_entry_points, capsys):
    stub_entry_points['entries'] = [_StubEntryPoint('entry_demo', EntryPlugin)]
    orchify_plugins.register(EntryPlugin)
    try:
        assert cli.main(['plugins', '--json']) == cli.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload['group'] == ENTRY_POINT_GROUP
        assert payload['entry_points'][0]['name'] == 'entry_demo'
        assert 'entry_demo' in payload['registered']
    finally:
        orchify_plugins.unregister('entry_demo')


def test_cli_plugins_text_mode_mentions_the_group(stub_entry_points, capsys):
    assert cli.main(['plugins']) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert f'entry point group: {ENTRY_POINT_GROUP}' in out
    assert 'registered plugins: 0' in out


def test_plugin_manager_exposes_the_default_group():
    assert PluginManager().entry_points() == orchify_plugins.entry_points()
