import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify import Agent, Tool
from orchify.broker import orchify_broker
from orchify.event import RuntimeEvent
from orchify.llm.base import Middleware
from orchify.plugin import Plugin
from orchify.plugin_manager import PluginError, PluginManager
from orchify.testing import FakeLLM, Recorder, final

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _watched_only(agent_name):
    '''Scope helper matching a single agent name.'''
    return agent_name == 'watched'


def _event(event_type='run:start', agent_name='probe', payload=None):
    '''Build a synthetic runtime event for hook tests.'''
    return RuntimeEvent(agent_name=agent_name, agent_code='code', turn_id='t1',
                        event_type=event_type, payload=payload or {})


def _agent(name, reply='ok'):
    '''Build a FakeLLM-backed agent.'''
    return Agent(name=name, llm=FakeLLM([[final(reply)]]), system_prompt='sys')


def _raises(exc_type, func, contains=None):
    '''Assert that calling ``func`` raises ``exc_type``.'''
    try:
        func()
    except exc_type as exc:
        if contains is not None:
            assert contains in str(exc), f'{contains!r} not in {exc!r}'
        return exc
    raise AssertionError(f'{exc_type.__name__} was not raised')


def _write(path, text):
    '''Write a UTF-8 text file.'''
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)


def test_declarative_plugin_registers_hooks_and_tools():
    '''Loading a declarative plugin registers its hooks and tools.'''
    seen = []

    class DemoPlugin(Plugin):
        '''A declarative plugin with one hook and one tool.'''

        name = 'demo-decl'
        description = 'declarative demo'
        version = '2.3.4'
        tags = ['demo']

        @Plugin.hook('run:start')
        def on_start(self, event):
            '''Record the event type.'''
            seen.append(event.event_type)

        @Plugin.tool(name='demo-echo')
        def echo(self, text: str) -> str:
            '''Echo the given text.'''
            return text

    manager = PluginManager(broker=orchify_broker)
    recorder = Recorder()
    try:
        instance = manager.register(DemoPlugin)
        assert instance is not None
        loaded = manager.load('demo-decl')
        assert loaded == [instance]
        assert instance.loaded is True
        assert manager.loaded == ['demo-decl']
        assert manager.get('demo-decl') is instance
        assert instance.id == 'demo-decl'

        assert len(instance._bindings) == 1
        binding = instance._bindings[0]
        assert binding.owner == 'demo-decl'
        assert binding.plugin is True
        registered = [binding.event_type] if isinstance(binding.event_type, str) else list(binding.event_type)
        assert registered == ['run:start']
        assert [tool.name for tool in instance._tools] == ['demo-echo']

        orchify_broker.emit(_event('run:start'))
        assert seen == ['run:start']

        load_events = recorder.of('plugin:load')
        assert len(load_events) == 1
        payload = load_events[0].payload
        assert payload['plugin'] == 'demo-decl'
        assert payload['version'] == '2.3.4'
        assert payload['description'] == 'declarative demo'
        assert payload['tags'] == ['demo']
        assert payload['tools'] == ['demo-echo']
        assert payload['hooks'] == ['run:start']
    finally:
        recorder.close()
        manager.unregister('demo-decl')
    assert orchify_broker.remove_hooks('demo-decl') == 0


def test_plugin_scope_filter_limits_hooks():
    '''A scoped plugin only hears events from its own agents.'''
    seen = []

    class ScopedPlugin(Plugin):
        '''A plugin scoped to one agent.'''

        name = 'demo-scoped'
        scope = 'agent:watched'

        @Plugin.hook('run:start')
        def on_start(self, event):
            '''Record the agent name.'''
            seen.append(event.agent_name)

    class CallableScopePlugin(Plugin):
        '''A plugin whose scope is a plain callable.'''

        name = 'demo-callable-scope'
        scope = _watched_only

    manager = PluginManager(broker=orchify_broker)
    try:
        instance = manager.register(ScopedPlugin)
        manager.load('demo-scoped')
        assert instance.matches('watched') is True
        assert instance.matches('other') is False
        assert instance.matches(_agent('watched')) is True
        binding = instance._bindings[0]
        assert callable(binding.match)

        orchify_broker.emit(_event('run:start', agent_name='other'))
        assert seen == []
        orchify_broker.emit(_event('run:start', agent_name='watched'))
        assert seen == ['watched']

        scoped = manager.register(CallableScopePlugin)
        assert scoped.matches('watched') is True
        assert scoped.matches('other') is False
        assert scoped._scope_filter() is not None
    finally:
        manager.unregister('demo-scoped')
        manager.unregister('demo-callable-scope')


def test_imperative_hooks_and_cleanup():
    '''Hooks registered with ``on`` are owned by the plugin and removed on cleanup.'''

    class ImperativePlugin(Plugin):
        '''A plugin that registers hooks imperatively.'''

        name = 'demo-imperative'

    seen = []
    instance = ImperativePlugin(broker=orchify_broker)
    try:
        instance.on('run:start')(lambda event: seen.append('hooked'))
        instance.on('run:start', once=True)(lambda event: seen.append('once'))
        assert len(instance._bindings) == 2

        orchify_broker.emit(_event('run:start'))
        assert seen == ['hooked', 'once']
        orchify_broker.emit(_event('run:start'))
        assert seen == ['hooked', 'once', 'hooked']

        instance.loaded = True
        instance.cleanup()
        assert instance.loaded is False
        assert instance._bindings == []
        orchify_broker.emit(_event('run:start'))
        assert seen == ['hooked', 'once', 'hooked']

        instance.cleanup()
        assert orchify_broker.remove_hooks('demo-imperative') == 0
    finally:
        orchify_broker.remove_hooks('demo-imperative')


def test_plugin_tools_attach_detach_and_type_errors():
    '''Plugin tools follow the plugin manager and can be detached again.'''
    agent = _agent('plug-agent')
    other = _agent('plug-other')

    class ToolPlugin(Plugin):
        '''A plugin that owns one tool bound to an agent.'''

        name = 'demo-tools'

        @Plugin.tool(name='attached-tool', agent=agent)
        def attached(self, text: str) -> str:
            '''Return the given text.'''
            return text

    manager = PluginManager(broker=orchify_broker)
    try:
        instance = manager.register(ToolPlugin)
        manager.load('demo-tools')
        assert 'attached-tool' in agent.tools
        assert instance._attached_agents == {agent}
        assert [tool.name for tool in instance._tools] == ['attached-tool']

        manager.attach('demo-tools', other)
        assert 'attached-tool' in other.tools

        instance.detach_tools()
        assert 'attached-tool' not in agent.tools
        assert 'attached-tool' not in other.tools
        assert instance._attached_agents == set()

        added = instance.add_tool(Tool(func=lambda text: text, name='manual-tool'), agent)
        assert added.name == 'manual-tool'
        assert 'manual-tool' in agent.tools
        _raises(TypeError, lambda: instance.add_tool('not a tool'), contains='requires a Tool')
    finally:
        manager.unregister('demo-tools')


def test_plugin_helpers_event_feedback_approvals_and_logging():
    '''The plugin helper surface delegates to the broker.'''

    class HelperPlugin(Plugin):
        '''A plugin with no declarative hooks.'''

        name = 'demo-helpers'

    instance = HelperPlugin(broker=orchify_broker)
    try:
        assert instance.state == {}
        instance.state['counter'] = 1

        event = instance.event('plugin:custom', payload={'n': 1})
        assert event.event_type == 'plugin:custom'
        assert event.source == 'plugin'
        orchify_broker.emit(event)

        assert instance.feedback('control:stop', run_id='demo#missing') is False

        first = orchify_broker.request_approval('demo#helpers', tool_name='risky')
        second = orchify_broker.request_approval('demo#helpers', tool_name='risky-two')
        try:
            pending = instance.pending_approvals()
            assert sorted(item.approval_id for item in pending) == sorted(
                [first.approval_id, second.approval_id])
            assert instance.approve(first.approval_id, reason='fine') is True
            assert instance.approve(first.approval_id) is False
            assert instance.reject(second.approval_id) is True
            assert instance.reject(second.approval_id) is False
        finally:
            orchify_broker.release_approval(first.approval_id)
            orchify_broker.release_approval(second.approval_id)

        instance.log('hello from the plugin')
    finally:
        orchify_broker.remove_hooks('demo-helpers')


def test_plugin_middleware_targets_llm_and_agents():
    '''Middleware installs on the shared LLM or on attached agents.'''

    class MwPlugin(Plugin):
        '''A plugin that installs middleware.'''

        name = 'demo-mw'

    async def plain(kwargs, next_request):
        '''A plain async callable used as middleware.'''
        async for resp in next_request(kwargs):
            yield resp

    llm = FakeLLM([[final('x')]])
    instance = MwPlugin(broker=orchify_broker)
    middleware = Middleware(name='mw-obj')

    instance.middleware(middleware, llm=llm)
    assert middleware.owner == 'demo-mw'
    assert llm.middlewares == [middleware]

    instance.middleware(plain, llm=llm)
    assert len(llm.middlewares) == 2
    assert llm.middlewares[1].name == 'plain'

    agent = _agent('mw-agent')
    instance.add_tool(Tool(func=lambda: '', name='mw-tool'), agent)
    instance.middleware(Middleware(name='on-agent'))
    assert [mw.name for mw in agent.llm.middlewares] == ['on-agent']

    logged = []
    detached = MwPlugin(broker=orchify_broker)
    detached.log = logged.append
    detached.middleware(Middleware(name='deferred'))
    assert logged and 'deferred' in logged[-1]


def test_plugin_get_config_reads_the_run_context():
    '''Plugin configuration is read from the active run context.'''

    class ConfigPlugin(Plugin):
        '''A plugin scoped to the configured agent.'''

        name = 'demo-config'
        scope = 'agent:cfg-agent'

    class OtherPlugin(Plugin):
        '''A plugin scoped to an agent that never runs.'''

        name = 'demo-config-other'
        scope = 'agent:nobody'

    instance = ConfigPlugin(broker=orchify_broker)
    run_id = f'plugtest#cfg_{os.getpid()}'
    plain_run = f'plugtest#plain_{os.getpid()}'
    orchify_broker.start_task(run_id, None, context={
        'run_id': run_id,
        'agent_name': 'cfg-agent',
        'plugin_config': {'demo-config': {'limit': 3, 'name': 'demo'}},
    })
    try:
        assert instance._find_active_run() == run_id
        assert instance.get_config() == {'limit': 3, 'name': 'demo'}
        assert instance.get_config('limit') == 3
        assert instance.get_config('missing') is None
        assert instance.get_config(run_id='ghost#1') is None
        assert instance.get_config(task_name=run_id) == {'limit': 3, 'name': 'demo'}
    finally:
        orchify_broker.finish_task(run_id)

    orchify_broker.start_task(plain_run, None, context={
        'run_id': plain_run,
        'agent_name': 'cfg-agent',
    })
    try:
        assert instance.get_config() is None
        assert instance.get_config(run_id=plain_run) is None
    finally:
        orchify_broker.finish_task(plain_run)

    other = OtherPlugin(broker=orchify_broker)
    assert other.get_config() is None
    assert instance.get_config() is None


def test_async_lifecycle_and_load_failures():
    '''Async lifecycle methods run, and a failing load is reported.'''
    async_seen = []

    class AsyncPlugin(Plugin):
        '''A plugin with an async lifecycle hook.'''

        name = 'demo-async'

        async def on_load(self):
            '''Record the async load.'''
            async_seen.append('loaded')

    class BrokenPlugin(Plugin):
        '''A plugin whose load raises.'''

        name = 'demo-broken'

        def on_load(self):
            '''Fail on purpose.'''
            raise RuntimeError('on_load exploded')

    manager = PluginManager(broker=orchify_broker)
    recorder = Recorder()
    try:
        manager.register(AsyncPlugin)
        loaded = manager.load('demo-async')
        assert [inst.name for inst in loaded] == ['demo-async']
        assert async_seen == ['loaded']

        manager.register(BrokenPlugin)
        _raises(PluginError, lambda: manager.load('demo-broken'),
                contains="Failed to load plugin 'demo-broken'")
        errors = recorder.of('plugin:error')
        assert [event.payload['action'] for event in errors] == ['load']
        assert 'on_load exploded' in errors[0].payload['error']
        assert manager.plugins['demo-broken'].loaded is False
        assert orchify_broker.remove_hooks('demo-broken') == 0
    finally:
        recorder.close()
        manager.unregister('demo-async')
        manager.unregister('demo-broken')


def test_manager_registration_errors_and_dependency_order():
    '''Registration, dependency order and dependency errors.'''

    class BasePlugin(Plugin):
        '''A plugin with no dependencies.'''

        name = 'demo-base'

    class LeafPlugin(Plugin):
        '''A plugin depending on the base plugin.'''

        name = 'demo-leaf'
        dependencies = ['demo-base']

    class CycleA(Plugin):
        '''Half of a dependency cycle.'''

        name = 'demo-cycle-a'
        dependencies = ['demo-cycle-b']

    class CycleB(Plugin):
        '''The other half of the cycle.'''

        name = 'demo-cycle-b'
        dependencies = ['demo-cycle-a']

    class UnknownDep(Plugin):
        '''A plugin with a missing dependency.'''

        name = 'demo-unknown-dep'
        dependencies = ['demo-missing']

    class DisabledPlugin(Plugin):
        '''A plugin registered while disabled.'''

        name = 'demo-disabled'

    manager = PluginManager(broker=orchify_broker)
    names = ['demo-base', 'demo-leaf', 'demo-cycle-a', 'demo-cycle-b',
             'demo-unknown-dep', 'demo-disabled']
    try:
        _raises(PluginError, lambda: manager.register(object), contains='Invalid plugin registration')
        assert manager.register(DisabledPlugin, enabled=False) is None
        assert 'demo-disabled' not in manager.plugins
        manager.unregister('demo-disabled')
        assert manager.register(DisabledPlugin) is not None
        manager.unregister('demo-disabled')

        manager.register(BasePlugin)
        _raises(PluginError, lambda: manager.register(BasePlugin), contains='already registered')
        _raises(PluginError, lambda: manager.get('demo-none'), contains='is not registered')

        manager.register(LeafPlugin)
        loaded = manager.load('demo-leaf')
        assert [inst.name for inst in loaded] == ['demo-base', 'demo-leaf']
        assert manager.loaded == ['demo-base', 'demo-leaf']
        assert manager.unload('demo-leaf') == ['demo-leaf']
        assert manager.loaded == ['demo-base']
        manager.load('demo-leaf')
        assert manager.unload('demo-base') == ['demo-leaf', 'demo-base']
        assert manager.loaded == []

        manager.register(CycleA)
        manager.register(CycleB)
        _raises(PluginError, lambda: manager.loaded, contains='Circular plugin dependency')

        manager.register(UnknownDep)
        clean = PluginManager(broker=orchify_broker)
        try:
            clean.register(UnknownDep)
            _raises(PluginError, lambda: clean.loaded, contains='depends on unknown plugin')
        finally:
            clean._registry.pop('demo-unknown-dep', None)
            clean._order_dirty = True
    finally:
        for name in names:
            manager.plugins.pop(name, None)
            manager._registry.pop(name, None)
            orchify_broker.remove_hooks(name)
        manager._order_dirty = True


def test_manager_load_all_unload_all_and_configure():
    '''load_all/unload_all, the registry copy and configure().'''

    class FirstPlugin(Plugin):
        '''A plugin without tools.'''

        name = 'demo-first'

    class SecondPlugin(Plugin):
        '''A plugin with one tool.'''

        name = 'demo-second'

        @Plugin.tool(name='second-tool')
        def second(self, text: str) -> str:
            '''Return the given text.'''
            return text

    manager = PluginManager(broker=orchify_broker)
    llm = FakeLLM([[final('x')]])
    agent = _agent('demo-second-agent')
    try:
        manager.register(FirstPlugin)
        manager.register(SecondPlugin)

        registry = manager.registered()
        registry.pop('demo-first')
        assert 'demo-first' in manager.plugins

        assert [inst.name for inst in manager.load_all()] == ['demo-first', 'demo-second']
        assert manager.loaded == ['demo-first', 'demo-second']
        assert manager.plugins['demo-first']._llm is None

        manager.attach('demo-second', agent)
        assert 'second-tool' in agent.tools

        manager.configure(llm=llm)
        assert manager.llm is llm
        assert manager.plugins['demo-first']._llm is llm

        assert manager.unload_all() == ['demo-second', 'demo-first']
        assert manager.loaded == []
        assert manager.unload() == []
        _raises(PluginError, lambda: manager.unload('demo-none'), contains='is not registered')
        _raises(PluginError, lambda: manager.attach('demo-none', agent), contains='is not registered')
    finally:
        manager.configure(llm=None)
        manager.unregister('demo-first')
        manager.unregister('demo-second')


def test_manager_discovers_and_loads_plugin_files():
    '''A directory of plugin modules is discovered, imported and loaded.'''
    pid = os.getpid()
    base = os.path.join(ROOT, f'plugdir{pid}')
    manager = PluginManager(broker=orchify_broker)
    recorder = Recorder()
    alpha = f'plugalpha{pid}'
    beta = f'plugbeta{pid}'
    broken = f'plugbroken{pid}'
    try:
        os.makedirs(base, exist_ok=True)
        _write(os.path.join(base, '__init__.py'), '')
        _write(os.path.join(base, f'{alpha}.py'), '\n'.join([
            'from orchify.plugin import Plugin',
            '',
            'class AlphaPlugin(Plugin):',
            "    name = 'file-alpha'",
            '',
            'PLUGINS = [AlphaPlugin]',
            '',
        ]))
        _write(os.path.join(base, f'{beta}.py'), '\n'.join([
            'from orchify.plugin import Plugin',
            '',
            'class BetaPlugin(Plugin):',
            "    name = 'file-beta'",
            '',
        ]))
        _write(os.path.join(base, f'{broken}.py'), "raise RuntimeError('import boom')\n")

        files = manager.discover(base)
        assert len(files) == 3
        assert all(os.path.basename(path).endswith('.py') for path in files)
        assert all('__init__' not in path for path in files)

        loaded = manager.load_from_dir(base)
        assert sorted(inst.name for inst in loaded) == ['file-alpha', 'file-beta']
        assert sorted(manager.loaded) == ['file-alpha', 'file-beta']

        errors = recorder.of('plugin:error')
        assert any(event.payload.get('action') == 'import' and 'import boom' in event.payload.get('error', '')
                   for event in errors)

        _raises(PluginError, lambda: manager.discover(os.path.join(base, 'nope')),
                contains='does not exist')
        _raises(PluginError, lambda: manager.load_from_file(os.path.join(base, 'nope.py')),
                contains='does not exist')
    finally:
        recorder.close()
        manager.unregister('file-alpha')
        manager.unregister('file-beta')
        for key in [k for k in list(sys.modules) if k.endswith(str(pid))]:
            sys.modules.pop(key, None)
        shutil.rmtree(base, ignore_errors=True)


def test_manager_load_from_package_and_single_file():
    '''Plugins load from an installed-style package and a single file.'''
    pid = os.getpid()
    package = f'plugpkg{pid}'
    package_dir = os.path.join(ROOT, package)
    single = os.path.join(ROOT, f'plugsingle{pid}.py')
    manager = PluginManager(broker=orchify_broker)
    recorder = Recorder()
    try:
        os.makedirs(package_dir, exist_ok=True)
        _write(os.path.join(package_dir, '__init__.py'), '')
        _write(os.path.join(package_dir, 'inner.py'), '\n'.join([
            'from orchify.plugin import Plugin',
            '',
            'class InnerPlugin(Plugin):',
            "    name = 'pkg-inner'",
            '',
        ]))
        loaded = manager.load_from_package(package)
        assert [inst.name for inst in loaded] == ['pkg-inner']

        _write(single, '\n'.join([
            'from orchify.plugin import Plugin',
            '',
            'class SinglePlugin(Plugin):',
            "    name = 'file-single'",
            '',
        ]))
        loaded = manager.load_from_file(single)
        assert 'file-single' in [inst.name for inst in loaded]
        assert sorted(manager.loaded) == ['file-single', 'pkg-inner']

        reloaded = manager.load_from_file(single, auto_load=False)
        assert reloaded == []
        errors = recorder.of('plugin:error')
        assert any(event.payload.get('action') == 'import' and 'already registered' in event.payload.get('error', '')
                   for event in errors)
    finally:
        recorder.close()
        manager.unregister('pkg-inner')
        manager.unregister('file-single')
        for key in [k for k in list(sys.modules)
                    if k == f'plugsingle{pid}' or k.startswith(f'{package}.') or k == package]:
            sys.modules.pop(key, None)
        shutil.rmtree(package_dir, ignore_errors=True)
        try:
            os.remove(single)
        except OSError:
            pass


def main():
    '''Run every test in this module and report a summary.'''
    tests = [
        test_declarative_plugin_registers_hooks_and_tools,
        test_plugin_scope_filter_limits_hooks,
        test_imperative_hooks_and_cleanup,
        test_plugin_tools_attach_detach_and_type_errors,
        test_plugin_helpers_event_feedback_approvals_and_logging,
        test_plugin_middleware_targets_llm_and_agents,
        test_plugin_get_config_reads_the_run_context,
        test_async_lifecycle_and_load_failures,
        test_manager_registration_errors_and_dependency_order,
        test_manager_load_all_unload_all_and_configure,
        test_manager_discovers_and_loads_plugin_files,
        test_manager_load_from_package_and_single_file,
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
