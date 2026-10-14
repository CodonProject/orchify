import glob
import importlib
import os
import sys
from typing import List, Optional, Type, Union

from orchify.plugin import Plugin
from orchify.broker import orchify_broker
from orchify.utils import get_logger

logger = get_logger(__name__)


class PluginError(Exception):
    '''Raised for plugin registration, loading and dependency errors.'''


class PluginManager:
    '''Registry and loader for Orchify plugins.

    Usage::

        from orchify.plugin_manager import orchify_plugins

        class MyPlugin(Plugin): ...

        orchify_plugins.register(MyPlugin)
        orchify_plugins.load('MyPlugin')
        orchify_plugins.unload('MyPlugin')

    Plugins are loaded in dependency order (a topological sort over each
    plugin's ``dependencies``). The lifecycle events ``plugin:load``,
    ``plugin:unload`` and ``plugin:error`` are emitted through the broker so
    other hooks and plugins can react.

    Attributes:
        plugins (Dict[str, Plugin]): Registered plugins by id.
    '''

    def __init__(self, broker=None, llm=None):
        '''Create an empty registry.

        Args:
            broker (Optional[Broker]): Broker handed to every plugin; defaults to
                the shared ``orchify_broker``.
            llm (Optional[LLMInterface]): LLM backend handed to plugins that
                register their own provider or middleware.
        '''
        from orchify.broker import orchify_broker as _default_broker
        self.broker = broker or _default_broker
        self.llm = llm
        self.plugins: dict[str, Plugin] = {}
        self._order: list[str] = []
        self._order_dirty: bool = True
        self._registry: dict[str, type] = {}


    def _emit(self, event_type: str, payload: dict) -> None:
        event = self.broker.event(event_type=event_type, payload=payload, source='plugin')
        self.broker.emit(event)


    def register(self, plugin: Union[Type[Plugin], Plugin], *, enabled: bool = True) -> Optional[Plugin]:
        '''Register a Plugin subclass or instance.

        Args:
            plugin (Union[Type[Plugin], Plugin]): The class or instance to register.
            enabled (bool): When False the plugin is ignored entirely.

        Returns:
            Optional[Plugin]: The registered instance, or None when ``enabled`` is
            False.

        Raises:
            PluginError: When the plugin has no id or that id is already registered.
        '''
        cls: Type[Plugin]
        instance: Optional[Plugin] = None
        if isinstance(plugin, type) and issubclass(plugin, Plugin):
            cls = plugin
        elif isinstance(plugin, Plugin):
            instance = plugin
            cls = type(plugin)
        else:
            raise PluginError(f'Invalid plugin registration: {plugin!r}')

        name = cls.name or cls.__name__
        if name in self.plugins or name in self._registry:
            raise PluginError(f"Plugin '{name}' is already registered.")

        self._registry[name] = cls
        self._order_dirty = True
        if enabled:
            if instance is None:
                instance = cls(broker=self.broker)
            instance._llm = self.llm
            self.plugins[name] = instance
        else:
            instance = None
        return instance

    def configure(self, llm=None, broker=None) -> None:
        '''Set the shared LLM and/or broker used by plugins.

        Takes effect immediately for registered plugins and is applied to plugins
        registered later.

        Args:
            llm (Optional[LLMInterface]): LLM shared with every plugin.
            broker (Optional[Broker]): Broker shared with every plugin.
        '''
        if broker is not None:
            self.broker = broker
        self.llm = llm
        for inst in self.plugins.values():
            inst._llm = llm

    def unregister(self, name: str) -> None:
        '''Unload (if loaded) and remove a plugin from the registry entirely.'''
        if name in self.plugins:
            self.unload(name)
        self._registry.pop(name, None)
        self.plugins.pop(name, None)
        self._order_dirty = True

    def get(self, name: str) -> Plugin:
        '''Return a registered plugin by id.

        Args:
            name (str): Plugin id.

        Returns:
            Plugin: The registered plugin.

        Raises:
            PluginError: When no plugin is registered under ``name``.
        '''
        if name not in self.plugins:
            raise PluginError(f"Plugin '{name}' is not registered.")
        return self.plugins[name]

    def registered(self) -> dict[str, Plugin]:
        '''Return a copy of the plugin registry.

        Returns:
            dict[str, Plugin]: Registered plugins keyed by id.
        '''
        return dict(self.plugins)

    def attach(self, name: str, agent) -> None:
        '''Attach all tools owned by a registered plugin to a specific Agent.'''
        if name not in self.plugins:
            raise PluginError(f"Plugin '{name}' is not registered.")
        self.plugins[name].attach_tools(agent)

    @property
    def loaded(self) -> list[str]:
        '''Return the ids of the currently loaded plugins.

        Returns:
            list[str]: Loaded plugin ids, in registration order.
        '''
        self._ensure_order()
        return [n for n in self._order if self.plugins[n].loaded]


    def _ensure_order(self) -> None:
        if self._order_dirty:
            self._recompute_order()
            self._order_dirty = False

    def _recompute_order(self) -> None:
        names = list(self.plugins)
        self._order = []
        visited, visiting = set(), set()

        def visit(name: str, trail: tuple):
            if name in visiting:
                cycle = ' -> '.join(trail + (name,))
                raise PluginError(f'Circular plugin dependency: {cycle}')
            if name in visited:
                return
            visiting.add(name)
            for dep in self.plugins[name].dependencies:
                if dep not in self.plugins:
                    raise PluginError(f"Plugin '{name}' depends on unknown plugin '{dep}'.")
                visit(dep, trail + (name,))
            visiting.discard(name)
            visited.add(name)
            self._order.append(name)

        for n in names:
            visit(n, ())


    def load(self, name: Optional[str] = None) -> List[Plugin]:
        '''Load one plugin (with its dependencies) or all registered plugins.'''
        if name is None:
            return self.load_all()
        if name not in self.plugins:
            raise PluginError(f"Plugin '{name}' is not registered.")
        self._ensure_order()
        closure = self._dependency_closure(name)
        resolved = [n for n in self._order if n in closure]
        return [self._load_one(n) for n in resolved]

    def _dependency_closure(self, name: str) -> set:
        closure = {name}
        changed = True
        while changed:
            changed = False
            for n in list(closure):
                for d in self.plugins[n].dependencies:
                    if d not in closure:
                        closure.add(d)
                        changed = True
        return closure

    def load_all(self) -> List[Plugin]:
        '''Load every registered plugin in dependency order.

        Returns:
            List[Plugin]: The plugin instances that are loaded.
        '''
        self._ensure_order()
        return [self._load_one(n) for n in self._order]

    def _load_one(self, name: str) -> Plugin:
        inst = self.plugins[name]
        if inst.loaded:
            return inst
        if inst._llm is None:
            inst._llm = self.llm
        try:
            inst._auto_register()
            inst._run_lifecycle('on_load')
        except Exception as e:
            inst.broker.remove_hooks(inst.id)
            self._emit('plugin:error', {'plugin': name, 'action': 'load', 'error': str(e)})
            raise PluginError(f"Failed to load plugin '{name}': {e}") from e
        inst.loaded = True
        self._emit('plugin:load', {
            'plugin': name,
            'version': inst.version,
            'description': inst.description,
            'tags': list(inst.tags),
            'scope': inst.scope,
            'dependencies': list(inst.dependencies),
            'events': list(inst.events) if isinstance(inst.events, (list, tuple, set)) else [inst.events],
            'hooks': [t for b in inst._bindings for t in b.event_type],
            'tools': [t.name for t in inst._tools],
        })
        return inst

    def unload(self, name: Optional[str] = None) -> List[str]:
        '''Unload one plugin or all loaded plugins (reverse dependency order). Returns unloaded names.'''
        self._ensure_order()
        if name is None:
            order = [n for n in reversed(self._order) if self.plugins[n].loaded]
        else:
            if name not in self.plugins:
                raise PluginError(f"Plugin '{name}' is not registered.")
            closure = self._dependent_closure(name)
            order = [n for n in reversed(self._order) if n in closure and self.plugins[n].loaded]
        for n in order:
            self._unload_one(n)
        return order

    def _dependent_closure(self, name: str) -> set:
        closure = {name}
        changed = True
        while changed:
            changed = False
            for n in self._order:
                if n not in closure and any(d in closure for d in self.plugins[n].dependencies):
                    closure.add(n)
                    changed = True
        return closure

    def unload_all(self) -> List[str]:
        '''Unload every loaded plugin in reverse order.

        Returns:
            List[str]: The ids of the plugins that were unloaded.
        '''
        return self.unload()

    def _unload_one(self, name: str) -> None:
        inst = self.plugins[name]
        try:
            inst.cleanup()
            self._emit('plugin:unload', {'plugin': name, 'version': inst.version})
        except Exception as e:
            self._emit('plugin:error', {'plugin': name, 'action': 'unload', 'error': str(e)})
            raise PluginError(f"Failed to unload plugin '{name}': {e}") from e


    def discover(self, path: Union[str, List[str]], pattern: str = '*.py') -> List[str]:
        '''Scan a directory (or directories) for loadable plugin modules.'''
        paths = [path] if isinstance(path, str) else path
        files: List[str] = []
        for p in paths:
            p = os.path.abspath(p)
            if not os.path.isdir(p):
                raise PluginError(f'Plugin directory does not exist: {p}')
            for f in glob.glob(os.path.join(p, pattern)):
                if os.path.basename(f) == '__init__.py':
                    continue
                files.append(f)
        return files

    def _register_from_module(self, module) -> List[Plugin]:
        declared = getattr(module, 'PLUGINS', None)
        candidates = []
        if declared is not None:
            candidates = list(declared)
        else:
            candidates = [obj for obj in vars(module).values()
                          if isinstance(obj, type) and issubclass(obj, Plugin)
                          and obj is not Plugin
                          and getattr(obj, '__module__', '') == module.__name__]
        registered = []
        for cand in candidates:
            if (isinstance(cand, type) and issubclass(cand, Plugin)) or isinstance(cand, Plugin):
                inst = self.register(cand)
            else:
                continue
            if inst is not None:
                registered.append(inst)
        return registered

    def _import_modules(self, mod_names: List[str], sys_path: Optional[str] = None) -> None:
        '''Import plugin modules, updating sys.path if needed.

        A module that fails to import is reported (``plugin:error`` plus a warning)
        but does not abort the remaining modules.

        Args:
            mod_names (List[str]): Module names to import.
            sys_path (Optional[str]): Directory prepended to ``sys.path`` for the
                duration of the import.
        '''
        added = sys_path is not None and sys_path not in sys.path
        if added:
            sys.path.insert(0, sys_path)
        try:
            for mod_name in mod_names:
                try:
                    self._import_module(mod_name, suppress_errors=True)
                except Exception as e:
                    logger.warning('[plugin] skipped module %s: %s', mod_name, e)
                    self._emit('plugin:error', {'plugin': mod_name, 'action': 'import', 'error': str(e)})
        finally:
            if added:
                sys.path.remove(sys_path)

    def load_from_dir(self, path: str, *, auto_load: bool = True) -> List[Plugin]:
        '''Import every ``*.py`` in a directory as plugins.

        Args:
            path (str): Directory to scan.
            auto_load (bool): When True, load each plugin right after registering it.

        Returns:
            List[Plugin]: The plugins registered from the directory.
        '''
        files = self.discover(path)
        mod_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        self._import_modules(mod_names, sys_path=os.path.abspath(path))
        return self.load_all() if auto_load else []

    def load_from_file(self, path: str, *, auto_load: bool = True) -> List[Plugin]:
        '''Import a single plugin file and register/load the plugins it defines.'''
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise PluginError(f'Plugin file does not exist: {path}')
        mod_name = os.path.splitext(os.path.basename(path))[0]
        self._import_modules([mod_name], sys_path=os.path.dirname(path))
        return self.load_all() if auto_load else []

    def load_from_package(self, package: str, *, auto_load: bool = True) -> List[Plugin]:
        '''Import plugin modules from inside an installed package (e.g. 'app.plugins').'''
        pkg = importlib.import_module(package)
        pkg_dir = os.path.dirname(os.path.abspath(pkg.__file__))
        mod_names = [f'{package}.{os.path.splitext(os.path.basename(f))[0]}' for f in self.discover(pkg_dir)]
        self._import_modules(mod_names)
        return self.load_all() if auto_load else []

    def _import_module(self, mod_name: str, *, suppress_errors: bool = False) -> Optional[object]:
        if mod_name in sys.modules:
            module = importlib.reload(sys.modules[mod_name])
        else:
            module = importlib.import_module(mod_name)
        try:
            self._register_from_module(module)
        except PluginError as e:
            if not suppress_errors:
                raise
            logger.warning('[plugin] skipped module %s: %s', mod_name, e)
            self._emit('plugin:error', {'plugin': mod_name, 'action': 'import', 'error': str(e)})
        return module


orchify_plugins = PluginManager()
