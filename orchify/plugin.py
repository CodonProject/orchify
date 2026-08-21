import inspect
import asyncio
import types
from typing import Callable, Optional, Any, List, Dict, Set, Union

from orchify.tool import Tool
from orchify.broker import orchify_broker
from orchify.llm.base import scope_matches, Middleware


class Plugin:
    '''
    Base class for Orchify plugins.

    Declare metadata as class attributes, then write hooks/tools in two styles:

      显式声明式            -> 装饰器标记，方法保持普通方法，可继续被调用
           @Plugin.hook('agent:finish', agent_name='Agent')
           def answer(self, event): ...

           @Plugin.tool('web_search', agent=agent)     # agent 给定时自动挂载
           def search(self, query: str) -> str: ...

      命令式（进阶）        -> 在 on_load 里动态注册
           def on_load(self):
               @self.on('run:next')
               def _(e): ...

    Lifecycle methods (on_load/on_unload) may be sync or async (async ones run
    on the WebBackend loop). Everything the plugin registered is cleaned up
    automatically on unload.

    `scope` restricts the plugin's hooks and middleware to certain agents:
      - '*' (default): every agent.
      - 'agent':       any named agent.
      - 'agent:name':  only the agent named `name`.
      - list:          any of the above at once.
      - callable(name): dynamic check on the agent name; return truthy to match.
    '''
    name: str = ''
    version: str = '0.0.1'
    description: str = ''
    dependencies: List[str] = []
    tags: List[str] = []
    events: Union[str, List[str]] = []
    scope: Union[str, List[str], Callable[[str], bool]] = '*'

    def __init__(self, broker=None):
        self.broker = broker or orchify_broker
        self.loaded: bool = False
        self.state: Dict[str, Any] = {}
        if isinstance(getattr(self, 'scope', None), types.MethodType) \
                and getattr(self.scope, '__self__', None) is self:
            self.scope = self.scope.__func__
        self._bindings: list = []
        self._tools: List[Tool] = []
        self._attached_agents: Set[Any] = set()
        self._llm = None

    @property
    def id(self) -> str:
        '''Stable plugin identifier used for ownership tracking (defaults to class name).'''
        return self.name or self.__class__.__name__

    # ---------- scope ----------

    def matches(self, agent) -> bool:
        '''Whether this plugin's scope includes the given Agent (or agent name string).'''
        name = getattr(agent, 'name', None)
        if name is None and isinstance(agent, str):
            name = agent
        return scope_matches(self.scope, name)

    def _scope_filter(self) -> Optional[Callable]:
        '''Returns a hook match filter enforcing this plugin's scope, or None if global.'''
        if self.scope in (None, '*'):
            return None
        return lambda event: scope_matches(self.scope, event.agent_name)

    # ---------- declarative markers (class-level decorators) ----------

    @staticmethod
    def hook(event_type: Union[str, List[str]] = '*', *, turn_id: Optional[str] = None,
             mode: str = 'normal', once: bool = False,
             agent_name: Optional[str] = None, match: Optional[Callable] = None) -> Callable:
        '''Mark a method to be registered as a hook when the plugin loads.
        `event_type` may be a single string or a list of strings.'''
        def decorator(func: Callable) -> Callable:
            func.__plugin_hook__ = dict(
                event_type=event_type, turn_id=turn_id,
                mode='once' if once else mode, agent_name=agent_name, match=match,
            )
            return func
        return decorator

    @staticmethod
    def tool(name: Optional[str] = None, *, agent=None, **kwargs) -> Callable:
        '''Mark a method to be collected as a Tool when the plugin loads.

        `agent=` attaches the tool to that Agent immediately; otherwise use
        `attach_tools(agent)` or `orchify_plugins.attach('name', agent)` later.
        Remaining kwargs are forwarded to Tool(func=..., name=..., **kwargs).
        '''
        def decorator(func: Callable) -> Callable:
            meta = getattr(func, '__plugin_tool__', {}) if hasattr(func, '__plugin_tool__') else {}
            meta.update(name=name, agent=agent, tool_kwargs=kwargs)
            func.__plugin_tool__ = meta
            return func
        return decorator

    # ---------- auto-discovery ----------

    def _auto_register(self) -> None:
        pid = self.id
        scope_filter = self._scope_filter()

        for method_name, _ in inspect.getmembers(type(self), predicate=inspect.isfunction):
            bound = getattr(self, method_name, None)

            meta = getattr(bound, '__plugin_hook__', None)
            if isinstance(meta, dict):
                kwargs = {k: v for k, v in meta.items() if k != 'event_type'}
                user_match = kwargs.get('match')
                if scope_filter is not None:
                    kwargs['match'] = (lambda e, m=user_match: (m(e) if m else True) and scope_filter(e))
                binding = self.broker.register_hook(meta['event_type'], bound, owner=pid, **kwargs)
                self._bindings.append(binding)
                continue

            tmeta = getattr(bound, '__plugin_tool__', None)
            if isinstance(tmeta, dict):
                tool = Tool(func=bound, name=tmeta.get('name'), **tmeta.get('tool_kwargs', {}))
                self._tools.append(tool)
                agent = tmeta.get('agent')
                if agent is not None:
                    agent.register_tool(tool)
                    self._attached_agents.add(agent)
                continue

    # ---------- lifecycle ----------

    def on_load(self) -> None:
        '''Additional imperative setup, called after declarative auto-registration.'''

    def on_unload(self) -> None:
        '''Teardown, called before the plugin's hooks/tools are cleaned up.'''

    def _run_lifecycle(self, method: str) -> Any:
        func = getattr(self, method)
        if inspect.iscoroutinefunction(func):
            from orchify.backend import orchify_web_backend
            future = asyncio.run_coroutine_threadsafe(func(), orchify_web_backend._loop)
            return future.result()
        return func()

    # ---------- imperative helpers (usable inside on_load) ----------

    def on(self, event_type: Union[str, List[str]] = '*', *, turn_id: Optional[str] = None,
           mode: str = 'normal', once: bool = False,
           agent_name: Optional[str] = None, match: Optional[Callable] = None) -> Callable:
        '''Register a hook tracked by this plugin (auto-removed on unload).
        `event_type` may be a single string or a list of strings.'''
        def decorator(func: Callable) -> Callable:
            binding = self.broker.register_hook(
                event_type, func,
                turn_id=turn_id,
                mode='once' if once else mode,
                agent_name=agent_name,
                match=match,
                owner=self.id,
            )
            self._bindings.append(binding)
            return func
        return decorator

    def event(self, event_type: str, **kwargs) -> Any:
        '''Build a custom event owned by this plugin.'''
        kwargs.setdefault('source', 'plugin')
        return self.broker.event(event_type, **kwargs)

    def feedback(self, ftype: str, *, task_name: Optional[str] = None,
                 turn_id: Optional[str] = None, comment: Optional[str] = None,
                 payload: Optional[dict] = None) -> bool:
        '''Convenience wrapper around broker.feedback() for the reverse control channel.'''
        return self.broker.feedback(ftype, task_name=task_name, turn_id=turn_id,
                                    comment=comment, payload=payload)

    def add_tool(self, tool: Tool, agent=None) -> Tool:
        '''Register an already-constructed Tool with this plugin; optionally attach to an agent.'''
        if not isinstance(tool, Tool):
            raise TypeError(f'add_tool() requires a Tool, got {type(tool)}')
        self._tools.append(tool)
        if agent is not None:
            agent.register_tool(tool)
            self._attached_agents.add(agent)
        return tool

    def attach_tools(self, agent, replace: bool = True) -> Any:
        '''Attach every tool this plugin owns to the given Agent.'''
        for t in self._tools:
            agent.register_tool(t, replace=replace)
            self._attached_agents.add(agent)
        return agent

    def detach_tools(self) -> None:
        '''Remove this plugin's tools from any agents they were attached to.'''
        for agent in list(self._attached_agents):
            for t in self._tools:
                agent.remove_tool(t.name)
        self._attached_agents.clear()

    def middleware(self, mw, llm=None) -> None:
        '''Register an LLM middleware scoped to this plugin's `scope` (uses the
        manager-provided llm or the attached agents' llm).
        The middleware's `owner` attribute is set to this plugin's id so that
        per-agent plugin filtering can skip it when the plugin is disabled.'''
        if llm is None:
            llm = self._llm
        if not isinstance(mw, Middleware):
            from orchify.llm.base import _FuncMiddleware
            mw = _FuncMiddleware(mw)
        mw.owner = self.id
        if llm is not None:
            llm.use(mw, scope=self.scope)
            return
        for agent in list(self._attached_agents):
            agent.llm.use(mw, scope=self.scope)
        if not self._attached_agents:
            self.log('middleware deferred: no llm available (configure the manager '
                     'with an llm or attach your plugin to an agent)')

    def log(self, msg: str) -> None:
        '''Namespaced logging with this plugin's id.'''
        print(f'[{self.id}] {msg}', flush=True)

    # ---------- plugin config ----------

    def get_config(self, key: str = None, task_name: str = None):
        '''
        Read plugin configuration from the broker task context.

        Args:
            key: specific config key to retrieve. If None, returns the full
                 config dict for this plugin.
            task_name: the task/run name to look up. If None, attempts to
                       find the currently active task for this plugin's agent.

        Returns the config value, or None if not found.
        '''
        if task_name is None:
            task_name = self._find_active_task()
        if task_name is None:
            return None
        ctx = self.broker.get_run_context(task_name)
        if ctx is None:
            return None
        cfg = ctx.get('plugin_config', {}).get(self.id)
        if cfg is None:
            return None
        if key is None:
            return cfg
        return cfg.get(key)

    def _find_active_task(self) -> Optional[str]:
        '''Find the most recent active task for this plugin's scope.'''
        for task_name, ctx in self.broker.run_contexts.items():
            agent_name = ctx.get('agent_name', '')
            if self.matches(agent_name):
                return task_name
        return None

    # ---------- cleanup ----------

    def cleanup(self) -> None:
        '''Full teardown: lifecycle, hooks, tools. Called by the PluginManager.'''
        if self.loaded:
            self._run_lifecycle('on_unload')
            self.loaded = False
        self.broker.remove_hooks(self.id)
        self.detach_tools()
        self._bindings.clear()