import inspect
import asyncio
import types
from typing import Callable, Optional, Any, List, Dict, Set, Union

from orchify.tool import Tool
from orchify.broker import orchify_broker
from orchify.llm.base import scope_matches, Middleware
from orchify.utils import get_logger

logger = get_logger(__name__)


class Plugin:
    '''Base class for Orchify plugins.

    Subclass it, declare metadata as class attributes, then declare hooks and
    tools in either style.

    Declarative, collected automatically when the plugin is loaded::

        @Plugin.hook('agent:finish', agent_name='Agent')
        def answer(self, event): ...

        @Plugin.tool('web_search', agent=agent)
        def search(self, query: str) -> str: ...

    Imperative, registered dynamically from ``on_load``::

        def on_load(self):
            @self.on('run:next')
            def _(event): ...

    The lifecycle hooks ``on_load`` and ``on_unload`` may be sync or async;
    async ones run on the shared WebBackend loop. Everything the plugin
    registered is removed automatically when it is unloaded.

    Attributes:
        name (str): Plugin id; defaults to the class name.
        version (str): Free-form version string.
        description (str): Human-readable summary.
        dependencies (List[str]): Plugin names that must be loaded first.
        tags (List[str]): Free-form tags.
        events (List[str]): Event types the plugin is interested in.
        scope (Any): Restricts the plugin's hooks and middleware to matching
            agents: ``'*'`` (every agent), ``'agent'`` (any named agent),
            ``'agent:name'`` (only that agent), a list of the above, or a
            callable that receives the agent name and returns a truthy value
            when it matches.
    '''

    name: str = ''
    version: str = '0.0.1'
    description: str = ''
    dependencies: List[str] = []
    tags: List[str] = []
    events: Union[str, List[str]] = []
    scope: Union[str, List[str], Callable[[str], bool]] = '*'

    def __init__(self, broker=None):
        '''Bind the plugin to a broker and initialise its per-instance state.

        Args:
            broker (Optional[Broker]): Broker the plugin hooks into and pushes
                feedback through; defaults to the shared ``orchify_broker``.
        '''
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


    @staticmethod
    def hook(event_type: Union[str, List[str]] = '*', *, run_id: Optional[str] = None,
             turn_id: Optional[str] = None,
             mode: str = 'normal', once: bool = False,
             agent_name: Optional[str] = None, match: Optional[Callable] = None) -> Callable:
        '''Mark a method as a hook to register when the plugin loads.

        Args:
            event_type (Union[str, List[str]]): Event type or types to listen for.
                Defaults to ``'*'`` (every event).
            run_id (Optional[str]): Bind the hook to a single run; the hook is
                removed when that run finishes.
            turn_id (Optional[str]): Legacy alias of ``run_id``.
            mode (str): ``'normal'``, ``'once'`` or ``'agent'`` (bound to one run).
            once (bool): Shorthand for ``mode='once'``.
            agent_name (Optional[str]): Only fire for events from this agent.
            match (Optional[Callable]): Extra predicate receiving the event; the
                hook runs only when it returns a truthy value.

        Returns:
            Callable: A decorator that records the method and returns it unchanged.
        '''
        def decorator(func: Callable) -> Callable:
            '''Record ``func`` as a plugin hook.

            Args:
                func (Callable): The hook implementation.

            Returns:
                Callable: ``func``, unchanged.
            '''
            func.__plugin_hook__ = dict(
                event_type=event_type, run_id=run_id, turn_id=turn_id,
                mode='once' if once else mode, agent_name=agent_name, match=match,
            )
            return func
        return decorator

    @staticmethod
    def tool(name: Optional[str] = None, *, agent=None, **kwargs) -> Callable:
        '''Mark a method as a Tool to collect when the plugin loads.

        Args:
            name (Optional[str]): Tool name; defaults to the method name.
            agent (Optional[Agent]): Attach the tool to this agent immediately.
                Otherwise attach it later with ``attach_tools()`` or
                ``orchify_plugins.attach('plugin_name', agent)``.
            **kwargs: Forwarded to :class:`Tool` (description, parameters, cache,
                cache_size, timeout, require_approval, ask_question).

        Returns:
            Callable: A decorator that records the method and returns it unchanged.
        '''
        def decorator(func: Callable) -> Callable:
            '''Record ``func`` as a Tool of this plugin.

            Args:
                func (Callable): The tool implementation.

            Returns:
                Callable: ``func``, unchanged.
            '''
            meta = getattr(func, '__plugin_tool__', {}) if hasattr(func, '__plugin_tool__') else {}
            meta.update(name=name, agent=agent, tool_kwargs=kwargs)
            func.__plugin_tool__ = meta
            return func
        return decorator


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
                    agent.add_tool(tool)
                    self._attached_agents.add(agent)
                continue


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


    def on(self, event_type: Union[str, List[str]] = '*', *, run_id: Optional[str] = None,
           turn_id: Optional[str] = None, mode: str = 'normal', once: bool = False,
           agent_name: Optional[str] = None, match: Optional[Callable] = None) -> Callable:
        '''Register a hook tracked by this plugin.

        Registered hooks are removed automatically when the plugin is unloaded.

        Args:
            event_type (Union[str, List[str]]): Event type or types to listen for.
                Defaults to ``'*'``.
            run_id (Optional[str]): Bind the hook to a single run.
            turn_id (Optional[str]): Legacy alias of ``run_id``.
            mode (str): ``'normal'``, ``'once'`` or ``'agent'``.
            once (bool): Shorthand for ``mode='once'``.
            agent_name (Optional[str]): Only fire for events from this agent.
            match (Optional[Callable]): Extra predicate receiving the event.

        Returns:
            Callable: A decorator that registers the hook and returns it unchanged.
        '''
        def decorator(func: Callable) -> Callable:
            '''Register ``func`` as a hook owned by this plugin.

            Args:
                func (Callable): The hook implementation.

            Returns:
                Callable: ``func``, unchanged.
            '''
            binding = self.broker.register_hook(
                event_type, func,
                run_id=run_id,
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

    def feedback(self, ftype: str, *, run_id: Optional[str] = None,
                 turn_id: Optional[str] = None, task_name: Optional[str] = None,
                 payload: Optional[dict] = None) -> bool:
        '''Queue a control message on the reverse channel.

        Args:
            ftype (str): Control type from ``FEEDBACK_TYPES``, e.g. ``'control:stop'``
                or ``'control:resume'``.
            run_id (Optional[str]): Target run id, as returned by ``Agent.run``.
            turn_id (Optional[str]): Legacy alias of ``run_id``.
            task_name (Optional[str]): Deprecated alias of ``run_id``.
            payload (Optional[dict]): Control-specific data.

        Returns:
            bool: True when the broker accepted the message.
        '''
        return self.broker.feedback(ftype, run_id=run_id, turn_id=turn_id,
                                    task_name=task_name, payload=payload)

    def pending_approvals(self, run_id: Optional[str] = None) -> list:
        '''Manual-confirmation requests still waiting for a decision.'''
        return self.broker.pending_approvals(run_id)

    def approve(self, approval_id: str, *, reason: str = '', remove: bool = False) -> bool:
        '''Accept a pending approval request. Returns False if it is unknown or stale.'''
        return self.broker.approve(approval_id, reason=reason, remove=remove)

    def reject(self, approval_id: str, *, reason: str = 'denied', remove: bool = False) -> bool:
        '''Deny a pending approval request. Returns False if it is unknown or stale.'''
        return self.broker.reject(approval_id, reason=reason, remove=remove)

    def add_tool(self, tool: Tool, agent=None) -> Tool:
        '''Register an already-constructed Tool with this plugin; optionally attach to an agent.'''
        if not isinstance(tool, Tool):
            raise TypeError(f'add_tool() requires a Tool, got {type(tool)}')
        self._tools.append(tool)
        if agent is not None:
            agent.add_tool(tool)
            self._attached_agents.add(agent)
        return tool

    def attach_tools(self, agent, replace: bool = True) -> Any:
        '''Attach every tool this plugin owns to the given Agent.'''
        for t in self._tools:
            agent.add_tool(t, replace=replace)
            self._attached_agents.add(agent)
        return agent

    def detach_tools(self) -> None:
        '''Remove this plugin's tools from any agents they were attached to.'''
        for agent in list(self._attached_agents):
            for t in self._tools:
                agent.remove_tool(t.name)
        self._attached_agents.clear()

    def middleware(self, mw, llm=None) -> None:
        '''Register an LLM middleware scoped to this plugin.

        The middleware is installed on the given LLM (or on the plugin's attached
        agents' LLM) with ``owner`` set to this plugin's id, so per-agent plugin
        filtering skips it while the plugin is disabled.

        Args:
            mw (Union[Middleware, Callable]): Middleware instance, or a plain
                ``(kwargs, next_request)`` async callable.
            llm (Optional[LLMInterface]): LLM to register on; defaults to the
                plugin's shared LLM.
        '''
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
        logger.info('[%s] %s', self.id, msg)


    def get_config(self, key: str = None, run_id: str = None, task_name: str = None):
        '''Read this plugin's configuration from the active run.

        Args:
            key (str): Configuration key to read. When None, the full config dict
                for this plugin is returned.
            run_id (str): Run to look up, as returned by ``Agent.run``. When None,
                the plugin's currently active run is used.
            task_name (str): Deprecated alias of ``run_id``.

        Returns:
            Any: The requested value, the whole config dict, or None when no
            configuration is available for the run.
        '''
        run_id = run_id or task_name
        if run_id is None:
            run_id = self._find_active_run()
        if run_id is None:
            return None
        ctx = self.broker.get_run_context(run_id)
        if ctx is None:
            return None
        cfg = ctx.get('plugin_config', {}).get(self.id)
        if cfg is None:
            return None
        if key is None:
            return cfg
        return cfg.get(key)

    def _find_active_run(self) -> Optional[str]:
        '''Find the most recent active run for this plugin's scope.'''
        for run_id, ctx in self.broker.run_contexts.items():
            agent_name = ctx.get('agent_name', '')
            if self.matches(agent_name):
                return run_id
        return None


    def cleanup(self) -> None:
        '''Full teardown: lifecycle, hooks, tools. Called by the PluginManager.'''
        if self.loaded:
            self._run_lifecycle('on_unload')
            self.loaded = False
        self.broker.remove_hooks(self.id)
        self.detach_tools()
        self._bindings.clear()
