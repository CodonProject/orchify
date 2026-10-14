from typing import List, Dict, Any, Optional, Set, Union, AsyncGenerator, Literal, Callable, TYPE_CHECKING
import asyncio
from concurrent.futures import ThreadPoolExecutor

from orchify.llm        import LLMInterface
from orchify.tool       import Tool
from orchify.event      import AgentEvent, ToolEvent, RuntimeEvent, ToolEventAssembler, ApprovalEvent
from orchify.utils      import safecode, get_logger
from orchify.env        import req_model
from orchify.broker     import orchify_broker, ApprovalTimeout, RunStopped, POLL_INTERVAL, make_turn_id
from orchify.backend    import orchify_web_backend
from orchify.call_chain import CallChain, CallBlocked, plan_call
from orchify.budget     import Budget
from orchify.gate       import get_llm_gate, get_llm_rpm, get_run_gate

if TYPE_CHECKING:
    from orchify.group import Group

import json
import os
import time
import threading


logger = get_logger(__name__)


def _read_timeout_env(name: str) -> Optional[float]:
    '''Read a non-negative float timeout from the environment, if valid.'''
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _read_tool_workers() -> int:
    '''Worker count for the shared tool executor (env ORCHIFY_TOOL_WORKERS).'''
    raw = os.getenv('ORCHIFY_TOOL_WORKERS')
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return 32


def _read_int_env(name: str, default: int = 0) -> int:
    '''A non-negative int from the environment, else `default`.'''
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _read_positive_float_env(name: str) -> Optional[float]:
    '''A positive float from the environment, else None.'''
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


_tool_executor: Optional[ThreadPoolExecutor] = None
_tool_executor_lock = threading.Lock()


def get_tool_executor() -> ThreadPoolExecutor:
    '''Return the process-wide executor that runs Tool calls off the event loop.

    Created lazily on first use (not at import time) so importing orchify never
    spawns threads, and so the worker count honours ORCHIFY_TOOL_WORKERS.

    Returns:
        ThreadPoolExecutor: The shared executor.
    '''
    global _tool_executor
    if _tool_executor is None:
        with _tool_executor_lock:
            if _tool_executor is None:
                _tool_executor = ThreadPoolExecutor(
                    max_workers=_read_tool_workers(),
                    thread_name_prefix='OrchifyToolExecutor',
                )
    return _tool_executor


def shutdown_tool_executor(wait: bool = True) -> None:
    '''Shut the shared tool executor down (call at process exit for a clean join).'''
    global _tool_executor
    with _tool_executor_lock:
        executor, _tool_executor = _tool_executor, None
    if executor is not None:
        executor.shutdown(wait=wait)


REQUEST_PARAMS = {'model', 'temperature', 'top_p', 'json_format', 'max_tokens', 'thinking', 'effort'}


class Agent:
    '''An LLM agent that runs a step loop with tool calls.

    Instantiate with a name and an LLM interface, then call ``run()`` to schedule
    an execution. Between steps the agent drains broker feedback, so a run can be
    stopped, paused, resumed, retried or answered directly from any thread; tool
    calls can optionally be gated behind manual approvals, and nested calls are
    capped by ``max_call_depth``.

    Attributes:
        name (str): Human-readable agent name.
        code (str): Four-character code identifying this instance.
        llm (LLMInterface): Backend used for requests.
        system_prompt (str): Effective system prompt (base prompt plus group context).
        model (str): Model id sent with every request.
        messages (List[dict]): Shared conversation history in 'agent' mode.
        tools (Dict[str, Tool]): Tools exposed to the model, keyed by name.
        enabled_plugins (Set[str]): Plugin names allowed to observe this agent.
        plugin_config (Dict[str, Any]): Per-plugin configuration.
        llm_kwargs (Dict[str, Any]): Request parameters forwarded to the LLM.
        extra_data (Dict[str, Any]): Provider-specific request fields.
        description (str): Blurb used when the agent is exposed via ``as_tool()``.
        last_run_id (str): Most recent run id, kept after the run finishes.
        last_answer (str): Final answer text of that run.
        budget (Budget): Token/USD/seconds guard rail of this agent.
    '''

    def __init__(
        self,
        name: str,
        llm: LLMInterface,
        system_prompt: str = '',
        tools: Optional[List[Tool]] = None,
        model: Optional[str] = None,
        messages_mode: Literal['agent', 'run'] = 'agent',
        *,
        plugins: Optional[List[str]] = None,
        plugin_config: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        json_format: bool = False,
        max_tokens: Optional[int] = None,
        thinking: Literal['disabled', 'enabled', 'auto', 'disable', 'on', 'off', 'true', 'false'] = 'auto',
        effort: Literal['minimal', 'low', 'medium', 'high', 'xhigh'] = 'medium',
        extra_data: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        pause_timeout: Optional[float] = None,
        approval_timeout: Optional[float] = None,
        require_approval: Union[bool, Callable[[Any, Dict[str, Any]], bool]] = False,
        allow_concurrent_runs: bool = False,
        max_call_depth: Optional[int] = None,
        budget_tokens: Optional[int] = None,
        budget_usd: Optional[float] = None,
        budget_seconds: Optional[float] = None,
        enable_ask_user: bool = False,
    ):
        '''Create an agent.

        Args:
            name (str): Agent name, used in events and tool descriptions.
            llm (LLMInterface): The LLM backend to request from.
            system_prompt (str): Base system prompt; group collaboration context
                is appended to it.
            tools (Optional[List[Tool]]): Tools available to the model.
            model (Optional[str]): Model id; defaults to ``MODEL_ID`` from the
                environment, then to 'gpt-4o'.
            messages_mode (Literal['agent', 'run']): How history is scoped per run.
                'agent' (default) keeps one message list on the Agent and mutates it
                in place, which assumes a single active run; 'run' builds an
                isolated list per run and is safe for concurrent runs (pass custom
                history with ``run(messages=...)``).
            plugins (Optional[List[str]]): Plugin names enabled for this agent;
                None or an empty list enables no plugins.
            plugin_config (Optional[Dict[str, Any]]): Per-plugin configuration
                keyed by plugin name, read by ``Plugin.get_config()``. Stored in
                the broker run context and never sent to the LLM.
            temperature (float): Sampling temperature forwarded on every request.
            top_p (float): Nucleus sampling parameter forwarded on every request.
            json_format (bool): Ask the provider for a JSON response.
            max_tokens (Optional[int]): Response token cap; None leaves it to the
                provider.
            thinking (Literal[...]): Reasoning toggle forwarded to the provider.
            effort (Literal[...]): Reasoning effort forwarded to the provider.
            extra_data (Optional[Dict[str, Any]]): Provider-specific request
                fields merged into the request.
            description (Optional[str]): Blurb used by ``as_tool()``; defaults to
                the first line of ``system_prompt``, then to 'Agent: <name>'.
            pause_timeout (Optional[float]): Seconds a 'control:pause' may park the
                run before it self-resumes; None parks until resumed or stopped.
            approval_timeout (Optional[float]): Seconds an approval request may sit
                in the pool before it is auto-denied; None waits indefinitely.
            require_approval (Union[bool, Callable[[Any, Dict[str, Any]], bool]]):
                Manual-confirmation policy. True parks every tool call in the
                broker's approval pool; a callable(tool, args) decides per call.
                Individual tools opt in with ``Tool(require_approval=True)``.
            allow_concurrent_runs (bool): By default an Agent refuses a second
                ``run()`` while one is in flight, because both would share
                ``messages`` and ``last_run_id``. Set True to allow overlapping
                executions; ``messages_mode='run'`` keeps their histories separate.
            max_call_depth (Optional[int]): Hard cap on nested
                ``call_agent()``/``call_group()``/``as_tool()`` depth, from
                ``ORCHIFY_MAX_CALL_DEPTH`` when None (default 8; 0 disables the
                cap). A call that would exceed the cap, or re-enter an entity
                already on the call path, raises ``CallBlocked`` and emits
                'run:call:blocked' instead of nesting forever.
            budget_tokens (Optional[int]): Token cap, from ORCHIFY_BUDGET_TOKENS.
            budget_usd (Optional[float]): Cost cap, from ORCHIFY_BUDGET_USD; needs
                a price for the run's model (see ``orchify.budget`` /
                ``ORCHIFY_PRICE_TABLE``).
            budget_seconds (Optional[float]): Wall-clock cap, from
                ORCHIFY_BUDGET_SECONDS. When a cap is exceeded the run emits
                'run:budget:exceeded' and stops cleanly before the next request.
            enable_ask_user (bool): Register the ready-made ``ask_user`` tool (see
                ``ask_user_tool()``) so the model can ask the human operator an
                open question and wait for a free-text answer.

        Note:
            Request parameters (temperature, top_p, json_format, max_tokens,
            thinking, effort) are stored on the agent and forwarded to
            ``llm.request()`` on every run; values passed to ``run()`` override
            them for that run.
        '''
        self.name = name
        self.code = safecode(length=4)
        self.llm = llm
        self.system_prompt = system_prompt
        self._base_system_prompt = system_prompt
        self._group_context: Optional[str] = None
        self._description = description
        self.model = model or req_model('gpt-4o')
        self.messages_mode = messages_mode

        self.pause_timeout = (
            pause_timeout if pause_timeout is not None
            else _read_timeout_env('ORCHIFY_PAUSE_TIMEOUT')
        )
        self.approval_timeout = (
            approval_timeout if approval_timeout is not None
            else _read_timeout_env('ORCHIFY_APPROVAL_TIMEOUT')
        )
        self.require_approval = require_approval
        self.last_run_id: str = ''
        self.last_answer: str = ''
        self.allow_concurrent_runs = allow_concurrent_runs
        self._run_slot = threading.Lock()
        self._active_run_id: str = ''

        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        if enable_ask_user and 'ask_user' not in self.tools:
            self.tools['ask_user'] = self.ask_user_tool()
        self.enabled_plugins: Set[str] = set(plugins) if plugins else set()
        self.plugin_config: Dict[str, Any] = dict(plugin_config or {})

        self.messages: List[Dict[str, Any]] = [
            {'role': 'system', 'content': system_prompt} if system_prompt else {}
        ]
        self.messages = [msg for msg in self.messages if msg]

        self.llm_kwargs: Dict[str, Any] = {
            'model': self.model,
            'temperature': temperature,
            'top_p': top_p,
            'json_format': json_format,
            'max_tokens': max_tokens,
            'thinking': thinking,
            'effort': effort,
        }
        self.extra_data: Dict[str, Any] = dict(extra_data or {})

        self._call_chain: Optional[CallChain] = None
        self._call_depth: int = 0
        self._call_path: List[str] = []
        self.max_call_depth = (
            max_call_depth if max_call_depth is not None
            else _read_int_env('ORCHIFY_MAX_CALL_DEPTH', 8)
        )
        self.budget = Budget(
            tokens=(
                budget_tokens if budget_tokens is not None
                else _read_int_env('ORCHIFY_BUDGET_TOKENS', 0)
            ),
            usd=(
                budget_usd if budget_usd is not None
                else _read_positive_float_env('ORCHIFY_BUDGET_USD')
            ),
            seconds=(
                budget_seconds if budget_seconds is not None
                else _read_positive_float_env('ORCHIFY_BUDGET_SECONDS')
            ),
        )

        try:
            from .manager import orchify_manager
            orchify_manager.register(self)
        except Exception:
            pass

    def _build_run_messages(
        self,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        system = {'role': 'system', 'content': self.system_prompt} if self.system_prompt else None

        if self.messages_mode == 'agent' and messages is None:
            return self.messages

        base: List[Dict[str, Any]] = []
        if system is not None:
            base.append(dict(system))

        if messages:
            for m in messages:
                m = dict(m)
                if m.get('role') == 'system':
                    continue
                base.append(m)

        return base

    def add_tool(self, tool: Tool, replace: bool = True) -> None:
        '''Attach a Tool to this agent. Raises if the name collides and replace=False.'''
        if not isinstance(tool, Tool):
            raise TypeError(f'add_tool() requires a Tool, got {type(tool)}')
        if not replace and tool.name in self.tools:
            raise ValueError(f"Tool \'{tool.name}\' is already registered on agent '{self.name}'.")
        self.tools[tool.name] = tool

    def remove_tool(self, name: str) -> bool:
        '''Detach a Tool by name. Returns True if a tool was removed.'''
        return self.tools.pop(name, None) is not None

    def enable_plugin(self, name: str) -> None:
        '''Enable a plugin for this agent.'''
        self.enabled_plugins.add(name)

    def disable_plugin(self, name: str) -> None:
        '''Disable a plugin for this agent.'''
        self.enabled_plugins.discard(name)

    def set_plugins(self, plugins: Optional[List[str]]) -> None:
        '''Set the list of enabled plugins. None or empty = no plugins.'''
        self.enabled_plugins = set(plugins) if plugins else set()

    def _apply_system_prompt(self) -> None:
        '''Recompute the effective system prompt.

        Rebuilds ``system_prompt`` from the base prompt plus the group collaboration
        context and, in 'agent' messages_mode, writes it back into the leading
        system message of the shared history.

        Without that write-back the appended collaboration context never reaches the
        LLM: in 'agent' mode the system message is captured once at ``__init__``
        time and reused for every request.
        '''
        parts = [self._base_system_prompt] if self._base_system_prompt else []
        if self._group_context:
            parts.append(f'Group Collaboration Context:\n{self._group_context}')
        self.system_prompt = '\n\n'.join(parts)

        if self.messages_mode != 'agent':
            return
        rest = [m for m in self.messages if isinstance(m, dict) and m.get('role') != 'system']
        self.messages[:] = ([{'role': 'system', 'content': self.system_prompt}] if self.system_prompt else []) + rest

    def set_system_prompt(self, system_prompt: str) -> None:
        '''Replace the agent's system prompt, keeping the message history in sync.'''
        self._base_system_prompt = system_prompt or ''
        self._apply_system_prompt()

    @property
    def description(self) -> str:
        '''Short description used when this agent is exposed as a tool.'''
        if self._description:
            return self._description
        base = (self._base_system_prompt or '').strip()
        if base:
            first_line = base.splitlines()[0].strip()
            if first_line:
                return first_line
        return f'Agent: {self.name}'

    @description.setter
    def description(self, value: Optional[str]) -> None:
        '''Set an explicit description for this agent.'''
        self._description = value

    @property
    def current_run_id(self) -> str:
        '''Return the id of this agent's live run.

        ``last_run_id`` is remembered after a run finishes, so it is only a
        reliable identity while that run is still registered.

        Returns:
            str: The live run id, or '' when the agent has no active run.
        '''
        run_id = self._active_run_id or self.last_run_id
        return run_id if run_id and run_id in orchify_broker.run_contexts else ''

    def needs_approval(self, tool: Optional[Tool], args: Optional[Dict[str, Any]] = None) -> bool:
        '''Decide whether one tool call must be manually confirmed.

        Checks the Agent's ``require_approval`` policy (bool or callable) and then
        the Tool's own ``require_approval`` flag.

        Args:
            tool (Optional[Tool]): The tool about to run.
            args (Optional[Dict[str, Any]]): Arguments the tool would receive.

        Returns:
            bool: True when the call must park in the approval pool; False for
            unknown tools and tools that are not gated.
        '''
        policy = self.require_approval
        if callable(policy):
            try:
                if policy(tool, args or {}):
                    return True
            except Exception:
                return True
        elif policy:
            return True
        return bool(getattr(tool, 'needs_approval', False))

    def _configure_with_tools(self, tools: List[Tool], extra_context: Optional[Dict[str, Any]] = None) -> None:
        '''Configure the agent with extra tools and collaboration context.

        Idempotent: re-wiring the same agent refreshes the collaboration context
        instead of appending a duplicate block every time.

        Args:
            tools (List[Tool]): Tools contributed by the surrounding group.
            extra_context (Optional[Dict[str, Any]]): Values rendered into the
                group collaboration context.
        '''
        for tool in tools:
            if isinstance(tool, Tool):
                self.add_tool(tool, replace=True)

        if extra_context:
            self._group_context = '\n'.join(f'{k}: {v}' for k, v in extra_context.items())
        else:
            self._group_context = None
        self._apply_system_prompt()

    def _outgoing_path(self, callee_label: str) -> List[str]:
        '''Validate one outgoing call and return the callee's call path.

        Args:
            callee_label (str): Label of the entity being called, for example
                'Group:g1'.

        Returns:
            List[str]: The path the callee will run on.

        Raises:
            CallBlocked: When the call would exceed ``max_call_depth`` or re-enter
                an entity already on the call path; 'run:call:blocked' is emitted
                first, so A -> B -> A can never nest until the process dies.
        '''
        try:
            return plan_call(f'Agent:{self.name}', self._call_path, callee_label, self.max_call_depth)
        except CallBlocked as exc:
            orchify_broker.emit(RuntimeEvent(
                agent_name=self.name,
                agent_code=self.code,
                run_id=self.current_run_id,
                turn_id='',
                event_type='run:call:blocked',
                payload={
                    'reason': exc.reason,
                    'caller': exc.caller,
                    'callee': exc.callee,
                    'call_path': list(exc.path),
                    'call_depth': max(0, len(exc.path) - 1),
                    'max_call_depth': self.max_call_depth,
                },
            ))
            raise

    def call_agent(self, agent: 'Agent', user_input: str, **kwargs) -> str:
        '''Call another Agent and track the call chain.

        Args:
            agent (Agent): Target agent to call.
            user_input (str): Input to send to the target agent.
            **kwargs: Additional parameters for the target agent's run.

        Returns:
            str: The run_id of the target agent's execution.
        '''
        path = self._outgoing_path(f'Agent:{agent.name}')

        if self._call_chain is None:
            self._call_chain = CallChain(root_caller=f'Agent:{self.name}')

        self._call_chain.add_frame(
            caller=f'Agent:{self.name}',
            callee=f'Agent:{agent.name}',
            call_type='agent',
            input_data=user_input
        )

        agent._call_chain = self._call_chain
        agent._call_path = path
        agent._call_depth = max(0, len(path) - 1)

        orchify_broker.emit(RuntimeEvent(
            agent_name=self.name,
            agent_code=self.code,
            run_id=self.current_run_id,
            turn_id=kwargs.get('run_id') or kwargs.get('turn_id') or '',
            event_type='agent:call',
            payload={
                'caller': self.name,
                'callee': agent.name,
                'call_type': 'agent',
                'input': user_input,
                'call_depth': self._call_depth
            }
        ))

        return agent.run(user_input, **kwargs)

    def call_group(self, group: 'Group', user_input: str, **kwargs) -> str:
        '''Call a Group and track the call chain.

        Args:
            group (Group): Target group to call.
            user_input (str): Input to send to the group.
            **kwargs: Additional parameters for the group's run.

        Returns:
            str: The run_id of the group's execution.
        '''
        path = self._outgoing_path(f'Group:{group.name}')

        if self._call_chain is None:
            self._call_chain = CallChain(root_caller=f'Agent:{self.name}')

        self._call_chain.add_frame(
            caller=f'Agent:{self.name}',
            callee=f'Group:{group.name}',
            call_type='group',
            input_data=user_input
        )

        if hasattr(group, '_call_chain'):
            group._call_chain = self._call_chain
        group._call_path = path
        orchify_broker.emit(RuntimeEvent(
            agent_name=self.name,
            agent_code=self.code,
            run_id=self.current_run_id,
            turn_id=kwargs.get('run_id') or kwargs.get('turn_id') or '',
            event_type='agent:call',
            payload={
                'caller': self.name,
                'callee': group.name,
                'call_type': 'group',
                'input': user_input,
                'call_depth': self._call_depth
            }
        ))

        return group.run(user_input, **kwargs)

    def get_call_chain(self) -> Optional[CallChain]:
        '''Returns the current call chain for this agent.'''
        return self._call_chain

    def get_call_chain_str(self) -> str:
        '''Returns a human-readable string of the call chain.'''
        if self._call_chain:
            return self._call_chain.get_chain_str()
        return f'Agent:{self.name} (no call chain)'

    def as_tool(self, caller_agent: Optional['Agent'] = None) -> Tool:
        '''Wrap this Agent into a Tool, with optional call chain tracking.

        Args:
            caller_agent (Optional[Agent]): When given, call chain tracking is set
                up so that executing the tool registers itself in the caller's call
                chain, preserving full call-chain visibility while composing agents
                through tools.

        Returns:
            Tool: A Tool whose execution runs this agent.
        '''
        agent_self = self
        _caller = caller_agent

        def agent_runner(**kwargs):
            '''Run this agent with the input supplied by the model.

            Args:
                **kwargs: Tool arguments; ``input`` (or ``query``) carries the request,
                    every other key is forwarded to ``Agent.run()``.

            Returns:
                str: The run_id of the agent's execution, as returned by ``run()``.
            '''
            user_input = kwargs.get('input', kwargs.get('query', ''))

            if _caller is not None:
                path = _caller._outgoing_path(f'Agent:{agent_self.name}')
                if _caller._call_chain is None:
                    _caller._call_chain = CallChain(root_caller=f'Agent:{_caller.name}')

                _caller._call_chain.add_frame(
                    caller=f'Agent:{_caller.name}',
                    callee=f'Agent:{agent_self.name}',
                    call_type='agent',
                    input_data=user_input
                )

                agent_self._call_chain = _caller._call_chain
                agent_self._call_path = path
                agent_self._call_depth = max(0, len(path) - 1)

                orchify_broker.emit(RuntimeEvent(
                    agent_name=_caller.name,
                    agent_code=_caller.code,
                    run_id=_caller.current_run_id,
                    turn_id=kwargs.get('run_id') or kwargs.get('turn_id') or '',
                    event_type='agent:call',
                    payload={
                        'caller': _caller.name,
                        'callee': agent_self.name,
                        'call_type': 'agent',
                        'input': user_input,
                        'call_depth': _caller._call_depth
                    }
                ))

            run_id = agent_self.run(user_input, **kwargs)
            return run_id

        agent_runner.__name__ = self.name
        agent_runner.__doc__ = self.description or f'Call agent \'{self.name}\' to perform a task.'

        from inspect import Parameter, Signature
        params = [
            Parameter(name='input', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        ]
        agent_runner.__signature__ = Signature(params)

        tool = Tool(func=agent_runner, name=self.name, description=self.description)
        return tool

    def ask_user_tool(self) -> Tool:
        '''Build the ready-made ``ask_user`` tool for this agent.

        The model calls it with a ``question``; the agent parks the call in the
        broker's pool with kind='question' (event 'run:approval:required') and waits
        for a human answer delivered through ``OrchifyManager.answer()`` /
        ``orchify_broker.answer_approval()``. The answer text becomes the tool
        result, so the model continues with real information instead of a guess; an
        unanswered question is reported back as "[no answer]" rather than failing
        the run. The tool function itself is never executed.

        Attach it with ``agent.ask_user_tool()``, or pass ``enable_ask_user=True`` to
        the constructor to register it automatically.

        Returns:
            Tool: The question tool, ready to add with ``add_tool()``.
        '''
        def ask_user(question: str) -> str:
            '''Ask the human operator an open question and wait for the answer.

            Args:
                question (str): The question to put to the human operator.

            Returns:
                str: The operator's free-text answer, or the "[no answer]" placeholder
                when nobody answered before the approval timeout.
            '''
            return question

        ask_user.__name__ = 'ask_user'
        ask_user.__doc__ = (
            'Ask the human operator an open question and wait for a free-text '
            'answer. Use it when the request is ambiguous or when it needs '
            'information only the user can provide.\n\n'
            'Args:\n'
            '    question (str): the question to put to the human operator.'
        )

        from inspect import Parameter, Signature
        ask_user.__signature__ = Signature([
            Parameter(name='question', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        ])

        return Tool(
            func=ask_user,
            name='ask_user',
            description=ask_user.__doc__,
            parameters=[{
                'name': 'question',
                'kind': 'positional or keyword',
                'default': 'N/A',
                'annotation': 'str',
                'description': 'The question to put to the human operator.',
                'required': True,
            }],
            ask_question=True,
        )

    def run(
        self,
        user_input: str,
        max_steps: Union[int, Dict[str, Any]] = 5,
        run_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **llm_kwargs
    ) -> str:
        '''Schedule a task on the WebBackend asyncio loop.

        This is a non-blocking synchronous entry point that avoids creating an OS
        thread per run. For convenience the second argument may also be a dict of
        run options (max_steps / run_id / messages / any LLM request kwargs):
        ``agent.run('hi', {'max_steps': 3, 'temperature': 0.2})``.

        Args:
            user_input (str): The user message that starts the run.
            max_steps (Union[int, Dict[str, Any]]): Step budget, or a dict of run
                options.
            run_id (Optional[str]): Explicit run handle.
            turn_id (Optional[str]): Legacy alias of ``run_id``.
            messages (Optional[List[Dict[str, Any]]]): Custom starting history for
                this run (mainly with ``messages_mode='run'``).
            **llm_kwargs: Per-run LLM request parameters, overriding the agent's.

        Returns:
            str: The ``run_id`` of this execution: the unique handle used to target
            the run with feedback (stop / pause / approval) and carried as
            ``run_id`` on every emitted event. Note that ``run_id`` identifies the
            whole execution while a *turn* is a single LLM round trip inside it,
            labelled f'{run_id}#{turn}'.
        '''
        if isinstance(max_steps, dict):
            opts = dict(max_steps)
            max_steps = opts.pop('max_steps', 5)
            if 'run_id' in opts:
                run_id = opts.pop('run_id')
            if 'turn_id' in opts:
                turn_id = opts.pop('turn_id')
            if 'messages' in opts:
                messages = opts.pop('messages')
            llm_kwargs.update(opts)
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError(f'max_steps must be a positive integer, got {max_steps!r}')
        run_id = run_id or turn_id or f'{self.name}#{self.code}_{safecode()}'
        if not self.allow_concurrent_runs and not self._run_slot.acquire(blocking=False):
            raise RuntimeError(
                f"Agent '{self.name}' is already running "
                f"(run_id={self._active_run_id!r}). An Agent keeps its messages "
                f"and last_run_id on the instance, so overlapping runs would "
                f"clobber each other: wait for the current run (Agent.wait()), "
                f"stop it (Agent.cancel()), use a separate Agent instance, or "
                f"construct this one with allow_concurrent_runs=True."
            )
        self._active_run_id = run_id
        self.last_run_id = run_id
        orchify_broker.start_task(run_id, None, context={
            'run_id': run_id,
            'task_name': run_id,
            'agent_name': self.name,
            'agent_code': self.code,
            'started_at': time.time(),
            'enabled_plugins': self.enabled_plugins,
            'plugin_config': self.plugin_config,
        })
        try:
            asyncio.run_coroutine_threadsafe(
                self._async_thread_process(
                    user_input,
                    max_steps,
                    run_id,
                    messages=messages,
                    llm_kwargs=llm_kwargs,
                ),
                orchify_web_backend._loop
            )
        except Exception:
            self._release_run_slot()
            raise
        return run_id

    def _release_run_slot(self) -> None:
        '''Release the one-run-at-a-time slot (safe to call more than once).'''
        self._active_run_id = ''
        try:
            self._run_slot.release()
        except RuntimeError:
            pass

    def is_running(self) -> bool:
        '''True while this agent has a run in flight.'''
        return self._run_slot.locked()

    def wait(self, run_id: Optional[str] = None, timeout: Optional[float] = None) -> bool:
        '''Block the CALLING thread until a run finishes.

        Defaults to the most recent run started by this agent.

        Args:
            run_id (Optional[str]): Run to wait for; defaults to ``last_run_id``.
            timeout (Optional[float]): Seconds to wait; resolved from
                ORCHIFY_RUN_TIMEOUT when not given (None = wait forever). Pass 0 to
                poll without waiting.

        Returns:
            bool: True when the run finished (including one that had already
            finished), False for an unknown run or on timeout.

        Raises:
            RuntimeError: When called from the orchestrator loop thread, which has
                to stay free to execute the run itself.
        '''
        target = run_id or self.last_run_id
        if not target:
            raise ValueError('wait() needs a run_id (this agent has not run yet).')
        if timeout is None:
            timeout = _read_timeout_env('ORCHIFY_RUN_TIMEOUT')
        if threading.current_thread() is orchify_web_backend._thread:
            raise RuntimeError(
                'Agent.wait() would deadlock: it is being called from the '
                'orchestrator loop thread, which must run the target itself. '
                'Use the broker\'s run:finish hook instead.'
            )
        return orchify_broker.wait_run(target, timeout=timeout)

    def cancel(self, run_id: Optional[str] = None) -> bool:
        '''Ask a run to stop.

        Args:
            run_id (Optional[str]): Run to stop; defaults to the live run, then to
                ``last_run_id``.

        Returns:
            bool: True when the stop request was queued; False when the run is
            unknown or already finished. Unlike ``wait()``, this is safe from any
            thread: it never blocks on the run.
        '''
        target = run_id or self.last_run_id
        if not target:
            return False
        return orchify_broker.feedback('control:stop', run_id=target)

    async def _async_thread_process(
        self,
        user_input: str,
        max_steps: int,
        run_id: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        orchify_broker.start_task(run_id, asyncio.current_task())

        run_gate = get_run_gate()
        gate_held = False
        if run_gate.enabled:
            queued = run_gate.would_block()
            wait_started = time.monotonic()
            if queued:
                orchify_broker.emit(RuntimeEvent(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=make_turn_id(run_id, 1),
                    event_type='run:queued',
                    payload={
                        'run_id': run_id,
                        'limit': run_gate.limit,
                        'in_use': run_gate.in_use,
                    },
                    run_id=run_id,
                    source='runtime',
                ))
            gate_held = await run_gate.acquire(
                stop=lambda: orchify_broker.pop_feedback(run_id, {'control:stop'}) is not None
            )
            if gate_held and queued:
                orchify_broker.emit(RuntimeEvent(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=make_turn_id(run_id, 1),
                    event_type='run:dequeued',
                    payload={
                        'run_id': run_id,
                        'wait_seconds': round(time.monotonic() - wait_started, 3),
                    },
                    run_id=run_id,
                    source='runtime',
                ))

        try:
            if run_gate.enabled and not gate_held:
                orchify_broker.emit(RuntimeEvent(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=make_turn_id(run_id, 1),
                    event_type='run:finish',
                    payload={
                        'total_steps': 0,
                        'usage': {
                            'completion_tokens': 0,
                            'prompt_tokens': 0,
                            'prompt_cache_hit_tokens': 0,
                            'prompt_cache_miss_tokens': 0,
                            'total_tokens': 0,
                        },
                        'runs': [],
                        'stopped': True,
                        'overridden': False,
                        'answer': '',
                        'budget': None,
                    },
                    run_id=run_id,
                    source='runtime',
                ))
                return

            generator = self._run(
                user_input=user_input,
                max_steps=max_steps,
                messages=messages,
                llm_kwargs=llm_kwargs,
                run_id=run_id,
            )
            async for e in generator:
                e.run_id = e.run_id or run_id
                e.source = e.source or 'runtime'
                orchify_broker.emit(e)
        except Exception:
            import traceback
            error_text = traceback.format_exc()
            logger.error('[Agent Error] run=%s: %s', run_id, error_text)
            orchify_broker.emit(RuntimeEvent(
                agent_name=self.name,
                agent_code=self.code,
                turn_id=make_turn_id(run_id, 1),
                event_type='agent:abort',
                payload={'error': error_text},
                run_id=run_id,
                source='runtime',
            ))
        finally:
            if gate_held:
                run_gate.release()
            orchify_broker.finish_task(run_id)
            self._release_run_slot()

    async def _run(
        self,
        user_input: str,
        max_steps: int = 5,
        messages: Optional[List[Dict[str, Any]]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        run_id: str = '',
    ) -> AsyncGenerator[Union[AgentEvent, ToolEvent, RuntimeEvent], None]:
        working_messages = self._build_run_messages(messages)
        turn_id = make_turn_id(run_id, 1) if run_id else safecode(length=4)

        def drain_feedback(types: Optional[set] = None) -> list:
            '''Pop queued broker feedback for this run.

            Args:
                types (Optional[set]): Accepted ``ftype`` values; None accepts any.

            Returns:
                list: The consumed ``EventFeedback`` items, in queue order.
            '''
            return orchify_broker.drain_feedback(run_id, types) if run_id else []

        def pop_feedback(types: Optional[set] = None):
            '''Consume at most one queued feedback item (non-blocking).'''
            return orchify_broker.pop_feedback(run_id, types) if run_id else None

        def apply_update(payload: dict) -> None:
            '''Apply one 'control:update_messages' payload to the history.

            Args:
                payload (dict): ``op`` is 'append' (extend with ``messages``),
                    'truncate' (keep the first ``keep`` messages) or 'replace'
                    (substitute ``messages`` wholesale).
            '''
            op = payload.get('op')
            msgs = payload.get('messages') or []
            clean = [dict(m) for m in msgs if isinstance(m, dict)]
            if op == 'append':
                working_messages.extend(clean)
            elif op == 'replace':
                system = working_messages[:1] if working_messages and working_messages[0].get('role') == 'system' else []
                replaced = [m for m in clean if m.get('role') != 'system']
                working_messages[:] = system + replaced
            elif op == 'truncate':
                keep = max(0, int(payload.get('keep', 0)))
                system = []
                rest = working_messages
                if rest and rest[0].get('role') == 'system':
                    system = [rest[0]]
                    rest = rest[1:]
                working_messages[:] = system + rest[:keep]

        stopped = False
        override_content: Optional[str] = None
        last_answer: str = ''
        req_overrides: Dict[str, Any] = {}
        suspend_result: Dict[str, Any] = {'status': 'resumed'}
        pause_timeout = self.pause_timeout
        approval_timeout = self.approval_timeout

        async def suspend(step: int, reason: str = 'control:pause') -> AsyncGenerator[RuntimeEvent, None]:
            '''Park this run until a matching 'control:resume'.

            The waiter is armed BEFORE 'run:paused' is emitted, so a resume sent in
            reaction to that event is delivered instead of being lost (the original
            lost-resume race). The status is published on suspend_result['status'] as
            'resumed' | 'stopped' | 'timeout'.

            Args:
                step (int): Current step, reported in the pause events.
                reason (str): Label recorded on the pause event.

            Yields:
                RuntimeEvent: The 'run:paused' event; 'run:resumed' is generated by the
                caller from the published status.
            '''
            suspend_result['status'] = 'resumed'
            if run_id:
                orchify_broker.arm_resume(run_id)
            yield RuntimeEvent(
                agent_name=self.name,
                agent_code=self.code,
                turn_id=turn_id,
                event_type='run:paused',
                payload={'run_id': run_id, 'step': step, 'reason': reason, 'timeout': pause_timeout},
                run_id=run_id,
                turn=step
            )
            status = await orchify_broker.wait_resume(run_id, timeout=pause_timeout)
            suspend_result['status'] = status
            yield RuntimeEvent(
                agent_name=self.name,
                agent_code=self.code,
                turn_id=turn_id,
                event_type='run:resumed',
                payload={'run_id': run_id, 'step': step, 'status': status},
                run_id=run_id,
                turn=step
            )

        async def request_approval(
            step: int,
            tool_id: str,
            tool_name: str,
            args: Dict[str, Any],
        ) -> AsyncGenerator[Union[ApprovalEvent, RuntimeEvent], None]:
            '''Park one tool call in the broker's approval pool.

            Emits the corresponding events and publishes the decision on
            approval_result['approved'] / ['reason'] / ['timed_out'].

            Fail-closed: an unresolved request (timeout, stop while parked, or a
            released entry) is treated as a DENIAL, never as consent.

            Args:
                step (int): Current step, reported in the approval events.
                tool_id (str): Model-assigned tool call id.
                tool_name (str): Name of the tool being confirmed.
                args (dict): Arguments the tool would be called with.

            Yields:
                Union[ApprovalEvent, RuntimeEvent]: The 'run:approval:required' and
                'run:approval:resolved' events.
            '''
            approval_result.update({'approved': False, 'reason': '', 'timed_out': False})
            approval = orchify_broker.request_approval(
                run_id,
                kind='tool',
                tool_id=tool_id,
                tool_name=tool_name,
                args=args,
                timeout=approval_timeout,
                turn_id=turn_id or '',
                agent_name=self.name,
                agent_code=self.code,
                step=step,
            )
            yield ApprovalEvent.build_required(
                agent_name=self.name,
                agent_code=self.code,
                turn_id=turn_id,
                approval_id=approval.approval_id,
                kind='tool',
                tool_id=tool_id,
                tool_name=tool_name,
                args=args,
                step=step,
                timeout=approval_timeout,
            )
            try:
                resolved = await orchify_broker.wait_approval(
                    approval.approval_id,
                    timeout=approval_timeout,
                    run_id=run_id,
                )
                approval_result.update({
                    'approved': bool(resolved.approved),
                    'reason': resolved.reason,
                    'timed_out': bool(resolved.timed_out),
                })
            except ApprovalTimeout:
                reason = f'approval timed out after {approval_timeout}s'
                orchify_broker.resolve_approval(
                    approval.approval_id, False, reason=reason, timed_out=True
                )
                approval_result.update({'approved': False, 'reason': reason, 'timed_out': True})
            except KeyError:
                reason = 'approval request was withdrawn'
                approval_result.update({'approved': False, 'reason': reason})
            except RunStopped:
                stop_state['stopped'] = True
                approval_result.update({'approved': False, 'reason': 'run stopped'})
            yield ApprovalEvent.build_resolved(
                agent_name=self.name,
                agent_code=self.code,
                turn_id=turn_id,
                approval_id=approval.approval_id,
                kind='tool',
                tool_id=tool_id,
                tool_name=tool_name,
                approved=approval_result['approved'],
                reason=approval_result['reason'],
                timed_out=approval_result['timed_out'],
                step=step,
            )
            orchify_broker.release_approval(approval.approval_id)

        async def request_question(
            step: int,
            tool_id: str,
            tool_name: str,
            args: Dict[str, Any],
        ) -> AsyncGenerator[Union[ApprovalEvent, RuntimeEvent], None]:
            '''Park one open question in the broker's pool.

            Emits the matching events. The human's free-text answer is published on
            question_result['text'] ('' when nobody answered).

            Unlike a tool approval this is not fail-closed on content: an unanswered
            question simply returns "no answer" to the model, while a stop while parked
            still ends the run.

            Args:
                step (int): Current step, reported in the approval events.
                tool_id (str): Model-assigned tool call id.
                tool_name (str): Name of the question tool.
                args (dict): Arguments of the question call; ``question`` (or
                    ``prompt``) carries the text.

            Yields:
                Union[ApprovalEvent, RuntimeEvent]: The 'run:approval:required' and
                'run:approval:resolved' events.
            '''
            question = ''
            if isinstance(args, dict):
                question = args.get('question') or args.get('prompt') or ''
            if not question:
                question = json.dumps(args, ensure_ascii=False) if args else ''
            question_result.update({
                'question': str(question),
                'text': '',
                'timed_out': False,
                'approval_id': '',
            })
            approval = orchify_broker.ask_question(
                run_id,
                question,
                tool_id=tool_id,
                tool_name=tool_name,
                timeout=approval_timeout,
                turn_id=turn_id or '',
                agent_name=self.name,
                agent_code=self.code,
                step=step,
            )
            question_result['approval_id'] = approval.approval_id
            yield ApprovalEvent.build_required(
                agent_name=self.name,
                agent_code=self.code,
                turn_id=turn_id,
                approval_id=approval.approval_id,
                kind='question',
                tool_id=tool_id,
                tool_name=tool_name,
                args={'question': question_result['question']},
                step=step,
                timeout=approval_timeout,
            )
            try:
                resolved = await orchify_broker.wait_approval(
                    approval.approval_id,
                    timeout=approval_timeout,
                    run_id=run_id,
                )
                answer = str(getattr(resolved, 'answer', '') or '')
                if not answer and resolved.approved and resolved.reason:
                    answer = str(resolved.reason)
                question_result.update({
                    'text': answer,
                    'timed_out': bool(resolved.timed_out),
                })
            except ApprovalTimeout:
                question_result['timed_out'] = True
                orchify_broker.resolve_approval(
                    approval.approval_id, False,
                    reason=f'question timed out after {approval_timeout}s',
                    timed_out=True
                )
            except KeyError:
                question_result['text'] = ''
            except RunStopped:
                stop_state['stopped'] = True
            yield ApprovalEvent.build_resolved(
                agent_name=self.name,
                agent_code=self.code,
                turn_id=turn_id,
                approval_id=approval.approval_id,
                kind='question',
                tool_id=tool_id,
                tool_name=tool_name,
                approved=bool(question_result['text']),
                reason=question_result['text'],
                timed_out=question_result['timed_out'],
                step=step,
                answer=question_result['text'],
            )
            orchify_broker.release_approval(approval.approval_id)

        approval_result: Dict[str, Any] = {'approved': False, 'reason': '', 'timed_out': False}
        question_result: Dict[str, Any] = {
            'question': '', 'text': '', 'timed_out': False, 'approval_id': ''
        }
        stop_state: Dict[str, bool] = {'stopped': False}

        accumulated_usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'prompt_cache_hit_tokens': 0,
            'prompt_cache_miss_tokens': 0,
            'total_tokens': 0
        }

        budget = self.budget
        run_started = time.monotonic()
        budget_result: Dict[str, Any] = {'exceeded': False, 'reasons': [], 'violations': []}

        def current_model() -> str:
            '''The model this step will use (overrides win over the constructor).'''
            return (
                req_overrides.get('model')
                or (llm_kwargs or {}).get('model')
                or self.llm_kwargs.get('model')
                or self.model
            )

        def stop_requested() -> bool:
            '''Consume a pending control:stop (non-blocking); True when found.'''
            return bool(run_id) and orchify_broker.pop_feedback(run_id, {'control:stop'}) is not None

        def budget_info(model_name: str) -> Dict[str, Any]:
            '''Serialisable budget state for the run so far.'''
            return budget.info(accumulated_usage, time.monotonic() - run_started, model_name)

        def budget_event(model_name: str, step: int) -> Optional[RuntimeEvent]:
            '''A 'run:budget:exceeded' event when a cap is exceeded, else None.'''
            if not budget.enabled():
                return None
            violations = budget.violations(
                accumulated_usage, time.monotonic() - run_started, model_name
            )
            if not violations:
                return None
            budget_result.update({
                'exceeded': True,
                'violations': violations,
                'reasons': [v.get('reason') for v in violations],
            })
            return RuntimeEvent(
                agent_name=self.name,
                agent_code=self.code,
                turn_id=turn_id,
                event_type='run:budget:exceeded',
                payload={
                    'run_id': run_id,
                    'step': step,
                    'violations': violations,
                    'reasons': budget_result['reasons'],
                    **budget_info(model_name),
                },
                run_id=run_id,
                turn=step
            )

        budget.require_price(self.model)

        yield RuntimeEvent(
            agent_name=self.name,
            agent_code=self.code,
            turn_id=turn_id,
            event_type='run:start',
            payload={'user_input': user_input},
            turn=1
        )

        working_messages.append({'role': 'user', 'content': user_input})

        yield AgentEvent(
            agent_name=self.name,
            agent_code=self.code,
            turn_id=turn_id,
            event_type='agent:start',
            content=user_input,
            turn=1
        )

        runs = []
        executed_step = 1

        for step in range(1, max_steps + 1):
            executed_step = step
            if run_id:
                turn_id = make_turn_id(run_id, step)
                orchify_broker.register_turn(turn_id, run_id)

            if step > 1:
                yield RuntimeEvent(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=turn_id,
                    event_type='run:next',
                    payload={'step': step},
                    turn=step
                )

            pause_requested = False
            for fb in drain_feedback({'control:stop', 'control:override_answer', 'control:update_messages', 'control:switch_model', 'control:pause'}):
                if fb.ftype == 'control:stop':
                    stopped = True
                elif fb.ftype == 'control:override_answer':
                    override_content = fb.payload.get('content', '')
                elif fb.ftype == 'control:update_messages':
                    apply_update(fb.payload)
                elif fb.ftype == 'control:switch_model':
                    req_overrides['model'] = fb.payload.get('model', '')
                elif fb.ftype == 'control:pause':
                    pause_requested = True
            if stopped:
                break

            if pause_requested:
                async for ev in suspend(step, 'control:pause'):
                    yield ev
                if suspend_result['status'] == 'stopped':
                    stopped = True
                    break

            if override_content is not None:
                last_answer = override_content
                working_messages.append({'role': 'assistant', 'content': override_content})
                yield AgentEvent(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=turn_id,
                    turn=step,
                    event_type='agent:answer',
                    content=override_content,
                )
                break

            budget_now = budget_event(current_model(), step)
            if budget_now is not None:
                stopped = True
                yield budget_now
                break

            tools_payload = [t.info for t in self.tools.values()] if self.tools else None

            def build_request():
                '''Merge the agent, run and feedback request parameters.

                Later sources win: the agent's stored ``llm_kwargs``, then the per-run
                kwargs, then values pushed through 'control:switch_model'. Keys outside
                REQUEST_PARAMS are routed into ``extra_data`` instead of being passed as
                provider keywords.

                Returns:
                    dict: The keyword arguments for ``llm.request()``.
                '''
                req_kwargs = {**self.llm_kwargs, **(llm_kwargs or {}), **req_overrides}

                extra_data = dict(self.extra_data or {})
                run_extra = req_kwargs.pop('extra_data', None)
                if run_extra:
                    extra_data.update(run_extra)
                for k in [k for k in req_kwargs if k not in REQUEST_PARAMS]:
                    extra_data[k] = req_kwargs.pop(k)
                return req_kwargs, extra_data

            final_status = None
            retries = 0
            stream_aborted = False
            llm_gate = get_llm_gate()
            llm_rpm = get_llm_rpm()
            while True:
                if llm_rpm.enabled and not await llm_rpm.acquire(stop=stop_requested):
                    stopped = True
                    break
                req_kwargs, extra_data = build_request()
                assembler = ToolEventAssembler()

                llm_held = True
                if llm_gate.enabled:
                    llm_held = await llm_gate.acquire(stop=stop_requested)
                    if not llm_held:
                        stopped = True
                        stream_aborted = True
                        break

                response_gen = self.llm.request(
                    messages=working_messages,
                    model=req_kwargs.pop('model', self.model),
                    tools=tools_payload,
                    extra_data=extra_data,
                    scope=self.name,
                    _enabled_plugins=self.enabled_plugins,
                    **req_kwargs,
                )
                try:
                    last_control_check = time.monotonic()
                    async for response in response_gen:
                        now = time.monotonic()
                        if now - last_control_check >= POLL_INTERVAL:
                            last_control_check = now
                            if pop_feedback({'control:stop'}) is not None:
                                stopped = True
                                stream_aborted = True
                                break
                            if pop_feedback({'control:pause'}) is not None:
                                async for ev in suspend(step, 'control:pause'):
                                    yield ev
                                if suspend_result['status'] == 'stopped':
                                    stopped = True
                                    stream_aborted = True
                                    break
                        if response.is_final:
                            final_status = response.final_status
                            yield AgentEvent.build_from_resp(
                                response,
                                agent_name=self.name,
                                agent_code=self.code,
                                turn_id=turn_id,
                                turn=step
                            )
                        else:
                            chunk = response.current_chunk
                            if chunk:
                                if chunk.is_assembly_tool:
                                    tool_events = assembler.build(
                                        response,
                                        agent_name=self.name,
                                        agent_code=self.code,
                                        turn_id=turn_id,
                                        turn=step
                                    )
                                    for te in tool_events: yield te
                                else:
                                    yield AgentEvent.build_from_resp(
                                        response,
                                        agent_name=self.name,
                                        agent_code=self.code,
                                        turn_id=turn_id,
                                        turn=step
                                    )
                    if stream_aborted:
                        try:
                            await response_gen.aclose()
                        except Exception:
                            pass
                    break
                except Exception:
                    retry_fb = None
                    for fb in drain_feedback({'control:retry'}):
                        if fb.ftype == 'control:retry':
                            retry_fb = fb
                    if retry_fb is not None and retries < int(retry_fb.payload.get('max_retries', 1)):
                        retries += 1
                        req_overrides.update(retry_fb.payload.get('kwargs', {}) or {})
                        continue
                    raise
                finally:
                    if llm_held and llm_gate.enabled:
                        llm_gate.release()

            if stream_aborted or stopped:
                break
            if final_status is None: break

            inject_map = {}
            deny_map = {}
            pause_requested = False
            for fb in drain_feedback({'control:stop', 'control:inject_tool_result', 'control:deny_tool', 'control:pause', 'control:update_messages'}):
                if fb.ftype == 'control:stop':
                    stopped = True
                elif fb.ftype == 'control:inject_tool_result':
                    key = fb.payload.get('tool_id') or fb.payload.get('tool_name')
                    if key:
                        inject_map[key] = fb.payload.get('result', '')
                elif fb.ftype == 'control:deny_tool':
                    key = fb.payload.get('tool_id') or fb.payload.get('tool_name')
                    if key:
                        deny_map[key] = fb.payload.get('reason', 'denied')
                elif fb.ftype == 'control:pause':
                    pause_requested = True
                elif fb.ftype == 'control:update_messages':
                    apply_update(fb.payload)
            if stopped:
                break

            if pause_requested:
                async for ev in suspend(step, 'control:pause'):
                    yield ev
                if suspend_result['status'] == 'stopped':
                    stopped = True
                    break

            accumulated_usage['completion_tokens'] += final_status.completion_tokens
            accumulated_usage['prompt_tokens'] += final_status.prompt_tokens
            accumulated_usage['prompt_cache_hit_tokens'] += final_status.prompt_cache_hit_tokens
            accumulated_usage['prompt_cache_miss_tokens'] += final_status.prompt_cache_miss_tokens
            accumulated_usage['total_tokens'] += final_status.total_tokens

            assistant_msg = {
                'role': 'assistant',
                'content': final_status.content or None
            }

            if final_status.tool_calls:
                assistant_msg['tool_calls'] = final_status.tool_calls
            working_messages.append(assistant_msg)

            if final_status.content:
                last_answer = final_status.content

            runs.append({
                'step': step,
                'content': assistant_msg,
                'tool_calls': final_status.tool_calls
            })

            budget_now = budget_event(current_model(), step)
            if budget_now is not None:
                stopped = True
                yield budget_now
                break

            if not final_status.tool_calls: break

            loop = asyncio.get_running_loop()
            tasks = []
            completed_results = {}

            for tool_call in final_status.tool_calls:
                tool_id = tool_call.get('id') or ''
                tool_name = tool_call.get('function', {}).get('name') or ''
                args_str = tool_call.get('function', {}).get('arguments') or '{}'

                try:
                    args = json.loads(args_str) if args_str.strip() else {}
                except Exception:
                    args = {}

                yield ToolEvent.build_call_start(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=turn_id,
                    tool_id=tool_id,
                    tool_name=tool_name,
                    args=args,
                    turn=step
                )

                injected = inject_map.get(tool_id)
                if injected is None:
                    injected = inject_map.get(tool_name)
                if injected is not None:
                    completed_results[tool_id] = (tool_name, injected)
                    yield ToolEvent.build_call_finish(
                        agent_name=self.name,
                        agent_code=self.code,
                        turn_id=turn_id,
                        tool_id=tool_id,
                        tool_name=tool_name,
                        result=injected,
                        turn=step
                    )
                    continue

                denied = deny_map.get(tool_id)
                if denied is None:
                    denied = deny_map.get(tool_name)
                if denied is not None:
                    result = f'[denied] {denied}'
                    completed_results[tool_id] = (tool_name, result)
                    yield ToolEvent.build_call_finish(
                        agent_name=self.name,
                        agent_code=self.code,
                        turn_id=turn_id,
                        tool_id=tool_id,
                        tool_name=tool_name,
                        result=result,
                        turn=step
                    )
                    continue

                tool = self.tools.get(tool_name)

                if tool is not None and getattr(tool, 'ask_question', False):
                    async for ev in request_question(step, tool_id, tool_name, args):
                        yield ev
                    if stop_state['stopped']:
                        stopped = True
                        break
                    answer_text = question_result['text']
                    result = answer_text if answer_text else (
                        '[no answer] the user did not answer the question'
                    )
                    completed_results[tool_id] = (tool_name, result)
                    yield ToolEvent.build_call_finish(
                        agent_name=self.name,
                        agent_code=self.code,
                        turn_id=turn_id,
                        tool_id=tool_id,
                        tool_name=tool_name,
                        result=result,
                        payload={
                            'question': question_result['question'],
                            'answered': bool(answer_text),
                            'timed_out': bool(question_result['timed_out']),
                        },
                        turn=step
                    )
                    continue

                if self.needs_approval(tool, args) and tool is not None:
                    async for ev in request_approval(step, tool_id, tool_name, args):
                        yield ev
                    if stop_state['stopped']:
                        stopped = True
                        break
                    if not approval_result['approved']:
                        reason = approval_result['reason'] or 'denied by user'
                        result = f'[approval denied] {reason}'
                        completed_results[tool_id] = (tool_name, result)
                        yield ToolEvent.build_call_finish(
                            agent_name=self.name,
                            agent_code=self.code,
                            turn_id=turn_id,
                            tool_id=tool_id,
                            tool_name=tool_name,
                            result=result,
                            payload={'approved': False, 'timed_out': bool(approval_result['timed_out'])},
                            turn=step
                        )
                        continue

                def run_single_tool(t_tool=tool, t_name=tool_name, t_args=args, t_id=tool_id):
                    '''Execute one tool call and normalise its result.

                    Runs in the tool executor thread, so it must never raise: failures are
                    returned to the model as text.

                    Args:
                        t_tool (Tool): The tool to execute.
                        t_name (str): Tool name, for the error message.
                        t_args (dict): Keyword arguments for the tool.
                        t_id (str): Model-assigned tool call id.

                    Returns:
                        tuple: ``(tool_id, tool_name, result_text, cached)`` where
                        ``result_text`` is the tool's string result (non-strings are
                        JSON-encoded) or "Error executing tool '<name>': ..." on failure.
                    '''
                    if t_tool:
                        cached = False
                        try:
                            if isinstance(t_args, dict):
                                res = t_tool.execute(**t_args)
                            else:
                                res = t_tool.execute()

                            cached = bool(getattr(t_tool, 'last_cached', False))

                            if not isinstance(res, str):
                                res = json.dumps(res, ensure_ascii=False)
                        except Exception as e:
                            res = f"Error executing tool '{t_name}': {str(e)}"
                    else:
                        res = f"Tool '{t_name}' not found."
                    return t_id, t_name, res, cached

                future = loop.run_in_executor(get_tool_executor(), run_single_tool)
                tasks.append(future)

            for future in asyncio.as_completed(tasks):
                t_id, t_name, result, cached = await future
                completed_results[t_id] = (t_name, result)

                yield ToolEvent.build_call_finish(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=turn_id,
                    tool_id=t_id,
                    tool_name=t_name,
                    result=result,
                    payload={'cached': cached},
                    turn=step
                )

            for tool_call in final_status.tool_calls:
                tool_id = tool_call.get('id') or ''
                tool_name = tool_call.get('function', {}).get('name') or ''
                t_name, result = completed_results.get(tool_id, (tool_name, f"Tool '{tool_name}' execution failed."))

                working_messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_id,
                    'name': tool_name,
                    'content': result
                })

            if stopped:
                break

        if not stopped:
            if pop_feedback({'control:stop'}) is not None:
                stopped = True
            if not stopped and pop_feedback({'control:pause'}) is not None:
                async for ev in suspend(executed_step, 'control:pause'):
                    yield ev
                if suspend_result['status'] == 'stopped':
                    stopped = True

        self.last_answer = last_answer
        yield RuntimeEvent(
            agent_name=self.name,
            agent_code=self.code,
            turn_id=turn_id,
            event_type='run:finish',
            payload={
                'total_steps': executed_step,
                'usage': accumulated_usage,
                'runs': runs,
                'stopped': stopped,
                'overridden': override_content is not None,
                'answer': last_answer,
                'budget': {
                    **budget_info(current_model()),
                    'exceeded': budget_result['exceeded'],
                    'reasons': budget_result['reasons'],
                    'violations': budget_result['violations'],
                },
            },
            turn=executed_step
        )
