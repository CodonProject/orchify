from typing import List, Dict, Any, Optional, Set, Union, AsyncGenerator, Literal, Callable, TYPE_CHECKING
import asyncio
from concurrent.futures import ThreadPoolExecutor

from orchify.llm        import LLMInterface
from orchify.tool       import Tool
from orchify.event      import AgentEvent, ToolEvent, RuntimeEvent, ToolEventAssembler, ApprovalEvent
from orchify.utils      import safecode
from orchify.env        import req_model
from orchify.broker     import orchify_broker, ApprovalTimeout, RunStopped, POLL_INTERVAL, make_turn_id
from orchify.backend    import orchify_web_backend
from orchify.call_chain import CallChain

if TYPE_CHECKING:
    from orchify.group import Group

import json
import os
import time
import threading


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


_tool_executor: Optional[ThreadPoolExecutor] = None
_tool_executor_lock = threading.Lock()


def get_tool_executor() -> ThreadPoolExecutor:
    '''
    Return the process-wide executor that runs Tool calls off the event loop.

    Created lazily on first use (not at import time) so importing orchify never
    spawns threads, and so the worker count honours ORCHIFY_TOOL_WORKERS.
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


# Named parameters of LLMInterface.request(); anything else is routed to extra_data.
REQUEST_PARAMS = {'model', 'temperature', 'top_p', 'json_format', 'max_tokens', 'thinking', 'effort'}


class Agent:
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
    ):
        '''
        messages_mode controls how conversation history is scoped per run:
          - 'agent' (default): messages are shared on the Agent and mutated in place.
            Intended for the case where only one run is active at a time.
          - 'run': each run gets its own isolated message list built at run time.
            Safe for concurrent runs of the same agent; pass custom history via run(messages=...).

        plugins: list of plugin names enabled for this agent. Default is empty list (no plugins).
          Pass a list of plugin names to enable specific plugins.
          None = no plugins (same as empty list).

        plugin_config: per-plugin configuration dict keyed by plugin name.
          Plugins read their config via Plugin.get_config(key). This config is
          stored in the broker task context and never sent to the LLM API.

        LLM request params (temperature, top_p, json_format, max_tokens, thinking,
        effort) are stored and forwarded to llm.request() on every run. The same
        values passed to run() override them. Provider-specific fields belong in
        extra_data.

        description: short blurb used when this agent is wrapped via as_tool().
          Defaults to the first line of system_prompt, then to 'Agent: <name>'.

        pause_timeout: seconds a 'control:pause' may park the run before it
          self-resumes (None = park until resumed or stopped).

        approval_timeout: seconds an approval request may sit in the pool before
          it is auto-denied (None = wait indefinitely).

        require_approval: manual-confirmation policy. True parks every tool call
          in the broker's approval pool; a callable(tool, args) -> bool decides
          per call. Individual tools opt in with Tool(require_approval=True).

        allow_concurrent_runs: by default an Agent refuses a second run() while
          one is still in flight, because both would share `messages` and
          `last_run_id`. Set True if you really want overlapping executions
          (messages_mode='run' keeps their histories separate).
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
        # run_id of the most recent execution started by run()
        self.last_run_id: str = ''
        # final answer of the most recent execution (what voting/competition and
        # user code can read without hooking events)
        self.last_answer: str = ''
        # one execution at a time per instance: a second run() on the SAME object
        # would otherwise share `messages`/`last_run_id` with the one in flight.
        self.allow_concurrent_runs = allow_concurrent_runs
        self._run_slot = threading.Lock()
        self._active_run_id: str = ''

        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        # Default: no plugins enabled unless explicitly passed
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
        
        # Call chain tracking
        self._call_chain: Optional[CallChain] = None
        self._call_depth: int = 0
    
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
                    continue  # system cannot be customized; the agent's prompt always wins
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
        '''
        Recompute self.system_prompt from the base prompt plus the group
        collaboration context, and — in 'agent' messages_mode — write it back
        into the shared history's leading system message.

        Without the write-back the appended collaboration context never reaches
        the LLM: in 'agent' mode the system message is captured once at
        __init__ time and reused for every request.
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
        self._description = value

    @property
    def current_run_id(self) -> str:
        '''The run_id of this agent's live run, or '' when it has none.

        `last_run_id` is remembered after a run finishes, so it is only a
        reliable identity while that run is still registered.
        '''
        run_id = self._active_run_id or self.last_run_id
        return run_id if run_id and run_id in orchify_broker.run_contexts else ''

    def needs_approval(self, tool: Optional[Tool], args: Optional[Dict[str, Any]] = None) -> bool:
        '''
        Decide whether one tool call must be manually confirmed.

        Checks, in order: the Agent's require_approval policy (bool or callable)
        and the Tool's own require_approval flag. Returns False for unknown tools.
        '''
        policy = self.require_approval
        if callable(policy):
            try:
                if policy(tool, args or {}):
                    return True
            except Exception:
                # a broken policy must not silently allow the call
                return True
        elif policy:
            return True
        return bool(getattr(tool, 'needs_approval', False))

    def _configure_with_tools(self, tools: List[Tool], extra_context: Optional[Dict[str, Any]] = None) -> None:
        '''Configures the agent with additional tools and context for group collaboration.

        Idempotent: re-wiring the same agent refreshes the collaboration context
        instead of appending a duplicate block every time.
        '''
        for tool in tools:
            if isinstance(tool, Tool):
                self.add_tool(tool, replace=True)

        if extra_context:
            self._group_context = '\n'.join(f'{k}: {v}' for k, v in extra_context.items())
        else:
            self._group_context = None
        self._apply_system_prompt()

    def call_agent(self, agent: 'Agent', user_input: str, **kwargs) -> str:
        '''
        Calls another Agent and tracks the call chain.
        
        Args:
            agent: The target agent to call.
            user_input: The input to send to the target agent.
            **kwargs: Additional parameters for the target agent's run.
            
        Returns:
            str: The run_id of the target agent's execution.
        '''
        # Initialize call chain if not exists
        if self._call_chain is None:
            self._call_chain = CallChain(root_caller=f'Agent:{self.name}')
        
        # Add frame to call chain
        self._call_chain.add_frame(
            caller=f'Agent:{self.name}',
            callee=f'Agent:{agent.name}',
            call_type='agent',
            input_data=user_input
        )
        
        # Transfer call chain to target agent
        agent._call_chain = self._call_chain
        agent._call_depth = self._call_depth + 1
        
        # Emit call event
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
        '''
        Calls a Group and tracks the call chain.
        
        Args:
            group: The target group to call.
            user_input: The input to send to the group.
            **kwargs: Additional parameters.
            
        Returns:
            str: The run_id of the group's execution.
        '''
        # Initialize call chain if not exists
        if self._call_chain is None:
            self._call_chain = CallChain(root_caller=f'Agent:{self.name}')
        
        # Add frame to call chain
        self._call_chain.add_frame(
            caller=f'Agent:{self.name}',
            callee=f'Group:{group.name}',
            call_type='group',
            input_data=user_input
        )
        
        # Emit call event
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
        '''Wraps this Agent into a Tool, with optional call chain tracking.

        Args:
            caller_agent: If provided, call chain tracking is set up so that
                when this tool is executed, it registers itself in the caller's
                call chain. Useful for composing agents via tools while
                preserving full call-chain visibility.

        Returns:
            A Tool whose execute() runs this agent.
        '''
        agent_self = self
        _caller = caller_agent

        def agent_runner(**kwargs):
            user_input = kwargs.get('input', kwargs.get('query', ''))

            if _caller is not None:
                if _caller._call_chain is None:
                    _caller._call_chain = CallChain(root_caller=f'Agent:{_caller.name}')

                _caller._call_chain.add_frame(
                    caller=f'Agent:{_caller.name}',
                    callee=f'Agent:{agent_self.name}',
                    call_type='agent',
                    input_data=user_input
                )

                agent_self._call_chain = _caller._call_chain
                agent_self._call_depth = _caller._call_depth + 1

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

    def run(
        self,
        user_input: str,
        max_steps: Union[int, Dict[str, Any]] = 5,
        run_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **llm_kwargs
    ) -> str:
        '''
        Synchronously schedules the agent's task to be executed on the WebBackend asyncio loop.
        This provides a non-blocking synchronous entry point, avoiding OS thread creation per run.

        For convenience, the second argument may also be a dict of run options
        (max_steps / run_id / messages / any LLM request kwargs):

            agent.run('hi', {'max_steps': 3, 'temperature': 0.2})

        Returns the `run_id` (str) of this execution: the unique handle used to
        target the run with feedback (stop / pause / approval) and carried as
        `run_id` on every emitted event. Pass your own `run_id` to control the
        handle; `turn_id` is accepted as a legacy alias for it.

        Note that `run_id` identifies the whole execution, while a *turn* is a
        single LLM round trip inside it, labelled f'{run_id}#{turn}'.
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
        # `turn_id` is kept as a legacy alias: callers used to pass the label of
        # the execution here, and run() returned it back to them.
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
        '''
        Block the CALLING thread until a run finishes.

        Defaults to the most recent run started by this agent. Returns True when
        the run is finished (including one that had already finished), False for
        an unknown run or on timeout. The timeout is resolved from
        ORCHIFY_RUN_TIMEOUT when not given (None = wait forever).

        Must not be called from the orchestrator loop thread, which has to stay
        free to execute the run itself. Pass 0 to poll without waiting.
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
        '''
        Ask a run to stop (control:stop). Returns False when the run is unknown
        or already finished. Unlike Agent.wait(), this is safe from any thread:
        it never blocks on the run.
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
        
        generator = self._run(
            user_input=user_input,
            max_steps=max_steps,
            messages=messages,
            llm_kwargs=llm_kwargs,
            run_id=run_id,
        )
        try:
            async for e in generator:
                e.run_id = e.run_id or run_id
                e.source = e.source or 'runtime'
                orchify_broker.emit(e)
        except Exception:
            import traceback
            error_text = traceback.format_exc()
            print(f'[Agent Error] run={run_id}: {error_text}', flush=True)
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
        # `turn_id` is recomputed per step below: one turn = one LLM round trip.
        turn_id = make_turn_id(run_id, 1) if run_id else safecode(length=4)

        def drain_feedback(types: Optional[set] = None) -> list:
            return orchify_broker.drain_feedback(run_id, types) if run_id else []

        def pop_feedback(types: Optional[set] = None):
            '''Consume at most one queued feedback item (non-blocking).'''
            return orchify_broker.pop_feedback(run_id, types) if run_id else None

        def apply_update(payload: dict) -> None:
            op = payload.get('op')
            msgs = payload.get('messages') or []
            # tolerate malformed entries instead of blowing up the whole run
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
            '''
            Park this run until a matching 'control:resume'.

            The waiter is armed BEFORE 'run:paused' is emitted, so a resume sent
            in reaction to that event is delivered instead of being lost (the
            original lost-resume race). The status is published on
            suspend_result['status'] as 'resumed' | 'stopped' | 'timeout'.
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
            '''
            Park one tool call in the broker's approval pool and emit the
            corresponding events. Publishes the decision on
            approval_result['approved'] / ['reason'] / ['timed_out'].

            Fail-closed: an unresolved request (timeout, stop while parked, or a
            released entry) is treated as a DENIAL, never as consent.
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
                # the run was asked to stop while parked: deny the call AND let
                # the caller end the run instead of continuing with a denial
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

        approval_result: Dict[str, Any] = {'approved': False, 'reason': '', 'timed_out': False}
        # set by request_approval() when a control:stop lands on a parked approval
        stop_state: Dict[str, bool] = {'stopped': False}

        accumulated_usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'prompt_cache_hit_tokens': 0,
            'prompt_cache_miss_tokens': 0,
            'total_tokens': 0
        }

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
            # one turn = one LLM round trip: label it as '<run_id>#<step>' and
            # index it so feedback addressed at a single turn finds this run
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

            # checkpoint 1: pre-request feedback (including pause)
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
                # plugin-provided answer: skip the LLM call entirely
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
            
            tools_payload = [t.info for t in self.tools.values()] if self.tools else None

            def build_request():
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
            while True:
                req_kwargs, extra_data = build_request()
                assembler = ToolEventAssembler()

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
                        # Mid-stream checkpoint: honour stop/pause while the model is
                        # still streaming instead of only after the full response.
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
                        # release the upstream stream before leaving the run
                        try:
                            await response_gen.aclose()
                        except Exception:
                            pass
                    break
                except Exception:
                    # control:retry feedback re-issues this request with overrides
                    retry_fb = None
                    for fb in drain_feedback({'control:retry'}):
                        if fb.ftype == 'control:retry':
                            retry_fb = fb
                    if retry_fb is not None and retries < int(retry_fb.payload.get('max_retries', 1)):
                        retries += 1
                        req_overrides.update(retry_fb.payload.get('kwargs', {}) or {})
                        continue
                    raise

            if stream_aborted or stopped:
                break
            if final_status is None: break

            # checkpoint 2: post-stream feedback (stop / inject / deny / pause / update)
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
                    # feedback-injected result: skip actual execution
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
                    # feedback-denied tool: skip execution, feed denial back
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

                if self.needs_approval(tool, args) and tool is not None:
                    # manual confirmation gate: park this call in the approval pool
                    async for ev in request_approval(step, tool_id, tool_name, args):
                        yield ev
                    if stop_state['stopped']:
                        # a stop landed on the parked approval: end the run
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

        # Final checkpoint: a pause/stop that arrived while the last step was
        # executing still applies, so the run cannot race past it into finish.
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
                'answer': last_answer
            },
            turn=executed_step
        )
