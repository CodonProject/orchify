from typing import List, Dict, Any, Optional, Set, Union, AsyncGenerator, Literal, TYPE_CHECKING
import asyncio
from concurrent.futures import ThreadPoolExecutor

from orchify.llm        import LLMInterface
from orchify.tool       import Tool
from orchify.event      import AgentEvent, ToolEvent, RuntimeEvent, ToolEventAssembler
from orchify.utils      import safecode
from orchify.env        import req_model
from orchify.broker     import orchify_broker
from orchify.backend    import orchify_web_backend
from orchify.call_chain import CallChain

if TYPE_CHECKING:
    from orchify.group import Group

import json
import time


tool_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix='OrchifyToolExecutor')

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
        '''
        self.name = name
        self.code = safecode(length=4)
        self.llm = llm
        self.system_prompt = system_prompt
        self.model = model or req_model('gpt-4o')
        self.messages_mode = messages_mode

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

    def register_tool(self, tool: Tool, replace: bool = True) -> None:
        '''Attach a Tool to this agent. Raises if the name collides and replace=False.'''
        if not isinstance(tool, Tool):
            raise TypeError(f'register_tool() requires a Tool, got {type(tool)}')
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

    def _configure_with_tools(self, tools: List[Tool], extra_context: Optional[Dict[str, Any]] = None) -> None:
        '''Configures the agent with additional tools and context for group collaboration.'''
        for tool in tools:
            if isinstance(tool, Tool):
                self.register_tool(tool, replace=True)
        
        if extra_context:
            context_str = '\n'.join([f'{k}: {v}' for k, v in extra_context.items()])
            if self.system_prompt:
                self.system_prompt += f'\n\nGroup Collaboration Context:\n{context_str}'
            else:
                self.system_prompt = f'Group Collaboration Context:\n{context_str}'

    def call_agent(self, agent: 'Agent', user_input: str, **kwargs) -> str:
        '''
        Calls another Agent and tracks the call chain.
        
        Args:
            agent: The target agent to call.
            user_input: The input to send to the target agent.
            **kwargs: Additional parameters for the target agent's run.
            
        Returns:
            str: The turn_id of the target agent's execution.
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
            turn_id=kwargs.get('turn_id', ''),
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
            str: The turn_id of the group's execution.
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
            turn_id=kwargs.get('turn_id', ''),
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
                    turn_id=kwargs.get('turn_id', ''),
                    event_type='agent:call',
                    payload={
                        'caller': _caller.name,
                        'callee': agent_self.name,
                        'call_type': 'agent',
                        'input': user_input,
                        'call_depth': _caller._call_depth
                    }
                ))

            turn_id = agent_self.run(user_input, **kwargs)
            return turn_id

        agent_runner.__name__ = self.name
        agent_runner.__doc__ = self.description or f'Call agent \'{self.name}\' to perform a task.'

        from inspect import Parameter, Signature
        params = [
            Parameter(name='input', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        ]
        agent_runner.__signature__ = Signature(params)

        tool = Tool(func=agent_runner, name=self.name, description=f'Agent: {self.name}')
        return tool

    def run(
        self,
        user_input: str,
        max_steps: Union[int, Dict[str, Any]] = 5,
        turn_id: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **llm_kwargs
    ) -> str:
        '''
        Synchronously schedules the agent's task to be executed on the WebBackend asyncio loop.
        This provides a non-blocking synchronous entry point, avoiding OS thread creation per run.

        For convenience, the second argument may also be a dict of run options
        (max_steps / turn_id / messages / any LLM request kwargs):

            agent.run('hi', {'max_steps': 3, 'temperature': 0.2})

        Returns the turn_id (str) for this run, which can be tracked via emitted events.
        '''
        if isinstance(max_steps, dict):
            opts = dict(max_steps)
            max_steps = opts.pop('max_steps', 5)
            if 'turn_id' in opts:
                turn_id = opts.pop('turn_id')
            if 'messages' in opts:
                messages = opts.pop('messages')
            llm_kwargs.update(opts)
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError(f'max_steps must be a positive integer, got {max_steps!r}')
        turn_id = turn_id or safecode(length=4)
        task_name = f'{self.name}#{self.code}_{safecode()}'
        orchify_broker.start_task(task_name, None, context={
            'task_name': task_name,
            'agent_name': self.name,
            'agent_code': self.code,
            'turn_id': turn_id,
            'started_at': time.time(),
            'enabled_plugins': self.enabled_plugins,
            'plugin_config': self.plugin_config,
        })
        asyncio.run_coroutine_threadsafe(
            self._async_thread_process(
                user_input,
                max_steps,
                task_name,
                turn_id=turn_id,
                messages=messages,
                llm_kwargs=llm_kwargs,
            ),
            orchify_web_backend._loop
        )
        return turn_id
    
    async def _async_thread_process(
        self,
        user_input: str,
        max_steps: int,
        task_name: str,
        turn_id: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        orchify_broker.start_task(task_name, asyncio.current_task())
        
        generator = self._run(
            user_input=user_input,
            max_steps=max_steps,
            turn_id=turn_id,
            messages=messages,
            llm_kwargs=llm_kwargs,
            task_name=task_name,
        )
        try:
            async for e in generator:
                e.run_id = e.run_id or task_name
                e.source = e.source or 'runtime'
                orchify_broker.emit(e)
        except Exception:
            import traceback
            error_text = traceback.format_exc()
            print(f'[Agent Error] task={task_name}: {error_text}', flush=True)
            orchify_broker.emit(RuntimeEvent(
                agent_name=self.name,
                agent_code=self.code,
                turn_id=turn_id or '',
                event_type='agent:abort',
                payload={'error': error_text},
                run_id=task_name,
                source='runtime',
            ))
        finally:
            orchify_broker.finish_task(task_name)

    async def _run(
        self,
        user_input: str,
        max_steps: int = 5,
        turn_id: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        task_name: str = '',
    ) -> AsyncGenerator[Union[AgentEvent, ToolEvent, RuntimeEvent], None]:
        turn_id = turn_id or safecode(length=4)
        working_messages = self._build_run_messages(messages)

        def drain_feedback(types: Optional[set] = None) -> list:
            return orchify_broker.drain_feedback(task_name, types) if task_name else []

        def apply_update(payload: dict) -> None:
            op = payload.get('op')
            msgs = payload.get('messages') or []
            if op == 'append':
                working_messages.extend(dict(m) for m in msgs)
            elif op == 'replace':
                system = working_messages[:1] if working_messages and working_messages[0].get('role') == 'system' else []
                replaced = [dict(m) for m in msgs if dict(m).get('role') != 'system']
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
        req_overrides: Dict[str, Any] = {}

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

            if step > 1:
                yield RuntimeEvent(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=turn_id,
                    event_type='run:next',
                    payload={'step': step},
                    turn=step
                )

            # checkpoint 1: pre-request feedback
            for fb in drain_feedback({'control:stop', 'control:override_answer', 'control:update_messages', 'control:switch_model'}):
                if fb.ftype == 'control:stop':
                    stopped = True
                elif fb.ftype == 'control:override_answer':
                    override_content = fb.payload.get('content', '')
                elif fb.ftype == 'control:update_messages':
                    apply_update(fb.payload)
                elif fb.ftype == 'control:switch_model':
                    req_overrides['model'] = fb.payload.get('model', '')
            if stopped:
                break

            if override_content is not None:
                # plugin-provided answer: skip the LLM call entirely
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
                    async for response in response_gen:
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
                yield RuntimeEvent(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=turn_id,
                    event_type='run:paused',
                    payload={'task_name': task_name},
                    turn=step
                )
                await orchify_broker.wait_resume(task_name)
                yield RuntimeEvent(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=turn_id,
                    event_type='run:resumed',
                    payload={},
                    turn=step
                )
            
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

                future = loop.run_in_executor(tool_executor, run_single_tool)
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
                'overridden': override_content is not None
            },
            turn=executed_step
        )
