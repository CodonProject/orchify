from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Literal
import asyncio
from concurrent.futures import ThreadPoolExecutor

from orchify.llm    import LLMInterface
from orchify.tool   import Tool
from orchify.event  import AgentEvent, ToolEvent, RuntimeEvent
from orchify.utils  import safecode
from orchify.env    import req_model
from orchify.broker import orchify_broker
from orchify.backend import orchify_web_backend

import json


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
        model: str = req_model('gpt-4o'),
        messages_mode: Literal['agent', 'run'] = 'agent',
        *,
        temperature: float = 0.7,
        top_p: float = 1.0,
        json_format: bool = False,
        max_tokens: Optional[int] = None,
        thinking: Literal['disabled', 'enabled', 'auto', 'disable', 'on', 'off', 'true', 'false'] = 'auto',
        effort: Literal['minimal', 'low', 'medium', 'high', 'xhigh'] = 'medium',
        extra_data: Optional[Dict[str, Any]] = None,
        **llm_kwargs,
    ):
        '''
        messages_mode controls how conversation history is scoped per run:
          - 'agent' (default): messages are shared on the Agent and mutated in place.
            Intended for the case where only one run is active at a time.
          - 'run': each run gets its own isolated message list built at run time.
            Safe for concurrent runs of the same agent; pass custom history via run(messages=...).

        LLM request params (temperature, top_p, json_format, max_tokens, thinking,
        effort, extra_data, plus any other kwargs) are stored and forwarded to
        llm.request() on every run. The same values passed to run() override them.
        '''
        self.name = name
        self.code = safecode(length=4)
        self.llm = llm
        self.system_prompt = system_prompt
        self.model = model
        self.messages_mode = messages_mode
        
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        
        self.messages: List[Dict[str, Any]] = [
            {'role': 'system', 'content': system_prompt} if system_prompt else {}
        ]
        self.messages = [msg for msg in self.messages if msg]

        known: Dict[str, Any] = {}
        extra: Dict[str, Any] = dict(extra_data or {})
        for k, v in llm_kwargs.items():
            if k in REQUEST_PARAMS:
                known[k] = v
            else:
                extra[k] = v

        self.llm_kwargs: Dict[str, Any] = {
            'model': model,
            'temperature': temperature,
            'top_p': top_p,
            'json_format': json_format,
            'max_tokens': max_tokens,
            'thinking': thinking,
            'effort': effort,
            **known,
        }
        self.extra_data: Dict[str, Any] = extra
    
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
    
    def run(
        self,
        user_input: str,
        max_steps: int = 5,
        turn_id: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **llm_kwargs
    ) -> str:
        '''
        Synchronously schedules the agent's task to be executed on the WebBackend asyncio loop.
        This provides a non-blocking synchronous entry point, avoiding OS thread creation per run.

        Returns the turn_id (str) for this run, which can be tracked via emitted events.
        '''
        turn_id = turn_id or safecode(length=4)
        task_name = f'{self.name}#{self.code}_{safecode()}'
        orchify_broker.start_task(task_name, None)
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
        current_task = asyncio.current_task()
        orchify_broker.start_task(task_name, current_task)
        
        generator = self._run(
            user_input=user_input,
            max_steps=max_steps,
            turn_id=turn_id,
            messages=messages,
            llm_kwargs=llm_kwargs,
        )
        try:
            async for e in generator: 
                orchify_broker.emit(e)
        except Exception:
            import traceback
            error_text = traceback.format_exc()
            print(f"[Agent Error] task={task_name}: {error_text}", flush=True)
            orchify_broker.emit(RuntimeEvent(
                agent_name=self.name,
                agent_code=self.code,
                turn_id=turn_id or '',
                event_type='agent:abort',
                payload={'error': error_text},
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
    ) -> AsyncGenerator[Union[AgentEvent, ToolEvent, RuntimeEvent], None]:
        turn_id = turn_id or safecode(length=4)
        working_messages = self._build_run_messages(messages)

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
            
            tools_payload = [t.info for t in self.tools.values()] if self.tools else None
            
            req_kwargs = {**self.llm_kwargs, **(llm_kwargs or {})}
            req_kwargs.pop('messages', None)
            req_kwargs.pop('tools', None)

            extra_data = dict(self.extra_data or {})
            run_extra = req_kwargs.pop('extra_data', None)
            if run_extra:
                extra_data.update(run_extra)
            for k, v in list(req_kwargs.items()):
                if k not in REQUEST_PARAMS:
                    extra_data[k] = req_kwargs.pop(k)

            response_gen = self.llm.request(
                messages=working_messages,
                model=req_kwargs.pop('model', self.model),
                tools=tools_payload,
                extra_data=extra_data,
                **req_kwargs,
            )
            
            final_status = None
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
                            tool_events = ToolEvent.build_from_resp(
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
            
            if final_status is None: break
            
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
                'status': final_status
            })
            
            if not final_status.tool_calls: break

            loop = asyncio.get_running_loop()
            tasks = []

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

                tool = self.tools.get(tool_name)
                
                def run_single_tool(t_tool=tool, t_name=tool_name, t_args=args, t_id=tool_id):
                    if t_tool:
                        try:
                            if isinstance(t_args, dict):
                                res = t_tool.execute(**t_args)
                            else:
                                res = t_tool.execute()
                            
                            if not isinstance(res, str):
                                res = json.dumps(res, ensure_ascii=False)
                        except Exception as e:
                            res = f"Error executing tool '{t_name}': {str(e)}"
                    else:
                        res = f"Tool '{t_name}' not found."
                    return t_id, t_name, res

                future = loop.run_in_executor(tool_executor, run_single_tool)
                tasks.append(future)

            completed_results = {}
            for future in asyncio.as_completed(tasks):
                t_id, t_name, result = await future
                completed_results[t_id] = (t_name, result)
                
                yield ToolEvent.build_call_finish(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=turn_id,
                    tool_id=t_id,
                    tool_name=t_name,
                    result=result,
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
            
            runs.append({
                'step': step,
                'content': assistant_msg,
                'tool_calls': final_status.tool_calls
            })
        
        yield RuntimeEvent(
            agent_name=self.name,
            agent_code=self.code,
            turn_id=turn_id,
            event_type='run:finish',
            payload={
                'total_steps': executed_step,
                'usage': accumulated_usage,
                'runs': runs
            },
            turn=executed_step
        )
