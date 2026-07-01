from agentforge.llm.base import Response
from typing import Literal, Optional

import json


EVENT_TYPES = Literal[
    'agent:start',
    'agent:reason:step',
    'agent:reason:finish',
    'agent:answer',
    'agent:finish',
    'tool:assembly:start',
    'tool:assembly:step',
    'tool:assembly:finish',
    'tool:call:start',
    'tool:call:finish',
]


class BaseEvent:
    def __init__(
        self,
        agent_name: str,
        agent_code: str,
        turn_id: str,
        event_type: EVENT_TYPES,
        payload: dict = None,
    ):
        self.agent_name = agent_name
        self.agent_code = agent_code
        self.turn_id = turn_id
        self.event_type = event_type
        self.payload = payload or {}


class AgentEvent(BaseEvent):
    def __init__(
        self,
        agent_name: str,
        agent_code: str,
        turn_id: str,
        event_type: EVENT_TYPES,
        content: str = '',
        payload: dict = None,
    ):
        super().__init__(agent_name, agent_code, turn_id, event_type, payload=payload)
        self.content = content
    
    @staticmethod
    def build_from_resp(response: Response, agent_name: str, agent_code: str, turn_id: str) -> 'AgentEvent':
        if response.is_final:
            return AgentEvent(
                agent_name=agent_name,
                agent_code=agent_code,
                turn_id=turn_id,
                event_type='agent:finish',
                content=response.final_status.content,
                payload={
                    'completion_tokens': response.final_status.completion_tokens,
                    'prompt_tokens': response.final_status.prompt_tokens,
                    'prompt_cache_hit_tokens': response.final_status.prompt_cache_hit_tokens,
                    'prompt_cache_miss_tokens': response.final_status.prompt_cache_miss_tokens,
                    'total_tokens': response.final_status.total_tokens,
                }
            )
        
        if response.current_chunk.is_cot_end:
            return AgentEvent(
                agent_name=agent_name,
                agent_code=agent_code,
                turn_id=turn_id,
                event_type='agent:reason:finish',
                content=response.current_chunk.content,
                payload={}
            )
        
        if not response.current_chunk.is_assembly_tool:
            return AgentEvent(
                agent_name=agent_name,
                agent_code=agent_code,
                turn_id=turn_id,
                event_type='agent:reason:step' if response.current_chunk.is_cot else 'agent:answer',
                content=response.current_chunk.content,
                payload={}
            )
        
        raise ValueError('Response must have either current_chunk or final_status.')


class ToolEvent(BaseEvent):
    
    _seen_tools = set()       # (turn_id, tool_id)
    _finished_tools = set()   # (turn_id, tool_id)
    _last_args_len = {}

    def __init__(
        self,
        agent_name: str,
        agent_code: str,
        turn_id: str,
        event_type: EVENT_TYPES,
        payload: dict = None,
        tool_id: str = '',
        tool_name: str = '',
        chunk_arg: str = '',
    ):
        super().__init__(agent_name, agent_code, turn_id, event_type, payload=payload)
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.chunk_arg = chunk_arg
        self.args: Optional[dict] = None
    
    @staticmethod
    def build_from_resp(response: Response, agent_name: str, agent_code: str, turn_id: str) -> list['ToolEvent']:
        if response.is_final: 
            return []
        if not response.current_chunk or not response.current_chunk.is_assembly_tool: 
            return []
        
        results = []
        total_tool_calls = response.current_chunk.total_tool_call or []
        for index, tool_call in enumerate(total_tool_calls):
            
            tool_id = tool_call.get('id') or f"idx_{index}"
            tool_name = tool_call.get('function', {}).get('name', '')
            arguments = tool_call.get('function', {}).get('arguments', '')
            state_key = (turn_id, tool_id)
            
            if state_key not in ToolEvent._seen_tools:
                ToolEvent._seen_tools.add(state_key)
                ToolEvent._last_args_len[state_key] = 0
                results.append(
                    ToolEvent(
                        agent_name=agent_name,
                        agent_code=agent_code,
                        turn_id=turn_id,
                        event_type='tool:assembly:start',
                        tool_id=tool_call.get('id', ''),
                        tool_name=tool_name
                    )
                )
                
            if state_key in ToolEvent._finished_tools: continue
            
            last_len = ToolEvent._last_args_len.get(state_key, 0)
            chunk_arg = arguments[last_len:]
            ToolEvent._last_args_len[state_key] = len(arguments)

            parsed_args = None
            is_json_valid = False
            if arguments.strip():
                try:
                    parsed_args = json.loads(arguments)
                    is_json_valid = True
                except json.JSONDecodeError: pass
            if is_json_valid:
                ToolEvent._finished_tools.add(state_key)
                event = ToolEvent(
                    agent_name=agent_name,
                    agent_code=agent_code,
                    turn_id=turn_id,
                    event_type='tool:assembly:finish',
                    tool_id=tool_call.get('id', ''),
                    tool_name=tool_name,
                    chunk_arg=chunk_arg
                )
                event.args = parsed_args
                results.append(event)
            else:
                if chunk_arg:
                    results.append(
                        ToolEvent(
                            agent_name=agent_name,
                            agent_code=agent_code,
                            turn_id=turn_id,
                            event_type='tool:assembly:step',
                            tool_id=tool_call.get('id', ''),
                            tool_name=tool_name,
                            chunk_arg=chunk_arg
                        )
                    )
        
        return results
    
    @staticmethod
    def build_call_start(
        agent_name: str,
        agent_code: str,
        turn_id: str,
        tool_id: str,
        tool_name: str,
        args: Optional[dict] = None,
    ) -> 'ToolEvent':
        event = ToolEvent(
            agent_name=agent_name,
            agent_code=agent_code,
            turn_id=turn_id,
            event_type='tool:call:start',
            tool_id=tool_id,
            tool_name=tool_name,
        )
        event.args = args
        return event
    
    @staticmethod
    def build_call_finish(
        agent_name: str,
        agent_code: str,
        turn_id: str,
        tool_id: str,
        tool_name: str,
        result: str = '',
        payload: dict = None,
    ) -> 'ToolEvent':
        event_payload = payload or {}
        if 'result' not in event_payload:
            event_payload['result'] = result
        return ToolEvent(
            agent_name=agent_name,
            agent_code=agent_code,
            turn_id=turn_id,
            event_type='tool:call:finish',
            tool_id=tool_id,
            tool_name=tool_name,
            payload=event_payload,
        )