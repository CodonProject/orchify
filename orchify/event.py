from orchify.llm import Response
from typing import Literal, Optional
from dataclasses import dataclass

import json
import time


EVENT_TYPES = Literal[
    'agent:start',
    'agent:reason:step',
    'agent:reason:finish',
    'agent:answer',
    'agent:finish',
    'agent:abort',
    'tool:assembly:start',
    'tool:assembly:step',
    'tool:assembly:finish',
    'tool:call:start',
    'tool:call:finish',
    'run:start',
    'run:next',
    'run:paused',
    'run:resumed',
    'run:finish'
]

FEEDBACK_TYPES = Literal[
    'control:stop',
    'control:pause',
    'control:resume',
    'control:retry',
    'control:override_answer',
    'control:deny_tool',
    'control:inject_tool_result',
    'control:update_messages',
    'control:switch_model'
]


@dataclass
class EventFeedback:
    ftype: FEEDBACK_TYPES
    payload: dict = None

    def __post_init__(self):
        self.payload = self.payload or {}


class BaseEvent:
    def __init__(
        self,
        agent_name: str,
        agent_code: str,
        turn_id: str,
        event_type: EVENT_TYPES,
        turn: int = 1,
        payload: dict = None,
        run_id: str = '',
        source: str = '',
        metadata: dict = None,
    ):
        self.agent_name = agent_name
        self.agent_code = agent_code
        self.turn_id = turn_id
        self.turn = turn
        self.event_type = event_type
        self.payload = payload or {}

        self.run_id = run_id
        self.source = source
        self.created_at = time.time()
        self.metadata = dict(metadata or {})


class AgentEvent(BaseEvent):
    def __init__(
        self,
        agent_name: str,
        agent_code: str,
        turn_id: str,
        event_type: EVENT_TYPES,
        content: str = '',
        payload: dict = None,
        turn: int = 1,
        run_id: str = '',
        source: str = 'agent',
        metadata: dict = None,
    ):
        super().__init__(agent_name, agent_code, turn_id, turn=turn, event_type=event_type, payload=payload, run_id=run_id, source=source, metadata=metadata)
        self.content = content
    
    @staticmethod
    def build_from_resp(response: Response, agent_name: str, agent_code: str, turn_id: str, turn: int = 1) -> 'AgentEvent':
        if response.is_final:
            return AgentEvent(
                agent_name=agent_name,
                agent_code=agent_code,
                turn_id=turn_id,
                turn=turn,
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


class ToolEventAssembler:
    '''
    Incrementally converts streamed tool-call assembly chunks into ToolEvents.

    One instance per response stream: dedup/progress state lives on the
    instance, never shared across streams or runs.
    '''

    def __init__(self):
        self._started = set()    # tool_ids that already emitted tool:assembly:start
        self._finished = set()   # tool_ids whose arguments parsed as complete JSON
        self._args_len = {}      # tool_id -> length of arguments already emitted

    def build(
        self,
        response: Response,
        agent_name: str,
        agent_code: str,
        turn_id: str,
        turn: int = 1
    ) -> list['ToolEvent']:
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

            if tool_id not in self._started:
                self._started.add(tool_id)
                self._args_len[tool_id] = 0
                results.append(
                    ToolEvent(
                        agent_name=agent_name,
                        agent_code=agent_code,
                        turn_id=turn_id,
                        turn=turn,
                        event_type='tool:assembly:start',
                        tool_id=tool_call.get('id', ''),
                        tool_name=tool_name
                    )
                )

            if tool_id in self._finished:
                continue

            last_len = self._args_len.get(tool_id, 0)
            chunk_arg = arguments[last_len:]
            self._args_len[tool_id] = len(arguments)

            parsed_args = None
            is_json_valid = False
            if arguments.strip():
                try:
                    parsed_args = json.loads(arguments)
                    is_json_valid = True
                except json.JSONDecodeError: pass

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

            if is_json_valid:
                self._finished.add(tool_id)
                event = ToolEvent(
                    agent_name=agent_name,
                    agent_code=agent_code,
                    turn_id=turn_id,
                    event_type='tool:assembly:finish',
                    tool_id=tool_call.get('id', ''),
                    tool_name=tool_name
                )
                event.args = parsed_args
                results.append(event)

        return results


class ToolEvent(BaseEvent):
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
        turn: int = 1,
        run_id: str = '',
        source: str = 'tool',
        metadata: dict = None,
    ):
        super().__init__(agent_name, agent_code, turn_id, turn=turn, event_type=event_type, payload=payload, run_id=run_id, source=source, metadata=metadata)
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.chunk_arg = chunk_arg
        self.args: Optional[dict] = None

    @staticmethod
    def build_call_start(
        agent_name: str,
        agent_code: str,
        turn_id: str,
        tool_id: str,
        tool_name: str,
        args: Optional[dict] = None,
        turn: int = 1
    ) -> 'ToolEvent':
        event = ToolEvent(
            agent_name=agent_name,
            agent_code=agent_code,
            turn_id=turn_id,
            turn=turn,
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
        turn: int = 1
    ) -> 'ToolEvent':
        event_payload = payload or {}
        if 'result' not in event_payload:
            event_payload['result'] = result
        return ToolEvent(
            agent_name=agent_name,
            agent_code=agent_code,
            turn_id=turn_id,
            turn=turn,
            event_type='tool:call:finish',
            tool_id=tool_id,
            tool_name=tool_name,
            payload=event_payload,
        )


class RuntimeEvent(BaseEvent):
    def __init__(
        self,
        agent_name: str,
        agent_code: str,
        turn_id: str,
        event_type: EVENT_TYPES,
        payload: dict = None,
        turn: int = 1,
        run_id: str = '',
        source: str = 'runtime',
        metadata: dict = None,
    ):
        super().__init__(agent_name, agent_code, turn_id, turn=turn, event_type=event_type, payload=payload, run_id=run_id, source=source, metadata=metadata)
    