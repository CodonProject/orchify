from orchify.llm import Response
from typing import Literal, Optional, Any, TypeAlias
from dataclasses import dataclass

import json
import time


EVENT_TYPES: TypeAlias = Literal[
    'agent:start',
    'agent:reason:step',
    'agent:reason:finish',
    'agent:answer',
    'agent:finish',
    'agent:abort',
    'agent:call',
    'tool:assembly:start',
    'tool:assembly:step',
    'tool:assembly:finish',
    'tool:call:start',
    'tool:call:finish',
    'run:start',
    'run:next',
    'run:queued',
    'run:dequeued',
    'run:paused',
    'run:resumed',
    'run:approval:required',
    'run:approval:resolved',
    'run:budget:exceeded',
    'run:call:blocked',
    'run:finish',
    'group:start',
    'group:step',
    'group:end',
    'group:error',
    'group:agent_call',
    'group:call_agent',
    'group:call_group',
]

FEEDBACK_TYPES: TypeAlias = Literal[
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
    '''One control message queued for a run.

    Attributes:
        ftype (FEEDBACK_TYPES): The control command, e.g. ``'control:pause'``.
        payload (dict): Command-specific data; never ``None``.
    '''

    ftype: FEEDBACK_TYPES
    payload: dict = None

    def __post_init__(self):
        self.payload = self.payload or {}


class BaseEvent:
    '''Base class of every event emitted by a run.

    Attributes:
        agent_name (str): Name of the emitting agent or group.
        agent_code (str): Short code of the emitting agent or group.
        turn_id (str): Turn identifier the event belongs to.
        turn (int): Step number inside the run.
        event_type (EVENT_TYPES): Event name, e.g. ``'agent:answer'``.
        payload (dict): Event-specific data.
        run_id (str): Run that produced the event, when known.
        source (str): Origin of the event, e.g. ``'agent'`` or ``'plugin'``.
        created_at (float): Unix timestamp of creation.
        metadata (dict): Arbitrary caller-supplied metadata.
        source_entity (Any): Object that emitted the event, when available.
    '''

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
        source_entity: Any = None,
    ):
        '''Create an event.

        Args:
            agent_name (str): Name of the emitting agent or group.
            agent_code (str): Short code of the emitting agent or group.
            turn_id (str): Turn identifier the event belongs to.
            event_type (EVENT_TYPES): Type of the event.
            turn (int): 1-based step index inside the run.
            payload (Optional[dict]): Event payload; defaults to an empty dict.
            run_id (str): Run the event belongs to.
            source (str): Emitter label, for example 'agent' or 'plugin'.
            metadata (Optional[dict]): Extra metadata carried by the event.
            source_entity (Any): Object that emitted the event, when available.
        '''
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

        self.source_entity = source_entity


class AgentEvent(BaseEvent):
    '''Streaming text event of an agent turn.

    Attributes:
        content (str): Text of the reasoning step, answer or final message.
    '''

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
        '''Build the agent event for one streamed response chunk.

        Args:
            response (Response): Streamed response; either a chunk or the final status.
            agent_name (str): Name of the emitting agent.
            agent_code (str): Short code of the emitting agent.
            turn_id (str): Turn identifier the event belongs to.
            turn (int): Step number inside the run. Defaults to 1.

        Returns:
            AgentEvent: ``agent:finish`` for the final status, ``agent:reason:finish``
            at the end of reasoning, otherwise ``agent:reason:step`` or
            ``agent:answer``.

        Raises:
            ValueError: If the response carries neither a chunk nor a final status.
        '''
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
    '''Incrementally convert streamed tool-call chunks into tool events.

    One instance per response stream: dedup and progress state lives on the
    instance and is never shared across streams or runs.

    Attributes:
        _started (set): Tool-call ids whose assembly already started.
        _finished (set): Tool-call ids whose arguments parsed as complete JSON.
        _args_len (dict): Length of the arguments already reported per call.
    '''

    def __init__(self):
        self._started = set()
        self._finished = set()
        self._args_len = {}

    def build(
        self,
        response: Response,
        agent_name: str,
        agent_code: str,
        turn_id: str,
        turn: int = 1
    ) -> list['ToolEvent']:
        '''Build the tool events for one streamed chunk.

        Args:
            response (Response): The streamed response chunk.
            agent_name (str): Name of the emitting agent.
            agent_code (str): Short code of the emitting agent.
            turn_id (str): Turn identifier the events belong to.
            turn (int): Step number inside the run. Defaults to 1.

        Returns:
            list[ToolEvent]: Zero or more ``tool:assembly:start``/``step``/``finish``
            events, in stream order.
        '''
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
    '''Event describing one tool call or one streamed assembly chunk.

    Attributes:
        tool_id (str): Provider id of the tool call.
        tool_name (str): Name of the tool.
        chunk_arg (str): Raw argument fragment, only on assembly steps.
        args (Optional[dict]): Parsed arguments once they are complete.
    '''

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
        '''Build a ``tool:call:start`` event.

        Args:
            agent_name (str): Name of the emitting agent.
            agent_code (str): Short code of the emitting agent.
            turn_id (str): Turn identifier the event belongs to.
            tool_id (str): Provider id of the tool call.
            tool_name (str): Name of the tool.
            args (Optional[dict]): Parsed call arguments.
            turn (int): Step number inside the run. Defaults to 1.

        Returns:
            ToolEvent: The start event.
        '''
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
        '''Build a ``tool:call:finish`` event.

        Args:
            agent_name (str): Name of the emitting agent.
            agent_code (str): Short code of the emitting agent.
            turn_id (str): Turn identifier the event belongs to.
            tool_id (str): Provider id of the tool call.
            tool_name (str): Name of the tool.
            result (str): Result text handed back to the model.
            payload (Optional[dict]): Extra data merged into the payload; a
                ``'result'`` key is added when missing.
            turn (int): Step number inside the run. Defaults to 1.

        Returns:
            ToolEvent: The finish event.
        '''
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
    '''Generic lifecycle event emitted by the runtime.

    Used for ``run:*`` events and for control feedback, so every listener can
    observe runs without touching internals.
    '''

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


class ApprovalEvent(BaseEvent):
    '''Approval-pool event for one tool call or human question.

    Attributes:
        approval_id (str): Id of the pending approval in the broker pool.
        kind (str): ``'tool'`` for a confirmation gate, ``'question'`` for an
            open question.
        tool_id (str): Provider id of the gated tool call.
        tool_name (str): Name of the gated tool.
    '''

    @staticmethod
    def build_required(
        agent_name: str,
        agent_code: str,
        turn_id: str,
        approval_id: str,
        kind: str = 'tool',
        tool_id: str = '',
        tool_name: str = '',
        args: Optional[dict] = None,
        step: int = 1,
        timeout: Optional[float] = None,
    ) -> 'ApprovalEvent':
        '''Build the ``run:approval:required`` event.

        Args:
            agent_name (str): Name of the parking agent.
            agent_code (str): Short code of the parking agent.
            turn_id (str): Turn identifier the event belongs to.
            approval_id (str): Id of the pending approval.
            kind (str): ``'tool'`` or ``'question'``. Defaults to ``'tool'``.
            tool_id (str): Provider id of the gated tool call.
            tool_name (str): Name of the gated tool.
            args (Optional[dict]): Arguments of the gated call, or the question text.
            step (int): Step number inside the run. Defaults to 1.
            timeout (Optional[float]): Seconds the agent will wait before failing
                closed, when configured.

        Returns:
            ApprovalEvent: The required event.
        '''
        event = ApprovalEvent(
            agent_name=agent_name,
            agent_code=agent_code,
            turn_id=turn_id,
            turn=step,
            event_type='run:approval:required',
            payload={
                'approval_id': approval_id,
                'kind': kind,
                'tool_id': tool_id,
                'tool_name': tool_name,
                'args': args or {},
                'step': step,
                'timeout': timeout,
            },
        )
        event.approval_id = approval_id
        event.kind = kind
        event.tool_id = tool_id
        event.tool_name = tool_name
        return event

    @staticmethod
    def build_resolved(
        agent_name: str,
        agent_code: str,
        turn_id: str,
        approval_id: str,
        kind: str = 'tool',
        tool_id: str = '',
        tool_name: str = '',
        approved: bool = False,
        reason: str = '',
        timed_out: bool = False,
        step: int = 1,
        answer: str = '',
    ) -> 'ApprovalEvent':
        '''Build the ``run:approval:resolved`` event.

        Args:
            agent_name (str): Name of the parking agent.
            agent_code (str): Short code of the parking agent.
            turn_id (str): Turn identifier the event belongs to.
            approval_id (str): Id of the pending approval.
            kind (str): ``'tool'`` or ``'question'``. Defaults to ``'tool'``.
            tool_id (str): Provider id of the gated tool call.
            tool_name (str): Name of the gated tool.
            approved (bool): Whether the user approved it.
            reason (str): Explanation shown to the model.
            timed_out (bool): Whether the wait expired instead of being decided.
            step (int): Step number inside the run. Defaults to 1.
            answer (str): Free text of a ``kind='question'`` entry; empty for a tool
                approval.

        Returns:
            ApprovalEvent: The resolved event.
        '''
        event = ApprovalEvent(
            agent_name=agent_name,
            agent_code=agent_code,
            turn_id=turn_id,
            turn=step,
            event_type='run:approval:resolved',
            payload={
                'approval_id': approval_id,
                'kind': kind,
                'tool_id': tool_id,
                'tool_name': tool_name,
                'approved': bool(approved),
                'reason': reason,
                'answer': answer,
                'timed_out': bool(timed_out),
                'step': step,
            },
        )
        event.approval_id = approval_id
        event.kind = kind
        event.tool_id = tool_id
        event.tool_name = tool_name
        return event


class GroupEvent(BaseEvent):
    '''Event emitted by a Group run.

    Attributes:
        group_name (str): Name of the emitting group (also stored as
            ``agent_name``).
    '''

    def __init__(
        self,
        agent_name: str,
        agent_code: str,
        turn_id: str,
        event_type: EVENT_TYPES,
        payload: dict = None,
        turn: int = 1,
        run_id: str = '',
        source: str = 'group',
        metadata: dict = None,
        source_entity: Any = None,
    ):
        super().__init__(agent_name, agent_code, turn_id, turn=turn, event_type=event_type, payload=payload, run_id=run_id, source=source, metadata=metadata, source_entity=source_entity)
        self.group_name = agent_name

    @staticmethod
    def build_group_event(
        group_name: str,
        group_code: str,
        turn_id: str,
        event_type: EVENT_TYPES,
        payload: dict = None,
        run_id: str = '',
        source_entity: Any = None,
    ) -> 'GroupEvent':
        '''Build a group event.

        Args:
            group_name (str): Name of the emitting group.
            group_code (str): Short code of the emitting group.
            turn_id (str): Turn identifier the event belongs to.
            event_type (EVENT_TYPES): Event name, e.g. ``'group:end'``.
            payload (Optional[dict]): Event-specific data.
            run_id (str): Run that produced the event, when known.
            source_entity (Any): Group instance that emitted the event.

        Returns:
            GroupEvent: The built event.
        '''
        return GroupEvent(
            agent_name=group_name,
            agent_code=group_code,
            turn_id=turn_id,
            event_type=event_type,
            payload=payload,
            run_id=run_id,
            source_entity=source_entity,
        )
