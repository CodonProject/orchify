from typing import List, Dict, Union, Optional, Any

import os
import threading
import time

from .agent import Agent
from .backend import orchify_web_backend
from .call_chain import CallChain
from .schema import Vote
from .tool import Tool
from .utils import safecode
from .broker import orchify_broker, make_turn_id
from .event import RuntimeEvent, GroupEvent as GroupEventBase


def _read_float_env(name: str) -> Optional[float]:
    '''Positive float from the environment, or None when unset/invalid.'''
    raw = os.getenv(name)
    if not raw or raw.strip() == '':
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


# how long a 'voting' run waits for its members before tallying, unless
# ORCHIFY_VOTE_TIMEOUT (or Group(vote_timeout=...)) says otherwise. 0 or a
# negative value restores the old "dispatch and return" behaviour.
DEFAULT_VOTE_TIMEOUT = 300.0


def _read_vote_timeout_env() -> Optional[float]:
    '''ORCHIFY_VOTE_TIMEOUT as a float ('0' means "do not wait"), else None.'''
    raw = os.getenv('ORCHIFY_VOTE_TIMEOUT')
    if raw is None or raw.strip() == '':
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class GroupEventBroker:
    '''Simple event broker for collecting group events.

    Unlike `orchify_broker.emit` (which may be called from a member's thread),
    this one is locked: members running in parallel append to the same event
    list, and callbacks are invoked on the emitting thread AFTER the lock is
    released so a slow callback cannot block appends.
    '''

    def __init__(self):
        self.events: List[GroupEventBase] = []
        self._callbacks: List[Any] = []
        self._lock = threading.RLock()

    def emit(self, event: GroupEventBase):
        with self._lock:
            self.events.append(event)
            callbacks = list(self._callbacks)
        for callback in callbacks:
            callback(event)

    def on_event(self, callback):
        with self._lock:
            self._callbacks.append(callback)

    def get_events_by_type(self, event_type: str) -> List[GroupEventBase]:
        with self._lock:
            return [e for e in self.events if e.event_type == event_type]


class Group:
    '''
    A team of Agents that can collaborate to accomplish complex tasks.

    A Group contains a set of Agents. It automatically 'wires' them up so that
    each Agent can see the other Agents in the team as expert tools to be called upon.
    
    Supports nested Groups for complex call chains like: Group -> Agent -> Group -> Agent
    '''

    def __init__(
        self,
        name: str,
        agents: List[Union[Agent, 'Group']],
        description: Optional[str] = None,
        manager_agent_name: Optional[str] = None,
        shared_tools: Optional[List[Tool]] = None,
        mode: str = 'broadcast',
        *,
        allow_concurrent_runs: bool = False,
        vote_timeout: Optional[float] = None,
    ):
        '''Initializes an Agent Group.

        Args:
            name (str): The name of the group.
            agents (List[Agent]): A list of Agent or Group instances in the group.
            description (Optional[str], optional): A description of the group's purpose.
            manager_agent_name (str, optional): The name of the designated manager Agent.
            shared_tools (Optional[List[Tool]], optional): A list of tools shared by the group.
            mode (str, optional): The communication mode between Agents.
                                  'broadcast': All Agents can call each other.
                                  'manager_delegation': Only the manager can call other Agents.
                                  'voting': All Agents answer, the majority answer wins.
                                  'competition': All Agents compete, best answer wins.
            allow_concurrent_runs (bool): refuse a second run() while one is in
              flight (default). The group keeps `current_run_id`/`last_run_ids` on
              the instance, so overlapping runs would clobber each other.
            vote_timeout (float, optional): seconds a 'voting' run waits for its
              members before tallying (defaults to ORCHIFY_VOTE_TIMEOUT; 0 or a
              negative value dispatches without waiting, as older versions did).
        '''
        self.name = name
        self.description = description or f'A group of agents named {name}.'
        self.agents: Dict[str, Union[Agent, 'Group']] = {agent.name: agent for agent in agents}
        self.agent_sequence: List[Union[Agent, 'Group']] = agents
        self.shared_tools = shared_tools or []
        self.mode = mode
        self.broker = GroupEventBroker()
        self.code = safecode(length=4)
        # run identity: the group's own execution, and the run_id of every member
        # it has dispatched (members run under their OWN run_id)
        self.current_run_id: str = ''
        self.last_run_id: str = ''
        self.last_run_ids: List[str] = []
        # voting results (filled by _run_voting)
        self.last_votes: List[Vote] = []
        self.last_winner: Optional[str] = None
        self.last_answer: str = ''

        # one group run at a time per instance: concurrent runs would share
        # current_run_id / last_run_ids / last_votes.
        self.allow_concurrent_runs = allow_concurrent_runs
        self._run_slot = threading.Lock()
        self._active_run_id: str = ''
        if vote_timeout is None:
            vote_timeout = _read_vote_timeout_env()
        if vote_timeout is None:
            vote_timeout = DEFAULT_VOTE_TIMEOUT
        self.vote_timeout = vote_timeout
        
        # 'round_robin' was removed: Agent.run() returns a run handle instead of
        # the answer, so a group cannot chain members synchronously.
        if mode == 'round_robin':
            raise ValueError(
                "mode 'round_robin' was removed. Chain agents yourself with "
                "Group.call_agent()/Agent.call_agent(), or use "
                "'manager_delegation' to have one agent drive the others."
            )
        if mode not in ['broadcast', 'manager_delegation', 'voting', 'competition']:
            raise ValueError(f'Unsupported mode: {mode}')

        self.manager_agent = None

        if not agents:
            raise ValueError('Group must contain at least one agent.')

        if manager_agent_name:
            if manager_agent_name not in self.agents:
                raise ValueError(f'Manager agent \'{manager_agent_name}\' not found in the group.')
            self.manager_agent = self.agents[manager_agent_name]
        elif self.agent_sequence:
            self.manager_agent = self.agent_sequence[0]
        
        self._wire_agents()

    def _emit(self, event_type: str, payload: dict = None):
        '''Emit a group event.

        Group events belong to the group's own run (`current_run_id`). A group is
        a dispatcher, so it labels them turn 1: the LLM round trips happen inside
        each member's own run.
        '''
        run_id = self.current_run_id
        event = GroupEventBase.build_group_event(
            group_name=self.name,
            group_code=self.code,
            turn_id=make_turn_id(run_id, 1) if run_id else '',
            event_type=event_type,
            payload=payload or {},
            run_id=run_id,
            source_entity=self,
        )
        self.broker.emit(event)
        return event

    def _wire_agents(self):
        '''Configures the toolset and context for each agent in the group.'''
        all_agents_as_tools = {}
        for name, agent in self.agents.items():
            if isinstance(agent, Agent):
                all_agents_as_tools[name] = self._agent_to_tool(agent)
            elif isinstance(agent, Group):
                all_agents_as_tools[name] = agent.as_tool()

        for agent in self.agent_sequence:
            final_toolset = list(agent.tools.values()) if isinstance(agent, Agent) else []
            final_toolset.extend(self.shared_tools)

            extra_context = {'collaboration_mode': self.mode}
            is_manager = (agent.name == self.manager_agent.name)

            if self.mode == 'manager_delegation':
                if is_manager:
                    extra_context['mode_description'] = 'You are the manager. Break down tasks and delegate to expert agents.'
                    for other_name, other_agent_as_tool in all_agents_as_tools.items():
                        if agent.name != other_name:
                            final_toolset.append(other_agent_as_tool)
                else:
                    extra_context['mode_description'] = 'You are an expert agent. Execute tasks assigned by your manager.'

            elif self.mode == 'broadcast':
                for other_name, other_agent_as_tool in all_agents_as_tools.items():
                    if agent.name != other_name:
                        final_toolset.append(other_agent_as_tool)
            
            elif self.mode == 'voting':
                extra_context['mode_description'] = 'You are part of a voting panel. Perform the task and provide a definitive final answer.'

            elif self.mode == 'competition':
                extra_context['mode_description'] = 'You are in a competition. Provide the best answer among all participants.'

            if isinstance(agent, Agent):
                if hasattr(agent, '_configure_with_tools'):
                    agent._configure_with_tools(final_toolset, extra_context=extra_context)
                else:
                    for tool in final_toolset:
                        if isinstance(tool, Tool):
                            agent.add_tool(tool, replace=True)
            elif isinstance(agent, Group):
                agent._wire_agents()

    def _agent_to_tool(self, agent: Agent) -> Tool:
        '''Converts an Agent to a Tool that can be called by other agents.'''
        group_self = self
        
        def agent_runner(**kwargs):
            user_input = kwargs.get('input', kwargs.get('query', ''))
            
            # Emit agent call event
            group_self._emit('agent_call', {
                'group_name': group_self.name,
                'agent_name': agent.name,
                'input': user_input
            })
            
            # Also emit runtime event for the broker
            orchify_broker.emit(RuntimeEvent(
                agent_name=group_self.name,
                agent_code=group_self.code,
                run_id=group_self.current_run_id,
                turn_id=kwargs.get('run_id') or kwargs.get('turn_id') or '',
                event_type='group:agent_call',
                payload={
                    'group_name': group_self.name,
                    'agent_name': agent.name,
                    'input': user_input
                }
            ))
            
            run_id = agent.run(user_input, **kwargs)
            return run_id
        
        agent_runner.__name__ = agent.name
        agent_runner.__doc__ = f'Call agent \'{agent.name}\' to perform a task.'
        
        from inspect import Parameter, Signature
        params = [
            Parameter(name='input', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        ]
        agent_runner.__signature__ = Signature(params)
        
        tool = Tool(func=agent_runner, name=agent.name, description=f'Agent: {agent.name}')
        return tool

    def as_tool(self, caller_agent: Optional[Agent] = None) -> Tool:
        '''Wraps the entire Group instance into a Tool, with optional call chain tracking.

        Args:
            caller_agent: If provided, call chain tracking is set up so that
                when this tool is executed, the caller's call chain records the
                Group invocation. Useful for nested group compositions.

        Raises:
            ValueError: If no manager agent is defined for this group.

        Returns:
            A Tool whose execute() runs this group.
        '''
        if not self.manager_agent:
            raise ValueError('A manager agent must be defined to expose the group as a tool.')

        group_self = self
        _caller = caller_agent

        def group_runner(**kwargs):
            user_input = kwargs.get('input', kwargs.get('query', ''))

            if _caller is not None:
                if _caller._call_chain is None:
                    _caller._call_chain = CallChain(root_caller=f'Agent:{_caller.name}')

                _caller._call_chain.add_frame(
                    caller=f'Agent:{_caller.name}',
                    callee=f'Group:{group_self.name}',
                    call_type='group',
                    input_data=user_input
                )

                orchify_broker.emit(RuntimeEvent(
                    agent_name=_caller.name,
                    agent_code=_caller.code,
                    run_id=_caller.current_run_id,
                    turn_id=kwargs.get('run_id') or kwargs.get('turn_id') or '',
                    event_type='agent:call',
                    payload={
                        'caller': _caller.name,
                        'callee': group_self.name,
                        'call_type': 'group',
                        'input': user_input,
                        'call_depth': _caller._call_depth
                    }
                ))

            run_id = group_self.run(user_input, **kwargs)
            return run_id

        group_runner.__name__ = self.name
        group_runner.__doc__ = self.description

        from inspect import Parameter, Signature
        params = [
            Parameter(name='input', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        ]
        group_runner.__signature__ = Signature(params)

        tool = Tool(func=group_runner, name=self.name, description=f'Group: {self.name}')
        return tool

    def run(self, user_input: str, run_id: Optional[str] = None, **kwargs) -> str:
        '''
        Runs the entire Group to perform a task.
        The execution flow depends on the group's mode.

        Args:
            user_input: The input task for the group.
            run_id: Optional handle for this group execution.
            **kwargs: Additional parameters.

        Returns:
            str: The group's run_id. Members run under their OWN run_ids, which
            are collected in `last_run_ids` so any single member can still be
            stopped, paused or approved on its own.

        `turn_id=` is accepted as a legacy alias of run_id.
        '''
        run_id = run_id or kwargs.pop('turn_id', None) or f'{self.name}#{self.code}_{safecode()}'

        if not self.allow_concurrent_runs and not self._run_slot.acquire(blocking=False):
            raise RuntimeError(
                f"Group '{self.name}' is already running (run_id={self._active_run_id or self.last_run_id}). "
                "A Group keeps current_run_id/last_run_ids/last_votes on the instance, so overlapping "
                "runs would clobber each other: wait for the current run (Group.wait()), stop it "
                "(Group.cancel()), use a separate Group instance, or construct this one with "
                "allow_concurrent_runs=True."
            )

        self._active_run_id = run_id
        self.current_run_id = run_id
        self.last_run_id = run_id
        self.last_run_ids = []
        self.last_votes = []
        self.last_winner = None
        self.last_answer = ''
        orchify_broker.start_task(run_id, None, context={
            'run_id': run_id,
            'task_name': run_id,
            'kind': 'group',
            'group_name': self.name,
            'group_code': self.code,
            'mode': self.mode,
            'started_at': time.time(),
        })

        self._emit('start', {'mode': self.mode, 'input': user_input, 'run_id': run_id})
        
        try:
            if self.mode == 'voting':
                self._run_voting(user_input, **kwargs)
            elif self.mode == 'competition':
                self._run_competition(user_input, **kwargs)
            else:
                self._run_manager_based(user_input, **kwargs)

            self._emit('end', {
                'run_id': run_id,
                'agent_run_ids': list(self.last_run_ids),
                'votes': [{'agent_name': v.agent_name, 'vote': v.vote, 'reason': v.reason} for v in self.last_votes],
                'winner': self.last_winner,
                'answer': self.last_answer,
            })
            return run_id
        except Exception as e:
            self._emit('error', {'message': str(e)})
            raise
        finally:
            orchify_broker.finish_task(run_id)
            self.current_run_id = ''
            self._release_run_slot()

    def _release_run_slot(self) -> None:
        '''Release this group's run slot (safe to call twice).'''
        self._active_run_id = ''
        try:
            self._run_slot.release()
        except RuntimeError:
            pass

    def is_running(self) -> bool:
        '''True while a run() is in flight on this instance.'''
        return self._run_slot.locked()

    def wait(self, run_id: Optional[str] = None, timeout: Optional[float] = None) -> bool:
        '''
        Block until the group's run finishes.

        Args:
            run_id: The run to wait for; defaults to the group's last run.
            timeout: Seconds to wait; defaults to ORCHIFY_RUN_TIMEOUT (None waits
                     forever).

        Returns:
            bool: True when the run had finished by the deadline, False otherwise
            (also False for a run id this process never started).

        Raises:
            ValueError: when there is no run to wait for.
            RuntimeError: when called from the orchestrator loop thread, where
                          blocking would stall every other run.
        '''
        target = run_id or self.last_run_id
        if not target:
            raise ValueError(f"Group '{self.name}' has no run to wait for; pass run_id=... .")
        if timeout is None:
            timeout = _read_float_env('ORCHIFY_RUN_TIMEOUT')
        if threading.current_thread() is orchify_web_backend._thread:
            raise RuntimeError(
                "Group.wait() would deadlock: it is being called from the orchestrator loop "
                "thread, which is the thread that executes the run. Wait from your own "
                "thread, or use the 'run:finish' hook instead."
            )
        return orchify_broker.wait_run(target, timeout=timeout)

    def cancel(self, run_id: Optional[str] = None) -> bool:
        '''Request a stop for this group's run (`control:stop` feedback).'''
        target = run_id or self.last_run_id
        if not target:
            return False
        return orchify_broker.feedback('control:stop', run_id=target)

    @staticmethod
    def _normalize_vote(text: Any) -> str:
        '''Normalise an answer for tallying (whitespace + case insensitive).'''
        return ' '.join(str(text or '').split()).strip().lower()

    def _can_block_for_votes(self) -> bool:
        '''True when this thread may block until members finish.

        `vote_timeout` of 0 (or less) keeps the old fire-and-forget behaviour, and
        the orchestrator loop thread must never block: it is the thread that
        executes the members.
        '''
        if self.vote_timeout is None or self.vote_timeout <= 0:
            return False
        return threading.current_thread() is not orchify_web_backend._thread

    def _collect_votes(self) -> None:
        '''Tally the members' answers after waiting for them (voting mode).'''
        votes: List[Vote] = []
        counts: Dict[str, int] = {}
        for agent in self.agent_sequence:
            answer = getattr(agent, 'last_answer', '') or ''
            votes.append(Vote(agent_name=agent.name, vote=answer))
            key = self._normalize_vote(answer)
            if key:
                counts[key] = counts.get(key, 0) + 1

        self.last_votes = votes
        self.last_winner = None
        self.last_answer = ''
        if not counts:
            return

        # majority; ties go to the answer that reached its count first
        winner_key = max(counts, key=lambda k: counts[k])
        for vote in votes:
            if self._normalize_vote(vote.vote) == winner_key:
                self.last_winner = vote.agent_name
                self.last_answer = str(vote.vote or '')
                break
        self._emit('vote', {
            'winner': self.last_winner,
            'answer': self.last_answer,
            'votes': [{'agent_name': v.agent_name, 'vote': v.vote} for v in votes],
        })

    def _run_voting(self, user_input: str, **kwargs) -> str:
        '''Runs the group in a parallel, voting-based fashion.'''
        self._emit('step', {'action': 'Starting voting execution'})
        
        for agent in self.agent_sequence:
            run_id = agent.run(user_input, **kwargs)
            self.last_run_ids.append(run_id)
            self._emit('step', {
                'action': f'Agent \'{agent.name}\' started voting',
                'run_id': run_id
            })

        if self._can_block_for_votes():
            deadline = time.monotonic() + self.vote_timeout
            for member_run_id in list(self.last_run_ids):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not orchify_broker.wait_run(member_run_id, timeout=remaining):
                    break
            self._collect_votes()

        return self.last_run_ids[0] if self.last_run_ids else ''

    def _run_competition(self, user_input: str, **kwargs) -> str:
        '''Runs the group in a parallel, competition-based fashion.'''
        self._emit('step', {'action': 'Starting competition execution'})
        
        for agent in self.agent_sequence:
            run_id = agent.run(user_input, **kwargs)
            self.last_run_ids.append(run_id)
            self._emit('step', {
                'action': f'Agent \'{agent.name}\' started competing',
                'run_id': run_id
            })
        
        return self.last_run_ids[0] if self.last_run_ids else ''

    def _run_manager_based(self, user_input: str, **kwargs) -> str:
        '''Runs the main loop for manager-based modes.'''
        self._emit('step', {
            'action': f'Delegating to manager \'{self.manager_agent.name}\''
        })
        
        run_id = self.manager_agent.run(user_input, **kwargs)
        self.last_run_ids.append(run_id)
        return run_id

    def to_dict(self) -> Dict[str, Any]:
        '''Serializes the group's configuration to a dictionary.'''
        return {
            'name': self.name,
            'description': self.description,
            'agents': [agent.name for agent in self.agent_sequence],
            'manager_agent_name': self.manager_agent.name if self.manager_agent else None,
            'shared_tools': [tool.name for tool in self.shared_tools],
            'mode': self.mode
        }

    def _configure_with_tools(self, tools: List[Tool], extra_context: Optional[Dict[str, Any]] = None) -> None:
        '''Configures the group with additional tools and context for nested groups.'''
        self.shared_tools.extend(tools)
        if extra_context:
            if not hasattr(self, '_extra_context'):
                self._extra_context = {}
            self._extra_context.update(extra_context)

    def call_agent(self, agent: Agent, user_input: str, **kwargs) -> str:
        '''
        Calls an agent within or outside this group.
        
        Args:
            agent: The target agent to call.
            user_input: The input to send to the agent.
            **kwargs: Additional parameters.
            
        Returns:
            str: The run_id of the agent's execution.
        '''
        self._emit('agent_call', {
            'caller': self.name,
            'callee': agent.name,
            'call_type': 'agent',
            'input': user_input
        })
        
        orchify_broker.emit(RuntimeEvent(
            agent_name=self.name,
            agent_code=self.code,
            run_id=self.current_run_id,
            turn_id=kwargs.get('run_id') or kwargs.get('turn_id') or '',
            event_type='group:call_agent',
            payload={
                'group_name': self.name,
                'agent_name': agent.name,
                'input': user_input
            }
        ))
        
        return agent.run(user_input, **kwargs)

    def call_group(self, group: 'Group', user_input: str, **kwargs) -> str:
        '''
        Calls another group (nested group call).
        
        Args:
            group: The target group to call.
            user_input: The input to send to the group.
            **kwargs: Additional parameters.
            
        Returns:
            str: The run_id of the group's execution.
        '''
        self._emit('group_call', {
            'caller': self.name,
            'callee': group.name,
            'call_type': 'group',
            'input': user_input
        })
        
        orchify_broker.emit(RuntimeEvent(
            agent_name=self.name,
            agent_code=self.code,
            run_id=self.current_run_id,
            turn_id=kwargs.get('run_id') or kwargs.get('turn_id') or '',
            event_type='group:call_group',
            payload={
                'caller_group': self.name,
                'callee_group': group.name,
                'input': user_input
            }
        ))
        
        return group.run(user_input, **kwargs)

    def get_call_chain(self) -> List[str]:
        '''
        Returns the call chain structure of this group.
        
        Returns:
            List[str]: A list representing the call hierarchy.
        '''
        chain = [f'Group:{self.name}']
        for agent in self.agent_sequence:
            if isinstance(agent, Agent):
                chain.append(f'  -> Agent:{agent.name}')
                # Check if agent has sub-calls
                if hasattr(agent, '_call_chain') and agent._call_chain:
                    for frame in agent._call_chain.frames:
                        chain.append(f'    -> {frame.callee}')
            elif isinstance(agent, Group):
                sub_chain = agent.get_call_chain()
                for item in sub_chain:
                    chain.append(f'  -> {item}')
        return chain
    
    def get_full_call_chain(self) -> CallChain:
        '''
        Returns a CallChain object with all nested calls.
        
        Returns:
            CallChain: The complete call chain.
        '''
        chain = CallChain(root_caller=f'Group:{self.name}')
        
        for agent in self.agent_sequence:
            if isinstance(agent, Agent):
                chain.add_frame(
                    caller=f'Group:{self.name}',
                    callee=f'Agent:{agent.name}',
                    call_type='agent'
                )
                # Include agent's sub-calls if any
                if hasattr(agent, '_call_chain') and agent._call_chain:
                    for frame in agent._call_chain.frames:
                        chain.frames.append(frame)
            elif isinstance(agent, Group):
                chain.add_frame(
                    caller=f'Group:{self.name}',
                    callee=f'Group:{agent.name}',
                    call_type='group'
                )
                # Include sub-group's chain
                sub_chain = agent.get_full_call_chain()
                chain.frames.extend(sub_chain.frames)
        
        return chain