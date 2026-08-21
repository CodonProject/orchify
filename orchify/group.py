from typing import List, Dict, Union, Optional, Any, Iterator
from collections import Counter
import time

from .agent import Agent
from .call_chain import CallChain, CallFrame
from .tool import Tool
from .schema import Vote
from .utils import safecode
from .broker import orchify_broker
from .event import RuntimeEvent, GroupEvent as GroupEventBase

import concurrent.futures
import json


class GroupEventBroker:
    '''Simple event broker for collecting group events.'''
    
    def __init__(self):
        self.events: List[GroupEventBase] = []
        self._callbacks: List[Any] = []
    
    def emit(self, event: GroupEventBase):
        self.events.append(event)
        for callback in self._callbacks:
            callback(event)
    
    def on_event(self, callback):
        self._callbacks.append(callback)
    
    def get_events_by_type(self, event_type: str) -> List[GroupEventBase]:
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
                                  'round_robin': Agents execute sequentially in a chain.
                                  'voting': All Agents vote on a final answer.
                                  'competition': All Agents compete, best answer wins.
        '''
        self.name = name
        self.description = description or f'A group of agents named {name}.'
        self.agents: Dict[str, Union[Agent, 'Group']] = {agent.name: agent for agent in agents}
        self.agent_sequence: List[Union[Agent, 'Group']] = agents
        self.shared_tools = shared_tools or []
        self.mode = mode
        self.broker = GroupEventBroker()
        self.code = safecode(length=4)
        
        if mode not in ['broadcast', 'manager_delegation', 'round_robin', 'voting', 'competition']:
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
        '''Emit a group event.'''
        event = GroupEventBase.build_group_event(
            group_name=self.name,
            group_code=self.code,
            turn_id='',
            event_type=event_type,
            payload=payload or {},
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

        for i, agent in enumerate(self.agent_sequence):
            final_toolset = list(agent.tools.values()) if isinstance(agent, Agent) else []
            final_toolset.extend(self.shared_tools)

            extra_context = {'collaboration_mode': self.mode}
            is_manager = (agent.name == self.manager_agent.name)

            if self.mode == 'round_robin':
                extra_context['mode_description'] = 'You are part of a sequential pipeline. Receive input, perform your specific task, and provide a clear final answer.'
                prev_agent = self.agent_sequence[i-1].name if i > 0 else 'the initial user input'
                next_agent = self.agent_sequence[i+1].name if i < len(self.agent_sequence) - 1 else 'the final output'
                extra_context['position_in_chain'] = f'You will receive input from \'{prev_agent}\' and your output will be passed to \'{next_agent}\'.'
            
            elif self.mode == 'manager_delegation':
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
                turn_id=kwargs.get('turn_id', ''),
                event_type='group:agent_call',
                payload={
                    'group_name': group_self.name,
                    'agent_name': agent.name,
                    'input': user_input
                }
            ))
            
            turn_id = agent.run(user_input, **kwargs)
            return turn_id
        
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
                    turn_id=kwargs.get('turn_id', ''),
                    event_type='agent:call',
                    payload={
                        'caller': _caller.name,
                        'callee': group_self.name,
                        'call_type': 'group',
                        'input': user_input,
                        'call_depth': _caller._call_depth
                    }
                ))

            turn_id = group_self.run(user_input, **kwargs)
            return turn_id

        group_runner.__name__ = self.name
        group_runner.__doc__ = self.description

        from inspect import Parameter, Signature
        params = [
            Parameter(name='input', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        ]
        group_runner.__signature__ = Signature(params)

        tool = Tool(func=group_runner, name=self.name, description=f'Group: {self.name}')
        return tool

    def run(self, user_input: str, **kwargs) -> str:
        '''
        Runs the entire Group to perform a task.
        The execution flow depends on the group's mode.

        Args:
            user_input: The input task for the group.
            **kwargs: Additional parameters.

        Returns:
            str: The turn_id for tracking the execution.
        '''
        self._emit('start', {'mode': self.mode, 'input': user_input})
        
        try:
            if self.mode == 'round_robin':
                turn_id = self._run_round_robin(user_input, **kwargs)
            elif self.mode == 'voting':
                turn_id = self._run_voting(user_input, **kwargs)
            elif self.mode == 'competition':
                turn_id = self._run_competition(user_input, **kwargs)
            else:
                turn_id = self._run_manager_based(user_input, **kwargs)
            
            self._emit('end', {'turn_id': turn_id})
            return turn_id
        except Exception as e:
            self._emit('error', {'message': str(e)})
            raise

    def _run_round_robin(self, user_input: str, **kwargs) -> str:
        '''Runs the group in a sequential, round-robin fashion.'''
        if not self.agent_sequence:
            return ''
        
        self._emit('step', {'action': 'Starting round-robin execution'})
        
        first_agent = self.agent_sequence[0]
        turn_id = first_agent.run(user_input, **kwargs)
        
        self._emit('step', {
            'action': f'Agent \'{first_agent.name}\' started',
            'turn_id': turn_id
        })
        
        return turn_id

    def _run_voting(self, user_input: str, **kwargs) -> str:
        '''Runs the group in a parallel, voting-based fashion.'''
        self._emit('step', {'action': 'Starting voting execution'})
        
        turn_ids = []
        for agent in self.agent_sequence:
            turn_id = agent.run(user_input, **kwargs)
            turn_ids.append(turn_id)
            self._emit('step', {
                'action': f'Agent \'{agent.name}\' started voting',
                'turn_id': turn_id
            })
        
        return turn_ids[0] if turn_ids else ''

    def _run_competition(self, user_input: str, **kwargs) -> str:
        '''Runs the group in a parallel, competition-based fashion.'''
        self._emit('step', {'action': 'Starting competition execution'})
        
        turn_ids = []
        for agent in self.agent_sequence:
            turn_id = agent.run(user_input, **kwargs)
            turn_ids.append(turn_id)
            self._emit('step', {
                'action': f'Agent \'{agent.name}\' started competing',
                'turn_id': turn_id
            })
        
        return turn_ids[0] if turn_ids else ''

    def _run_manager_based(self, user_input: str, **kwargs) -> str:
        '''Runs the main loop for manager-based modes.'''
        self._emit('step', {
            'action': f'Delegating to manager \'{self.manager_agent.name}\''
        })
        
        turn_id = self.manager_agent.run(user_input, **kwargs)
        return turn_id

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
            str: The turn_id of the agent's execution.
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
            turn_id=kwargs.get('turn_id', ''),
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
            str: The turn_id of the group's execution.
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
            turn_id=kwargs.get('turn_id', ''),
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