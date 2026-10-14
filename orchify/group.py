from typing import List, Dict, Union, Optional, Any

import os
import threading
import time

from .agent import Agent
from .backend import orchify_web_backend
from .call_chain import CallChain, CallBlocked, plan_call
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


DEFAULT_VOTE_TIMEOUT = 300.0

DEFAULT_MAX_CALL_DEPTH = 8


def _read_vote_timeout_env() -> Optional[float]:
    '''ORCHIFY_VOTE_TIMEOUT as a float ('0' means "do not wait"), else None.'''
    raw = os.getenv('ORCHIFY_VOTE_TIMEOUT')
    if raw is None or raw.strip() == '':
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _read_max_call_depth_env() -> Optional[int]:
    '''ORCHIFY_MAX_CALL_DEPTH as a non-negative int, else None.'''
    raw = os.getenv('ORCHIFY_MAX_CALL_DEPTH')
    if raw is None or raw.strip() == '':
        return None
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


class GroupEventBroker:
    '''Collect the events emitted by a group and fan them out to callbacks.

    Unlike ``orchify_broker.emit``, which may be called from a member's thread,
    this broker is locked: parallel members append to the same event list, and
    callbacks run on the emitting thread after the lock is released, so a slow
    callback cannot block appends.

    Attributes:
        events (List[GroupEventBase]): Every event received, in emission order.
    '''

    def __init__(self):
        self.events: List[GroupEventBase] = []
        self._callbacks: List[Any] = []
        self._lock = threading.RLock()

    def emit(self, event: GroupEventBase):
        '''Append an event and notify the registered callbacks.

        Args:
            event (GroupEventBase): The event to record and forward.
        '''
        with self._lock:
            self.events.append(event)
            callbacks = list(self._callbacks)
        for callback in callbacks:
            callback(event)

    def on_event(self, callback):
        '''Register a callback invoked for every group event.

        Args:
            callback (Callable): Called with the event on the emitting thread.
        '''
        with self._lock:
            self._callbacks.append(callback)

    def get_events_by_type(self, event_type: str) -> List[GroupEventBase]:
        '''Return the recorded events of one type.

        Args:
            event_type (str): Event type to filter on.

        Returns:
            List[GroupEventBase]: Matching events, in emission order.
        '''
        with self._lock:
            return [e for e in self.events if e.event_type == event_type]


class Group:
    '''A team of Agents that collaborate on a task.

    A Group owns a set of Agents (or nested Groups) and wires them together so
    that every member sees the others as expert tools it can call. Nested groups
    allow call chains such as Group -> Agent -> Group -> Agent.

    Attributes:
        name (str): Group name.
        description (str): What the group is for.
        agents (Dict[str, Union[Agent, Group]]): Members by name.
        agent_sequence (List[Union[Agent, Group]]): Members in insertion order.
        shared_tools (List[Tool]): Tools handed to every member.
        mode (str): Collaboration mode: ``'broadcast'``, ``'manager_delegation'``,
            ``'voting'`` or ``'competition'``.
        code (str): Short random code that makes the group's run ids unique.
        current_run_id (str): Run id of the active run.
        last_run_ids (List[str]): Run ids of the members of the last run.
        last_votes (List[Vote]): Votes collected by a ``'voting'`` run.
        last_winner (Optional[str]): Name of the winning member.
        last_answer (str): Answer produced by the last run.
        allow_concurrent_runs (bool): Whether overlapping runs are permitted.
        vote_timeout (float): Seconds a ``'voting'`` run waits for its members.
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
        max_call_depth: Optional[int] = None,
    ):
        '''Initialise a group of Agents.

        Args:
            name (str): Group name.
            agents (List[Union[Agent, Group]]): Members of the group.
            description (Optional[str]): Description of the group's purpose.
            manager_agent_name (Optional[str]): Name of the designated manager
                agent; defaults to the first member.
            shared_tools (Optional[List[Tool]]): Tools shared by every member.
            mode (str): Collaboration mode between the members:

                * ``'broadcast'``: every member may call every other member.
                * ``'manager_delegation'``: only the manager may call members.
                * ``'voting'``: every member answers, the majority answer wins.
                * ``'competition'``: every member competes, the best answer wins.
            allow_concurrent_runs (bool): When False (the default) a second
                ``run()`` is refused while one is in flight, because the group
                keeps ``current_run_id``/``last_run_ids`` on the instance and
                overlapping runs would clobber each other.
            vote_timeout (Optional[float]): Seconds a ``'voting'`` run waits for
                its members before tallying; defaults to
                ``ORCHIFY_VOTE_TIMEOUT``. Zero or a negative value dispatches
                without waiting, as older versions did.
            max_call_depth (Optional[int]): How deep nested Group/Agent calls may
                go before a call is refused; defaults to
                ``ORCHIFY_MAX_CALL_DEPTH``, else 8. Zero disables the cap. A call
                whose target is already on the call path is always refused as a
                cycle.

        Raises:
            ValueError: When ``mode`` is not a known collaboration mode.
        '''
        self.name = name
        self.description = description or f'A group of agents named {name}.'
        self.agents: Dict[str, Union[Agent, 'Group']] = {agent.name: agent for agent in agents}
        self.agent_sequence: List[Union[Agent, 'Group']] = agents
        self.shared_tools = shared_tools or []
        self.mode = mode
        self.broker = GroupEventBroker()
        self.code = safecode(length=4)
        self.current_run_id: str = ''
        self.last_run_id: str = ''
        self.last_run_ids: List[str] = []
        self.last_votes: List[Vote] = []
        self.last_winner: Optional[str] = None
        self.last_answer: str = ''

        self.allow_concurrent_runs = allow_concurrent_runs
        self._run_slot = threading.Lock()
        self._active_run_id: str = ''
        if vote_timeout is None:
            vote_timeout = _read_vote_timeout_env()
        if vote_timeout is None:
            vote_timeout = DEFAULT_VOTE_TIMEOUT
        self.vote_timeout = vote_timeout

        if max_call_depth is None:
            max_call_depth = _read_max_call_depth_env()
        if max_call_depth is None:
            max_call_depth = DEFAULT_MAX_CALL_DEPTH
        self.max_call_depth = max_call_depth
        self._call_path: List[str] = []

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

        try:
            from .manager import orchify_manager
            orchify_manager.register(self)
        except Exception:
            pass

    def _emit(self, event_type: str, payload: dict = None):
        '''Emit a group event.

        Group events belong to the group's own run (``current_run_id``). A group
        is a dispatcher, so its events are labelled turn 1: the LLM round trips
        happen inside each member's own run.

        Args:
            event_type (str): Event type to emit.
            payload (Optional[dict]): Event payload.
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
            '''Run the wrapped member agent.

            Args:
                **kwargs: Tool arguments; ``input`` (or ``query``) carries the
                    request for the member.

            Returns:
                str: The member agent's run id.
            '''
            user_input = kwargs.get('input', kwargs.get('query', ''))

            group_self._emit('agent_call', {
                'group_name': group_self.name,
                'agent_name': agent.name,
                'input': user_input
            })

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
        '''Wrap this group in a Tool so that agents can call it.

        Args:
            caller_agent (Optional[Agent]): When given, the caller's call chain
                records the group invocation, which is what allows nested group
                compositions.

        Returns:
            Tool: A tool whose execution runs this group.

        Raises:
            ValueError: When the group has no manager agent.
        '''
        if not self.manager_agent:
            raise ValueError('A manager agent must be defined to expose the group as a tool.')

        group_self = self
        _caller = caller_agent

        def group_runner(**kwargs):
            '''Run this group for a tool call.

            Args:
                **kwargs: Tool arguments; ``input`` (or ``query``) carries the
                    request for the group.

            Returns:
                str: The group's run id.
            '''
            user_input = kwargs.get('input', kwargs.get('query', ''))

            if _caller is not None:
                path = _caller._outgoing_path(f'Group:{group_self.name}')
                group_self._call_path = path

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
                        'call_depth': max(0, len(path) - 1)
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
        '''Run the group on a task.

        The execution flow depends on the group's ``mode``.

        Args:
            user_input (str): The task for the group.
            run_id (Optional[str]): Handle for this group execution;
                ``turn_id=`` is accepted as a legacy alias.
            **kwargs: Extra arguments forwarded to the members.

        Returns:
            str: The group's run id. Members run under their own run ids, which
            are collected in ``last_run_ids`` so that any single member can still
            be stopped, paused or approved on its own.
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
        '''Block until a group run finishes.

        Args:
            run_id (Optional[str]): Run to wait for; defaults to the group's last
                run.
            timeout (Optional[float]): Seconds to wait; defaults to
                ``ORCHIFY_RUN_TIMEOUT``, where None waits forever.

        Returns:
            bool: True when the run had finished by the deadline, False
            otherwise, including for a run id this process never started.

        Raises:
            ValueError: When there is no run to wait for.
            RuntimeError: When called from the orchestrator loop thread, where
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
        '''Check whether this thread may block until members finish.

        Returns:
            bool: False when ``vote_timeout`` is 0 or less (the older
            fire-and-forget behaviour), and False on the orchestrator loop
            thread, which executes the members and must never block.
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

    def _outgoing_path(self, callee_label: str) -> List[str]:
        '''Return the call path of a new outgoing call, or refuse it.

        Depth and cycle checks live in ``call_chain.plan_call``; a blocked call is
        reported as a ``run:call:blocked`` event before the error is re-raised.

        Args:
            callee_label (str): Label of the agent or group being called.

        Returns:
            List[str]: The outgoing call path.

        Raises:
            CallBlocked: When the call would exceed the depth cap or form a cycle.
        '''
        try:
            return plan_call(
                f'Group:{self.name}', self._call_path, callee_label, self.max_call_depth
            )
        except CallBlocked as blocked:
            orchify_broker.emit(RuntimeEvent(
                agent_name=self.name,
                agent_code=self.code,
                run_id=self.current_run_id,
                turn_id='',
                event_type='run:call:blocked',
                payload={
                    'reason': blocked.reason,
                    'caller': blocked.caller,
                    'callee': blocked.callee,
                    'call_path': list(blocked.path or []),
                    'call_depth': max(0, len(blocked.path or []) - 1),
                    'max_call_depth': blocked.limit,
                }
            ))
            raise

    def call_agent(self, agent: Agent, user_input: str, **kwargs) -> str:
        '''Call an agent inside or outside this group.

        Args:
            agent (Agent): Target agent.
            user_input (str): Input for the agent.
            **kwargs: Extra arguments forwarded to the agent.

        Returns:
            str: The agent's run id.
        '''
        path = self._outgoing_path(f'Agent:{agent.name}')
        agent._call_path = path
        agent._call_depth = max(0, len(path) - 1)

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
        '''Call another group (a nested group call).

        Args:
            group (Group): Target group.
            user_input (str): Input for the group.
            **kwargs: Extra arguments forwarded to the group.

        Returns:
            str: The group's run id.
        '''
        path = self._outgoing_path(f'Group:{group.name}')
        group._call_path = path

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
        '''Return this group's call chain as indented labels.

        Returns:
            List[str]: The call hierarchy, starting with ``Group:<name>``.
        '''
        chain = [f'Group:{self.name}']
        for agent in self.agent_sequence:
            if isinstance(agent, Agent):
                chain.append(f'  -> Agent:{agent.name}')
                if hasattr(agent, '_call_chain') and agent._call_chain:
                    for frame in agent._call_chain.frames:
                        chain.append(f'    -> {frame.callee}')
            elif isinstance(agent, Group):
                sub_chain = agent.get_call_chain()
                for item in sub_chain:
                    chain.append(f'  -> {item}')
        return chain

    def get_full_call_chain(self) -> CallChain:
        '''Return this group's call chain including nested calls.

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
                if hasattr(agent, '_call_chain') and agent._call_chain:
                    for frame in agent._call_chain.frames:
                        chain.frames.append(frame)
            elif isinstance(agent, Group):
                chain.add_frame(
                    caller=f'Group:{self.name}',
                    callee=f'Group:{agent.name}',
                    call_type='group'
                )
                sub_chain = agent.get_full_call_chain()
                chain.frames.extend(sub_chain.frames)

        return chain
