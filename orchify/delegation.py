'''Runtime sub-agent delegation.

A single agent can hand a self-contained task to a fresh agent and use the
answer, instead of cramming everything into one prompt::

    from orchify import Agent, Delegation

    delegates = Delegation(factory=lambda name: Agent.from_env(name=name), max_depth=2)
    parent = Agent(name='lead', llm=llm, tools=[delegates.tool()])
    parent.ask('Coordinate: ask a helper how many bytes a UTF-8 Chinese character takes.')

Depth is tracked with a :class:`contextvars.ContextVar`, so nested delegations
stop at ``max_depth`` even when every level shares the same tool object. A
delegation that runs out of depth returns a plain text note instead of raising:
the parent keeps going and can answer with what it has.
'''

from __future__ import annotations

import contextvars
import itertools
import logging
import threading
from typing import Any, Callable, List, Optional

from orchify.tool import Tool, tool

logger = logging.getLogger('orchify.delegation')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

_DEPTH: 'contextvars.ContextVar[int]' = contextvars.ContextVar(
    'orchify_delegation_depth', default=0)


class DelegationError(RuntimeError):
    '''Raised when a delegation cannot be created at all.'''


class Delegation:
    '''Factory for child agents plus the ``delegate`` tool that uses them.

    Attributes:
        factory (Callable[[str], Any]): Builds an agent for a child name.
        max_depth (int): How many nested delegations are allowed.
        max_concurrent (int): Simultaneous child runs; extra callers wait.
        timeout (Optional[float]): Per-child timeout in seconds.
        max_steps (int): Step budget handed to each child.
        names (List[str]): Child names created by this instance.
    '''

    def __init__(
        self,
        factory: Optional[Callable[[str], Any]] = None,
        *,
        agent: Any = None,
        max_depth: int = 2,
        max_concurrent: int = 4,
        timeout: Optional[float] = None,
        max_steps: int = 8,
        prefix: str = 'helper',
        tool_name: str = 'delegate',
        require_approval: bool = False,
    ):
        '''Create a delegation manager.

        Args:
            factory (Optional[Callable[[str], Any]]): Builds a child agent given
                its name; takes precedence over ``agent``.
            agent (Any): Template agent cloned for every child (shared LLM and
                tools, minus the delegate tool itself).
            max_depth (int): Maximum nesting depth. Defaults to 2.
            max_concurrent (int): Pending child runs allowed at once.
            timeout (Optional[float]): Timeout passed to ``child.ask``.
            max_steps (int): Steps per child run.
            prefix (str): Base for generated child names.
            tool_name (str): Name of the tool exposed to the parent.
            require_approval (bool): Ask before running a child agent.
        '''
        if factory is None and agent is None:
            raise DelegationError('Delegation needs a factory= or an agent= template')
        self.factory = factory
        self.agent = agent
        self.max_depth = max(1, int(max_depth))
        self.max_concurrent = max(1, int(max_concurrent))
        self.timeout = timeout
        self.max_steps = max(1, int(max_steps))
        self.prefix = prefix
        self.tool_name = tool_name
        self.require_approval = bool(require_approval)
        self.names: List[str] = []
        self._counter = itertools.count(1)
        self._slots = threading.BoundedSemaphore(self.max_concurrent)
        self._tools: Dict[int, Tool] = {}

    @property
    def depth(self) -> int:
        '''Current delegation depth of the calling thread.'''
        return _DEPTH.get()

    def _child_name(self, name: str = '') -> str:
        '''Pick a display name for a child agent.'''
        if name:
            return str(name)
        return '%s-%d' % (self.prefix, next(self._counter))

    def build(self, name: str = '', **overrides: Any) -> Any:
        '''Create a child agent without running it.

        Args:
            name (str): Child name; generated when empty.
            **overrides: Attributes applied on top of the template agent.

        Returns:
            Any: The child agent.
        '''
        if _DEPTH.get() >= self.max_depth:
            raise DelegationError(
                'delegation limit reached (%d); do the task here instead' % self.max_depth)
        return self._build_child(name, overrides)

    def _build_child(self, name: str = '', overrides: Optional[dict] = None) -> Any:
        '''Create a child agent without the depth check (caller owns it).'''
        child_name = self._child_name(name)
        if self.factory is not None:
            child = self.factory(child_name)
        else:
            child = self._clone(self.agent, child_name)
        for key, value in (overrides or {}).items():
            setattr(child, key, value)
        self.names.append(child_name)
        return child

    @staticmethod
    def _clone(template: Any, name: str) -> Any:
        '''Clone a template agent, dropping the delegate tool from its toolbox.'''
        from orchify.agent import Agent

        tools = []
        for candidate in getattr(template, 'tools', {}).values():
            if getattr(candidate, 'name', '') in ('delegate',) or getattr(
                    candidate, 'description', '').startswith('Run a sub-agent'):
                continue
            tools.append(candidate)
        return Agent(
            name=name,
            llm=getattr(template, 'llm', None),
            model=getattr(template, 'model', '') or '',
            system_prompt=getattr(template, 'system_prompt', '') or '',
            tools=tools,
            messages_mode=getattr(template, 'messages_mode', 'agent'),
            temperature=getattr(template, 'temperature', 0.7),
            top_p=getattr(template, 'top_p', 1.0),
            max_tokens=getattr(template, 'max_tokens', None),
        )

    def delegate(self, task: str, name: str = '', *, depth: Optional[int] = None) -> str:
        '''Run one task in a child agent and return its answer.

        Args:
            task (str): Self-contained instruction for the child.
            name (str): Optional child name for logs and events.
            depth (Optional[int]): Nesting level of the caller. The tool passes
                its bound level because tool calls run on worker threads, where
                the :class:`contextvars.ContextVar` cannot follow.

        Returns:
            str: The child's answer, or a note explaining why it did not run.
        '''
        if not isinstance(task, str) or not task.strip():
            return '[delegate error] an empty task cannot be delegated'
        level = self.depth if depth is None else int(depth)
        if level >= self.max_depth:
            return ('[delegate limit] nesting reached %d level(s); solve the task '
                    'directly instead' % self.max_depth)
        if not self._slots.acquire(blocking=False):
            return '[delegate error] too many sub-agents are running; retry later'
        token = _DEPTH.set(level + 1)
        child = None
        try:
            child = self._build_child(name)
            if (level + 1 < self.max_depth
                    and self.tool_name not in getattr(child, 'tools', {})
                    and callable(getattr(child, 'add_tool', None))):
                child.add_tool(self.tool(depth=level + 1))
            logger.info('delegating to %s (%d char task)', getattr(child, 'name', '?'),
                        len(task))
            answer = child.ask(task, timeout=self.timeout,
                               max_steps=self.max_steps)
        except Exception as exc:  # noqa: BLE001 - report instead of killing the run
            logger.warning('delegate to %s failed: %s',
                           getattr(child, 'name', '?') if child is not None else '?', exc)
            return '[delegate error] %s: %s' % (type(exc).__name__, exc)
        finally:
            self._slots.release()
            _DEPTH.reset(token)
        return str(answer)

    async def adelegate(self, task: str, name: str = '') -> str:
        '''Async wrapper around :meth:`delegate` for loop callers.

        Args:
            task (str): Self-contained instruction.
            name (str): Optional child name.

        Returns:
            str: The child's answer.
        '''
        import asyncio

        return await asyncio.get_running_loop().run_in_executor(
            None, self.delegate, task, name, None)

    def tool(self, depth: Optional[int] = None) -> Tool:
        '''Build the ``delegate`` tool bound to a nesting level.

        Args:
            depth (Optional[int]): Level of the agent that will hold this tool;
                defaults to the current depth.

        Returns:
            Tool: A tool named ``tool_name`` taking ``task`` and ``name``.
        '''
        level = self.depth if depth is None else int(depth)

        def delegate(task: str, name: str = '') -> str:
            '''Run a sub-agent on a self-contained task and return its answer.

            Use this when a piece of work needs focused attention, a different
            skill set, or a separate conversation. Give the sub-agent everything
            it needs in ``task``; it cannot see the current conversation.

            Args:
                task (str): Complete instruction for the sub-agent.
                name (str): Optional name for the sub-agent.

            Returns:
                str: The sub-agent's answer.
            '''
            return self.delegate(task, name, depth=level)

        existing = self._tools.get(level)
        if existing is not None and existing.name == self.tool_name:
            return existing
        built = tool(
            name=self.tool_name,
            description='Run a sub-agent on a self-contained task and return its answer.',
            require_approval=self.require_approval,
            cache=False,
        )(delegate)
        self._tools[level] = built
        return built

    def snapshot(self) -> dict:
        '''Return counters as a plain dict.

        Returns:
            dict: Config plus children created.
        '''
        return {
            'max_depth': self.max_depth,
            'max_concurrent': self.max_concurrent,
            'max_steps': self.max_steps,
            'children': list(self.names),
            'depth': self.depth,
        }
