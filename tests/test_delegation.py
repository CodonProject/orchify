"""Runtime sub-agent delegation."""

import json

from orchify import Agent, Delegation
from orchify.delegation import DelegationError
from orchify.testing import FakeLLM, final, tool_call


def _q(text):
    '''JSON-encode a short string for a tool-call argument.'''
    return json.dumps(text)


def _delegate_plan(task='say hello', then='done'):
    '''Plan (one request per step): call the delegate tool, then answer.'''
    first = {'final': {'content': '', 'reasoning': '',
                       'tool_calls': [tool_call('delegate',
                                                '{"task": %s}' % _q(task))]}}
    second = {'final': {'content': then, 'reasoning': '', 'tool_calls': []}}
    return [[first], [second]]


def test_delegation_runs_a_child_agent_and_returns_its_answer():
    '''The parent's delegate tool returns the child answer.'''
    calls = []

    def factory(name):
        calls.append(name)
        return Agent(name=name, llm=FakeLLM([[final('child answer')]]))

    delegation = Delegation(factory=factory)
    agent = Agent(name='lead', llm=FakeLLM(_delegate_plan('look up the docs')),
                  tools=[delegation.tool()])
    assert agent.ask('coordinate', max_steps=2) == 'done'

    tool_message = [m for m in agent.messages if m.get('role') == 'tool'][-1]
    assert tool_message['content'] == 'child answer'
    assert calls == ['helper-1']
    assert delegation.names == ['helper-1']


def test_delegation_stops_at_the_depth_limit():
    '''A nested delegation returns a note instead of recursing forever.'''
    captured = []

    def factory(name):
        child = Agent(name=name, llm=FakeLLM(_delegate_plan('deeper', 'child done')),
                      tools=[delegation.tool()])
        captured.append(child)
        return child

    delegation = Delegation(factory=factory, max_depth=1)
    result = delegation.delegate('first level')

    # The first level ran (depth 0 -> 1); its child tried again and was refused.
    assert result == 'child done'
    assert len(captured) == 1
    tool_message = [m for m in captured[0].messages if m.get('role') == 'tool'][-1]
    assert 'delegate limit' in tool_message['content']
    assert delegation.depth == 0


def test_empty_tasks_are_rejected_without_building_an_agent():
    '''An empty task is a tool error, not a model call.'''
    built = []
    delegation = Delegation(factory=lambda name: built.append(name) or None)
    assert delegation.delegate('   ') == '[delegate error] an empty task cannot be delegated'
    assert built == []


def test_a_failing_child_is_reported_as_tool_text():
    '''Child errors become tool text so the parent can keep going.'''

    def factory(name):
        class Broken(object):
            def __init__(self):
                self.name = name

            def ask(self, *args, **kwargs):
                raise RuntimeError('child exploded')

        return Broken()

    delegation = Delegation(factory=factory)
    result = delegation.delegate('do it')
    assert result == '[delegate error] RuntimeError: child exploded'
    assert delegation.depth == 0


def test_clone_template_shares_llm_and_drops_the_delegate_tool():
    '''The template path clones without copying the delegate tool.'''
    parent = Agent(name='lead', llm=FakeLLM([[final('ok')]]))
    delegation = Delegation(agent=parent, prefix='worker')
    parent.add_tool(delegation.tool())

    child = delegation.build()
    assert child.name == 'worker-1'
    assert child.llm is parent.llm
    assert 'delegate' not in child.tools
    assert delegation.snapshot()['children'] == ['worker-1']


def test_delegation_needs_a_factory_or_template():
    '''Configuration errors are explicit.'''
    try:
        Delegation()
    except DelegationError as exc:
        assert 'factory=' in str(exc)
    else:  # pragma: no cover - the constructor must raise
        raise AssertionError('Delegation() should have raised')


def test_build_refuses_past_the_limit():
    '''build() raises rather than silently truncating.'''
    from orchify.delegation import _DEPTH

    delegation = Delegation(
        factory=lambda name: Agent(name=name, llm=FakeLLM([[final('x')]])),
        max_depth=1)
    token = _DEPTH.set(1)
    try:
        try:
            delegation.build()
        except DelegationError as exc:
            assert 'delegation limit reached' in str(exc)
        else:  # pragma: no cover
            raise AssertionError('build() should have raised at the limit')
    finally:
        _DEPTH.reset(token)
