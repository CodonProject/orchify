'''Tool permission policy: decide allow / ask / deny before a tool runs.

The agent's own ``require_approval`` flag answers a yes/no question. A policy
answers a three-way one: run it, ask me first, or never run it. Rules are
matched in order against the tool name (glob) and, optionally, the arguments,
which is what makes "allow ``read_file`` anywhere but ``write_file`` only under
``docs/``" expressible.

Example:
    from orchify import Agent
    from orchify.policy import ToolPolicy, ToolRule

    policy = ToolPolicy([
        ToolRule('read_*', 'allow'),
        ToolRule('write_file', 'ask', args={'path': 'src/*'}),
        ToolRule('write_file', 'deny'),
    ])
    agent = Agent(name='editor', tool_policy=policy, tools=[read_file, write_file])
'''

import re
from fnmatch import translate
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

ALLOW = 'allow'
ASK = 'ask'
DENY = 'deny'

ACTIONS = (ALLOW, ASK, DENY)


@lru_cache(maxsize=512)
def _compile_glob(pattern: str):
    '''Compile one glob into a regex, caching the result like :mod:`fnmatch`.'''
    return re.compile(translate(pattern))


def _compile_globs(patterns: Sequence[str]):
    '''Compile several globs into one alternation, or None for an empty set.'''
    unique = tuple(dict.fromkeys(str(pattern) for pattern in patterns))
    if not unique:
        return None
    if len(unique) == 1:
        return _compile_glob(unique[0])
    return re.compile('|'.join('(?:%s)' % translate(pattern) for pattern in unique))


def _as_tuple(value: Any) -> tuple:
    '''Normalise a rule's ``tools`` value to a tuple of globs.'''
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _args_match(args: Any, expected: Dict[str, Any]) -> bool:
    '''Whether every ``key: glob`` pair matches ``args``.

    A value that is not a string is compared with ``==``; a string is compared
    with :func:`fnmatch.fnmatchcase`, so ``'src/*'`` matches every path below
    ``src``. A missing key never matches.
    '''
    if not expected:
        return True
    if not isinstance(args, dict):
        return False
    for key, wanted in expected.items():
        if key not in args:
            return False
        actual = args.get(key)
        if isinstance(wanted, str):
            if not isinstance(actual, str) or _compile_glob(wanted).match(actual) is None:
                return False
        elif actual != wanted:
            return False
    return True



class ToolRule:
    '''One policy rule: a tool glob, an action, and optional argument filters.

    Attributes:
        tools (tuple): Tool-name globs the rule applies to; empty matches any.
        action (str): ``'allow'``, ``'ask'`` or ``'deny'``.
        args (Dict[str, Any]): Argument globs that must all match.
        match (Optional[Callable]): Extra predicate ``(tool_name, args) -> bool``.
        reason (str): Shown to the user or the model when the rule denies.
    '''

    def __init__(self, tools: Union[str, Sequence[str], None] = None,
                 action: str = ASK, *, args: Optional[Dict[str, Any]] = None,
                 match: Optional[Callable[[str, Any], bool]] = None,
                 reason: str = ''):
        '''Create a rule.

        Args:
            tools (Union[str, Sequence[str], None]): Tool glob(s); None matches any
                tool.
            action (str): ``'allow'``, ``'ask'`` or ``'deny'``.
            args (Optional[Dict[str, Any]]): Argument globs that must all match.
            match (Optional[Callable[[str, Any], bool]]): Extra predicate.
            reason (str): Explanation attached to the decision.

        Raises:
            ValueError: When ``action`` is not one of the three actions.
        '''
        normalised = str(action or '').strip().lower()
        if normalised not in ACTIONS:
            raise ValueError(f'Invalid policy action {action!r}; expected one of {ACTIONS}.')
        self.tools = _as_tuple(tools)
        self.action = normalised
        self.args: Dict[str, Any] = dict(args or {})
        self.match = match
        self.reason = reason
        # Globs are compiled once here, not per call: a policy with N rules used
        # to cost N fnmatch lookups (each one a cache hit plus a regex match) for
        # every tool call the model makes. A rule is immutable after
        # construction, so mutating ``args`` in place is not supported.
        self._tool_rx = _compile_globs(self.tools)
        self._arg_rx = {
            key: _compile_glob(value)
            for key, value in self.args.items()
            if isinstance(value, str)
        }

    def _match_args(self, args: Any) -> bool:
        '''Fast path of :func:`_args_match` that reuses the compiled globs.'''
        if not isinstance(args, dict):
            return False
        for key, wanted in self.args.items():
            if key not in args:
                return False
            actual = args[key]
            compiled = self._arg_rx.get(key)
            if compiled is not None:
                if not isinstance(actual, str) or compiled.match(actual) is None:
                    return False
            elif actual != wanted:
                return False
        return True

    def matches(self, tool_name: str, args: Any = None) -> bool:
        '''Whether this rule applies to a call.

        Args:
            tool_name (str): Tool being called.
            args (Any): Parsed arguments, usually a dict.

        Returns:
            bool: True when every condition of the rule holds.
        '''
        if self._tool_rx is not None:
            if self._tool_rx.match(str(tool_name or '')) is None:
                return False
        if self.args and not self._match_args(args):
            return False
        if self.match is not None and not self.match(tool_name, args):
            return False
        return True

    def __repr__(self) -> str:
        '''Readable form for logs.'''
        return f'ToolRule({self.tools!r}, {self.action!r}, args={self.args!r})'


class Decision:
    '''The outcome of a policy evaluation.

    Attributes:
        action (str): ``'allow'``, ``'ask'`` or ``'deny'``.
        reason (str): Why the decision was taken.
        rule (Optional[ToolRule]): The rule that produced it, when any.
        tool_name (str): Tool the decision is about.
    '''

    __slots__ = ('action', 'reason', 'rule', 'tool_name')

    def __init__(self, action: str, reason: str = '', rule: Optional[ToolRule] = None,
                 tool_name: str = ''):
        '''Create a decision.

        Args:
            action (str): ``'allow'``, ``'ask'`` or ``'deny'``.
            reason (str): Explanation.
            rule (Optional[ToolRule]): Rule that matched.
            tool_name (str): Tool the decision is about.
        '''
        self.action = action
        self.reason = reason
        self.rule = rule
        self.tool_name = tool_name

    @property
    def allowed(self) -> bool:
        '''True when the call may proceed without asking.'''
        return self.action == ALLOW

    @property
    def denied(self) -> bool:
        '''True when the call must not run.'''
        return self.action == DENY

    @property
    def needs_approval(self) -> bool:
        '''True when a human should decide.'''
        return self.action == ASK

    def as_dict(self) -> Dict[str, Any]:
        '''Plain-dict view for events and logs.

        Returns:
            Dict[str, Any]: Action, reason, rule and tool name.
        '''
        return {'action': self.action, 'reason': self.reason,
                'rule': repr(self.rule) if self.rule is not None else '',
                'tool_name': self.tool_name}

    def __repr__(self) -> str:
        '''Readable form for logs.'''
        return f'Decision({self.tool_name!r}, {self.action!r}, {self.reason!r})'


class ToolPolicy:
    '''Ordered rules plus a default action for anything they do not cover.

    Attributes:
        rules (List[ToolRule]): Rules in evaluation order.
        default (str): Action for a tool no rule matches; defaults to ``'allow'``
            so an empty policy changes nothing.
    '''

    def __init__(self, rules: Optional[Iterable[Union[ToolRule, tuple, dict]]] = None,
                 *, default: str = ALLOW, deny_reason: str = ''):
        '''Create a policy.

        Args:
            rules (Optional[Iterable]): :class:`ToolRule` instances, or ``(tools,
                action)`` tuples / ``{'tools': ..., 'action': ...}`` dicts.
            default (str): ``'allow'``, ``'ask'`` or ``'deny'`` for unmatched tools.
            deny_reason (str): Message used when the default is ``'deny'``.

        Raises:
            ValueError: When ``default`` is not a valid action.
        '''
        normalised = str(default or '').strip().lower()
        if normalised not in ACTIONS:
            raise ValueError(f'Invalid policy default {default!r}; expected one of {ACTIONS}.')
        self.default = normalised
        self.deny_reason = deny_reason
        self.rules: List[ToolRule] = [self._coerce(rule) for rule in (rules or ())]

    @staticmethod
    def _coerce(rule: Union[ToolRule, tuple, dict, str]) -> ToolRule:
        '''Accept a rule instance, a ``(tools, action)`` pair or a dict.'''
        if isinstance(rule, ToolRule):
            return rule
        if isinstance(rule, str):
            return ToolRule(rule, ASK)
        if isinstance(rule, dict):
            data = dict(rule)
            tools = data.pop('tools', data.pop('tool', None))
            action = data.pop('action', ASK)
            return ToolRule(tools, action, **data)
        if isinstance(rule, (tuple, list)) and len(rule) >= 2:
            tools, action = rule[0], rule[1]
            rest = dict(rule[2]) if len(rule) > 2 and isinstance(rule[2], dict) else {}
            return ToolRule(tools, action, **rest)
        raise ValueError(f'Cannot build a ToolRule from {rule!r}.')

    def add(self, rule: Union[ToolRule, tuple, dict, str]) -> ToolRule:
        '''Append a rule and return it.

        Args:
            rule (Union[ToolRule, tuple, dict, str]): Rule to add.

        Returns:
            ToolRule: The normalised rule.
        '''
        normalised = self._coerce(rule)
        self.rules.append(normalised)
        return normalised

    def decide(self, tool_name: str, args: Any = None) -> Decision:
        '''Evaluate the rules for one call.

        Args:
            tool_name (str): Tool being called.
            args (Any): Parsed arguments.

        Returns:
            Decision: The first matching rule's action, or the default.
        '''
        for rule in self.rules:
            if rule.matches(tool_name, args):
                reason = rule.reason or f'policy rule {rule.tools or "*"} -> {rule.action}'
                return Decision(rule.action, reason, rule, tool_name)
        if self.default == ALLOW:
            return Decision(ALLOW, 'no policy rule matched', None, tool_name)
        reason = self.deny_reason or f'no policy rule matched; default is {self.default}'
        return Decision(self.default, reason, None, tool_name)

    def allow(self, tool_name: str, args: Any = None) -> bool:
        '''True when the call is allowed outright.

        Args:
            tool_name (str): Tool being called.
            args (Any): Parsed arguments.

        Returns:
            bool: ``decide(...).allowed``.
        '''
        return self.decide(tool_name, args).allowed

    def as_dict(self) -> Dict[str, Any]:
        '''Plain-dict view of the policy.

        Returns:
            Dict[str, Any]: Default plus every rule in order.
        '''
        return {
            'default': self.default,
            'deny_reason': self.deny_reason,
            'rules': [{'tools': list(rule.tools), 'action': rule.action,
                       'args': dict(rule.args), 'reason': rule.reason}
                      for rule in self.rules],
        }


DEFAULT_POLICY_ENV = 'ORCHIFY_TOOL_POLICY'


def policy_from_env(environ: Optional[Dict[str, str]] = None) -> Optional[ToolPolicy]:
    '''Build a policy from ``ORCHIFY_TOOL_POLICY``.

    The value is a comma-separated list of ``tool_glob=action`` entries, for
    example ``read_*=allow,write_*=ask,bash=deny``. Unparsable entries are
    skipped, so a typo cannot silently open a tool.

    Args:
        environ (Optional[Dict[str, str]]): Environment mapping; defaults to
            ``os.environ``.

    Returns:
        Optional[ToolPolicy]: The policy, or None when the variable is unset or
        yields no usable rule.
    '''
    import os

    env = os.environ if environ is None else environ
    raw = str(env.get(DEFAULT_POLICY_ENV, '') or '').strip()
    if not raw:
        return None
    rules: List[ToolRule] = []
    for chunk in raw.split(','):
        entry = chunk.strip()
        if not entry or '=' not in entry:
            continue
        pattern, action = entry.rsplit('=', 1)
        pattern, action = pattern.strip(), action.strip().lower()
        if not pattern or action not in ACTIONS:
            continue
        rules.append(ToolRule(pattern, action))
    if not rules:
        return None
    return ToolPolicy(rules, default=ALLOW)


__all__ = ['ToolPolicy', 'ToolRule', 'Decision', 'policy_from_env',
           'ALLOW', 'ASK', 'DENY', 'ACTIONS']
