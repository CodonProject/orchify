'''Context-window plugin: keep prompts inside a token budget.

The plugin installs an LLM middleware that measures every outgoing prompt with
:mod:`orchify.tokens` and, when it would overflow the budget, drops the oldest
whole turns and leaves a one-message marker in their place. It never splits a
tool call from its tool results, and it always keeps the newest turn, so the
model still sees the request it is answering.

The plugin is opt-in in two ways:

* register it explicitly (``orchify_plugins.register(ContextWindowPlugin(...))``)
  and enable it per agent (``Agent(..., plugins=['context_window'])``), and
* it only acts on agents that enabled it, because plugin middleware is skipped
  while the plugin is not in ``Agent.enabled_plugins``.

Example:
    from orchify.plugins import ContextWindowPlugin
    from orchify import Agent, orchify_plugins

    orchify_plugins.register(ContextWindowPlugin(max_prompt_tokens=16000))
    agent = Agent(name='assistant', plugins=['context_window'])
'''

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from orchify.llm.base import Middleware
from orchify.plugin import Plugin
from orchify.tokens import estimate_message, estimate_messages, estimate_tools

DEFAULT_MAX_PROMPT_TOKENS = 32000
DEFAULT_RESERVE_TOKENS = 1024
DEFAULT_MARKER = '[context compacted: {count} earlier message(s), about {tokens} tokens, were dropped to fit the context window]'


@dataclass
class TrimResult:
    '''Outcome of one compaction pass.

    Attributes:
        messages (List[dict]): The messages that fit the budget.
        dropped (int): How many messages were removed.
        dropped_tokens (int): Estimated tokens the removed messages occupied.
        before (int): Estimated prompt tokens before compaction.
        after (int): Estimated prompt tokens after compaction.
        budget (int): Token budget the prompt was fitted into.
        changed (bool): True when ``messages`` differs from the input.
    '''

    messages: List[dict] = field(default_factory=list)
    dropped: int = 0
    dropped_tokens: int = 0
    before: int = 0
    after: int = 0
    budget: int = 0
    changed: bool = False

    def as_dict(self) -> Dict[str, Any]:
        '''Return the result as a plain dict.

        Returns:
            Dict[str, Any]: Field values, ready for events or logs.
        '''
        return {
            'dropped': self.dropped,
            'dropped_tokens': self.dropped_tokens,
            'before': self.before,
            'after': self.after,
            'budget': self.budget,
            'changed': self.changed,
        }


def _is_role(message: Any, role: str) -> bool:
    '''True when ``message`` is a dict with the given role.'''
    return isinstance(message, dict) and message.get('role') == role


def split_system_prefix(messages: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    '''Split leading system messages from the rest of the conversation.

    Args:
        messages (Sequence[dict]): Chat messages in OpenAI format.

    Returns:
        Tuple[List[dict], List[dict]]: The leading system messages and the
        remaining messages.
    '''
    index = 0
    for message in messages:
        if not _is_role(message, 'system'):
            break
        index += 1
    return list(messages[:index]), list(messages[index:])


def group_turns(messages: Sequence[dict]) -> List[List[dict]]:
    '''Group messages into turns that must stay together.

    A turn starts at a ``user`` message (or a system message in the middle of the
    conversation) and swallows everything after it until the next such message,
    which keeps an assistant message with ``tool_calls`` attached to the ``tool``
    results that answer it.

    Args:
        messages (Sequence[dict]): Messages without the leading system prefix.

    Returns:
        List[List[dict]]: Non-empty groups, oldest first.
    '''
    turns: List[List[dict]] = []
    current: List[dict] = []
    for message in messages:
        role = message.get('role') if isinstance(message, dict) else None
        if (role == 'user' or role == 'system') and current:
            turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)
    return turns


def _content_text(message: dict) -> str:
    '''Best-effort string content of one message.'''
    content = message.get('content')
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    return str(content)


def _prepend_marker(messages: List[dict], marker: str) -> List[dict]:
    '''Attach ``marker`` to the oldest kept message without adding a new role.

    Prepending the note to an existing ``user`` message keeps the conversation
    valid for providers that reject two user messages in a row, and it survives
    ``control:update_messages`` with ``op='replace'`` because it stays in the
    message the agent keeps.
    '''
    if not messages:
        return messages
    if _is_role(messages[0], 'user') and isinstance(messages[0].get('content'), str):
        first = dict(messages[0])
        first['content'] = f'{marker}\n\n{first["content"]}'
        return [first] + messages[1:]
    return [{'role': 'user', 'content': marker}] + messages


class ContextWindowPlugin(Plugin):
    '''Compaction plugin: fit prompts into ``max_prompt_tokens``.

    Attributes:
        max_prompt_tokens (int): Budget for one outgoing prompt, tool schemas
            included.
        reserve_tokens (int): Headroom kept free for the response and for
            estimate error.
        keep_recent (int): Number of newest turns that survive compaction even
            when the budget says otherwise; 0 is allowed and still keeps the
            newest turn.
        persist (bool): When True the compacted history is pushed back to the
            agent with ``control:update_messages`` so it does not grow again.
        marker (str): Template for the placeholder note, with ``{count}`` and
            ``{tokens}`` placeholders.
        stats (Dict[str, Any]): Running counters for observability.
    '''

    name = 'context_window'
    version = '1.0.0'
    description = 'Drop the oldest turns of a prompt that would exceed the context window.'
    tags = ('context', 'cost', 'prompt')

    def __init__(
        self,
        broker=None,
        *,
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
        keep_recent: int = 0,
        keep_system: bool = True,
        marker: str = DEFAULT_MARKER,
        persist: bool = False,
        min_savings: int = 256,
    ):
        '''Create the plugin.

        Args:
            broker (Optional[Broker]): Broker to attach to; defaults to the shared
                one.
            max_prompt_tokens (int): Prompt budget in tokens.
            reserve_tokens (int): Tokens kept free on top of the budget.
            keep_recent (int): Newest turns never dropped, on top of the newest
                one.
            keep_system (bool): Always keep the leading system prompt.
            marker (str): Placeholder text for dropped turns.
            persist (bool): Mirror the compaction into the agent's history.
            min_savings (int): Skip the rewrite when fewer tokens than this would
                be saved (avoids churn near the boundary).
        '''
        super().__init__(broker=broker)
        self.max_prompt_tokens = max(1, int(max_prompt_tokens))
        self.reserve_tokens = max(0, int(reserve_tokens))
        self.keep_recent = max(0, int(keep_recent))
        self.keep_system = bool(keep_system)
        self.marker = marker
        self.persist = bool(persist)
        self.min_savings = max(0, int(min_savings))
        self.stats: Dict[str, Any] = {
            'requests': 0,
            'compacted': 0,
            'dropped_messages': 0,
            'saved_tokens': 0,
            'last': {},
        }
        self._middleware = _ContextMiddleware(self)
        self._persisted: Dict[str, int] = {}

    @property
    def budget(self) -> int:
        '''Effective prompt budget in tokens.'''
        return max(1, self.max_prompt_tokens - self.reserve_tokens)

    def on_load(self) -> None:
        '''Install the middleware on the shared LLM, or defer until ``bind``.'''
        self.middleware(self._middleware)

    def compact(self, messages: Sequence[dict], tools: Optional[list] = None) -> TrimResult:
        '''Fit ``messages`` into the budget by dropping whole oldest turns.

        Args:
            messages (Sequence[dict]): Prompt messages in OpenAI format.
            tools (Optional[list]): Tool schemas that ride along with the prompt.

        Returns:
            TrimResult: The fitted list and what it cost.
        '''
        budget = self.budget - estimate_tools(tools)
        message_list = list(messages)
        # One estimation pass: every message is measured once and the numbers are
        # reused for the total, the per-turn costs, the dropped total and the final
        # size. Measuring per turn (and again for `before`/`after`) tripled the
        # work, and `estimate_message` above all json.dumps the tool calls.
        costs = [estimate_message(message) for message in message_list]
        before = sum(costs)
        result = TrimResult(messages=message_list, before=before, budget=budget, after=before)
        if not message_list or before <= budget:
            return result

        system, rest = split_system_prefix(message_list) if self.keep_system else ([], message_list)
        system_tokens = sum(costs[:len(system)]) if self.keep_system else 0
        cost_by_id = {}
        for message, cost in zip(message_list, costs):
            cost_by_id[id(message)] = cost
        turns = group_turns(rest)
        if not turns:
            return result

        keep_at_least = 1 + self.keep_recent
        kept: List[List[dict]] = []
        used = system_tokens
        for index, turn in enumerate(reversed(turns)):
            cost = 0
            for message in turn:
                cost += cost_by_id.get(id(message)) or 0
            forced = len(kept) < keep_at_least
            if not forced and used + cost > budget:
                break
            kept.append(turn)
            used += cost
        kept.reverse()
        dropped_turns = turns[:len(turns) - len(kept)]
        if not dropped_turns:
            return result

        kept_messages = [message for turn in kept for message in turn]
        dropped_messages = [message for turn in dropped_turns for message in turn]
        dropped_tokens = before - used
        marker = self.marker.format(count=len(dropped_messages), tokens=dropped_tokens)
        fitted = _prepend_marker(kept_messages, marker)
        after = system_tokens + (used - system_tokens)
        if fitted:
            # The marker either rewrote the oldest kept message (same length) or
            # arrived as its own message (one extra entry); only the first case
            # replaces the cost that is already counted in ``after``.
            after += estimate_message(fitted[0])
            if len(fitted) == len(kept_messages):
                after -= cost_by_id.get(id(kept_messages[0])) or 0
        if before - after < self.min_savings:
            return result

        result.messages = system + fitted
        result.dropped = len(dropped_messages)
        result.dropped_tokens = dropped_tokens
        result.after = after
        result.changed = True
        return result

    def trim(self, kwargs: Dict[str, Any]) -> TrimResult:
        '''Compact one in-flight request and record the outcome.

        Args:
            kwargs (Dict[str, Any]): Request kwargs from the middleware chain; its
                ``messages`` entry is replaced in place when compaction applies.

        Returns:
            TrimResult: What was done for this request.
        '''
        messages = kwargs.get('messages') or []
        result = self.compact(messages, kwargs.get('tools'))
        self.stats['requests'] += 1
        self.stats['last'] = result.as_dict()
        if not result.changed:
            return result
        kwargs['messages'] = result.messages
        self.stats['compacted'] += 1
        self.stats['dropped_messages'] += result.dropped
        self.stats['saved_tokens'] += max(0, result.before - result.after)
        self._maybe_persist(kwargs.get('run_id'), result)
        return result

    def _maybe_persist(self, run_id: Optional[str], result: TrimResult) -> None:
        '''Push the compacted history back to the agent when ``persist`` is set.'''
        if not self.persist or not run_id:
            return
        total = len(result.messages)
        if self._persisted.get(run_id) == result.dropped:
            return
        kept = [message for message in result.messages if not _is_role(message, 'system')]
        queued = self.feedback(
            'control:update_messages',
            run_id=run_id,
            payload={'op': 'replace', 'messages': kept},
        )
        if queued:
            self._persisted[run_id] = result.dropped
        self.state.setdefault('persisted', {})[run_id] = bool(queued)
        self.log(f'compacted run {run_id}: kept {total} message(s), dropped {result.dropped}')

    def cleanup(self) -> None:
        '''Remove the middleware from every LLM it was installed on.'''
        self._detach_middleware(self._middleware)
        super().cleanup()

    def snapshot(self) -> Dict[str, Any]:
        '''Return the counters as a plain dict.

        Returns:
            Dict[str, Any]: Compaction statistics.
        '''
        data = dict(self.stats)
        data['budget'] = self.budget
        data['keep_recent'] = self.keep_recent
        data['persist'] = self.persist
        return data


class _ContextMiddleware(Middleware):
    '''Middleware adapter that compacts the prompt before it is sent.'''

    def __init__(self, plugin: ContextWindowPlugin):
        '''Wrap the plugin as an LLM middleware.

        Args:
            plugin (ContextWindowPlugin): Owner of the compaction policy.
        '''
        super().__init__(name='context_window', scope=plugin.scope)
        self.plugin = plugin

    async def __call__(self, kwargs, next_request):
        '''Compact ``kwargs['messages']`` and stream the inner response.

        Args:
            kwargs (Dict[str, Any]): Request kwargs.
            next_request (Callable): The rest of the middleware chain.

        Yields:
            Response: Response items from the inner chain.
        '''
        self.plugin.trim(kwargs)
        async for response in next_request(kwargs):
            yield response
