'''Summarizing context plugin: condense old turns instead of dropping them.

The companion plugin in :mod:`orchify.plugins.context` fits a prompt into the
budget by *dropping* the oldest turns and leaving a one-line marker. That keeps
the request small but the model loses what happened. This plugin keeps the
information by rewriting the dropped turns into a short recap, so a long
conversation can keep going without the old messages.

It is opt-in, like every plugin::

    from orchify import Agent, orchify_plugins, FakeLLM
    from orchify.plugins.summarize import SummarizingContextPlugin, Summarizer

    plugin = SummarizingContextPlugin(
        summarizer=Summarizer(llm=Agent.from_env().llm),
        max_prompt_tokens=16000,
        keep_recent=2,
    )
    orchify_plugins.register(plugin)
    agent = Agent(name='writer', llm=FakeLLM([[final('ok')]]), plugins=['summarizing_context'])

The summarizer call goes through ``llm.request``, so it is billed like any
other request: point it at a cheap model when you care about cost.
'''

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence

from orchify.llm.base import Middleware
from orchify.plugin import Plugin
from orchify.plugins.context import (
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_RESERVE_TOKENS,
    TrimResult,
    group_turns,
    split_system_prefix,
)
from orchify.tokens import estimate_message, estimate_tools

DEFAULT_SUMMARY_PROMPT = (
    'You compress conversation history for another assistant. Keep every fact, '
    'decision, tool result, file path, number and open question; drop small '
    'talk and repeated wording. Reply with the recap only, in at most 200 words.'
)
DEFAULT_MARKER = '[summary of {count} earlier message(s), about {tokens} tokens]'


class Summarizer:
    '''Rewrites a list of messages into one short recap.

    Attributes:
        llm (Any): Provider used for the summary call.
        model (str): Model override; the provider default is used when empty.
        max_tokens (int): Cap for the recap.
        prompt (str): System instruction sent with the recap request.
        scope (str): Middleware scope of the summary call.
    '''

    def __init__(
        self,
        llm: Any = None,
        *,
        model: str = '',
        max_tokens: int = 512,
        prompt: str = DEFAULT_SUMMARY_PROMPT,
        scope: str = '*',
    ):
        '''Create a summarizer.

        Args:
            llm (Any): Provider to call; without one the owning plugin falls back
                to a local placeholder recap.
            model (str): Model override.
            max_tokens (int): Maximum recap length.
            prompt (str): System instruction for the recap.
            scope (str): Scope passed to the provider request.
        '''
        self.llm = llm
        self.model = model
        self.max_tokens = max(32, int(max_tokens))
        self.prompt = prompt
        self.scope = scope

    def render(self, messages: Sequence[dict]) -> str:
        '''Flatten messages into plain text for the summarizer prompt.

        Args:
            messages (Sequence[dict]): Prompt messages in OpenAI format.

        Returns:
            str: One block of ``role: text`` lines; tool calls keep their name and
            arguments and tool results keep their content.
        '''
        lines: List[str] = []
        for message in messages:
            role = str(message.get('role') or 'user')
            content = message.get('content')
            if isinstance(content, str):
                text = content
            elif content is None:
                text = ''
            else:
                text = json.dumps(content, ensure_ascii=False, default=str)
            calls = message.get('tool_calls') or []
            if calls:
                rendered = []
                for call in calls:
                    function = call.get('function') or {}
                    rendered.append('%s(%s)' % (
                        function.get('name') or 'tool',
                        function.get('arguments') or '{}',
                    ))
                text = (text + ' ' if text else '') + 'calls: ' + '; '.join(rendered)
            if not text:
                continue
            lines.append(f'{role}: {text}')
        return '\n'.join(lines)

    async def asummarize(self, messages: Sequence[dict]) -> str:
        '''Ask the provider for a recap of ``messages``.

        Args:
            messages (Sequence[dict]): The turns about to be dropped.

        Returns:
            str: The recap text, or an empty string when no provider is
            configured or the call fails.
        '''
        llm = self.llm
        if llm is None or not messages:
            return ''
        prompt = [
            {'role': 'system', 'content': self.prompt},
            {'role': 'user', 'content': self.render(messages)},
        ]
        model = self.model or str(getattr(llm, 'model', '') or '')
        text = ''
        try:
            async for response in llm.request(
                messages=prompt,
                model=model,
                max_tokens=self.max_tokens,
                scope=self.scope,
            ):
                chunk = getattr(response, 'current_chunk', None)
                status = getattr(response, 'final_status', None)
                if status is not None and getattr(status, 'content', ''):
                    text = status.content
                elif chunk is not None and getattr(chunk, 'total_content', ''):
                    text = chunk.total_content
        except Exception:  # noqa: BLE001 - a failed recap must not kill the run
            return ''
        return text.strip()

    def summarize(self, messages: Sequence[dict]) -> str:
        '''Blocking wrapper around :meth:`asummarize`.

        Args:
            messages (Sequence[dict]): The turns to compress.

        Returns:
            str: The recap text, or an empty string.
        '''
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.asummarize(messages))
        raise RuntimeError('summarize() cannot run inside a loop; await asummarize().')


class SummarizingContextPlugin(Plugin):
    '''Compaction plugin that recaps old turns instead of dropping their text.

    Attributes:
        max_prompt_tokens (int): Budget for one outgoing prompt.
        reserve_tokens (int): Headroom kept free for the response.
        keep_recent (int): Newest turns that always survive.
        keep_system (bool): Keep the leading system prompt.
        min_savings (int): Skip rewrites that save less than this.
        persist (bool): Push the rewritten history back to the agent.
        marker (str): Template for the recap header.
        stats (Dict[str, Any]): Running counters.
    '''

    name = 'summarizing_context'
    version = '1.0.0'
    description = 'Condense old turns into a summary instead of dropping them.'
    tags = ('context', 'cost', 'prompt', 'summary')

    def __init__(
        self,
        broker=None,
        *,
        summarizer: Optional[Summarizer] = None,
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
        keep_recent: int = 2,
        keep_system: bool = True,
        marker: str = DEFAULT_MARKER,
        persist: bool = False,
        min_savings: int = 256,
    ):
        '''Create the plugin.

        Args:
            broker (Optional[Broker]): Broker to attach to; defaults to the shared
                one.
            summarizer (Optional[Summarizer]): Recap provider; when omitted the
                plugin uses its own LLM (set by ``PluginManager``/``bind``) and
                degrades to a placeholder recap without one.
            max_prompt_tokens (int): Prompt budget in tokens.
            reserve_tokens (int): Tokens kept free on top of the budget.
            keep_recent (int): Newest turns never compressed.
            keep_system (bool): Always keep the leading system prompt.
            marker (str): Header for the recap message.
            persist (bool): Mirror the rewrite into the agent's history.
            min_savings (int): Minimum token saving before rewriting.
        '''
        super().__init__(broker=broker)
        self.summarizer = summarizer
        self.max_prompt_tokens = max(1, int(max_prompt_tokens))
        self.reserve_tokens = max(0, int(reserve_tokens))
        self.keep_recent = max(0, int(keep_recent))
        self.keep_system = bool(keep_system)
        self.marker = marker
        self.persist = bool(persist)
        self.min_savings = max(0, int(min_savings))
        self.stats: Dict[str, Any] = {
            'requests': 0,
            'summarized': 0,
            'summarized_messages': 0,
            'saved_tokens': 0,
            'fallbacks': 0,
            'last': {},
        }
        self._persisted: Dict[str, int] = {}
        self._middleware = _SummaryMiddleware(self)

    @property
    def budget(self) -> int:
        '''Effective prompt budget in tokens.'''
        return max(1, self.max_prompt_tokens - self.reserve_tokens)

    def _provider(self) -> Optional[Summarizer]:
        '''Resolve the summarizer, falling back to the plugin LLM.'''
        if self.summarizer is not None:
            if self.summarizer.llm is None:
                self.summarizer.llm = self._llm
            return self.summarizer
        if self._llm is not None:
            self.summarizer = Summarizer(llm=self._llm)
            return self.summarizer
        return None

    def on_load(self) -> None:
        '''Install the middleware on the shared LLM, or defer until ``bind``.'''
        self.middleware(self._middleware)

    async def acompact(
        self,
        messages: Sequence[dict],
        tools: Optional[list] = None,
        run_id: str = '',
    ) -> TrimResult:
        '''Fit ``messages`` into the budget by recapping the oldest turns.

        Args:
            messages (Sequence[dict]): Prompt messages in OpenAI format.
            tools (Optional[list]): Tool schemas riding with the prompt.
            run_id (str): Run the prompt belongs to, for persistence.

        Returns:
            TrimResult: The fitted list and what it cost.
        '''
        budget = self.budget - estimate_tools(tools)
        message_list = list(messages)
        costs = [estimate_message(message) for message in message_list]
        before = sum(costs)
        result = TrimResult(messages=message_list, before=before, budget=budget, after=before)
        if not message_list or before <= budget:
            return result

        system, rest = split_system_prefix(message_list) if self.keep_system else ([], message_list)
        system_tokens = sum(costs[:len(system)]) if self.keep_system else 0
        cost_by_id = {id(message): cost for message, cost in zip(message_list, costs)}
        turns = group_turns(rest)
        if not turns:
            return result

        keep_at_least = 1 + self.keep_recent
        kept: List[List[dict]] = []
        used = system_tokens
        for turn in reversed(turns):
            cost = sum(cost_by_id.get(id(message)) or 0 for message in turn)
            forced = len(kept) < keep_at_least
            if not forced and used + cost > budget:
                break
            kept.append(turn)
            used += cost
        kept.reverse()
        dropped_turns = turns[:len(turns) - len(kept)]
        if not dropped_turns:
            return result

        dropped_messages = [message for turn in dropped_turns for message in turn]
        kept_messages = [message for turn in kept for message in turn]
        dropped_tokens = before - used
        header = self.marker.format(count=len(dropped_messages), tokens=dropped_tokens)
        provider = self._provider()
        recap = ''
        if provider is not None:
            recap = await provider.asummarize(dropped_messages)
        if recap:
            note = f'{header}\n\n{recap}'
        else:
            self.stats['fallbacks'] += 1
            note = header
        fitted = [{'role': 'user', 'content': note}] + kept_messages
        after = system_tokens + estimate_message(fitted[0]) + (used - system_tokens)
        if before - after < self.min_savings:
            return result

        result.messages = system + fitted
        result.dropped = len(dropped_messages)
        result.dropped_tokens = dropped_tokens
        result.after = after
        result.changed = True
        self.stats['summarized'] += 1
        self.stats['summarized_messages'] += len(dropped_messages)
        self.stats['saved_tokens'] += max(0, before - after)
        self._maybe_persist(run_id, result)
        return result

    async def atrim(self, kwargs: Dict[str, Any]) -> TrimResult:
        '''Rewrite one in-flight request and record the outcome.

        Args:
            kwargs (Dict[str, Any]): Request kwargs from the middleware chain.

        Returns:
            TrimResult: What was done for this request.
        '''
        messages = kwargs.get('messages') or []
        result = await self.acompact(messages, kwargs.get('tools'), kwargs.get('run_id') or '')
        self.stats['requests'] += 1
        self.stats['last'] = result.as_dict()
        if result.changed:
            kwargs['messages'] = result.messages
        return result

    def compact(self, messages: Sequence[dict], tools: Optional[list] = None) -> TrimResult:
        '''Blocking wrapper around :meth:`acompact`.

        Args:
            messages (Sequence[dict]): Prompt messages in OpenAI format.
            tools (Optional[list]): Tool schemas riding with the prompt.

        Returns:
            TrimResult: The fitted list and what it cost.
        '''
        return asyncio.run(self.acompact(messages, tools))

    def _maybe_persist(self, run_id: Optional[str], result: TrimResult) -> None:
        '''Push the rewritten history back to the agent when ``persist`` is set.'''
        if not self.persist or not run_id:
            return
        if self._persisted.get(run_id) == result.dropped:
            return
        kept = [message for message in result.messages if message.get('role') != 'system']
        queued = self.feedback(
            'control:update_messages',
            run_id=run_id,
            payload={'op': 'replace', 'messages': kept},
        )
        if queued:
            self._persisted[run_id] = result.dropped
        self.log(f'summarized run {run_id}: kept {len(result.messages)} message(s), '
                 f'rewrote {result.dropped}')

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
        data['provider'] = type(self.summarizer.llm).__name__ if (
            self.summarizer is not None and self.summarizer.llm is not None) else None
        return data


class _SummaryMiddleware(Middleware):
    '''Middleware adapter that summarizes the prompt before it is sent.'''

    def __init__(self, plugin: SummarizingContextPlugin):
        '''Wrap the plugin as an LLM middleware.

        Args:
            plugin (SummarizingContextPlugin): Owner of the policy.
        '''
        super().__init__(name='summarizing_context', scope=plugin.scope)
        self.plugin = plugin

    async def __call__(self, kwargs, next_request):
        '''Rewrite ``kwargs['messages']`` and stream the inner response.

        Args:
            kwargs (Dict[str, Any]): Request kwargs.
            next_request (Callable): The rest of the middleware chain.

        Yields:
            Response: Response items from the inner chain.
        '''
        await self.plugin.atrim(kwargs)
        async for response in next_request(kwargs):
            yield response
