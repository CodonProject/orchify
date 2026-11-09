'''Cost plugin: price every LLM step and expose spend per run.

The provider reports exact token counts on every ``agent:finish`` event, so the
plugin prices those counters with :mod:`orchify.budget` (longest-prefix model
match against the built-in table plus ``ORCHIFY_PRICE_TABLE``) and keeps a
running total per run and for the process.

Besides the numbers it reports, the plugin can enforce a cap: with
``max_run_cost`` it queues ``control:stop`` for the run as soon as the spend
crosses the limit, and with ``max_total_cost`` it stops new runs from starting
by emitting ``cost:budget`` with ``scope='global'``.

Example:
    from orchify.plugins import CostPlugin
    from orchify import Agent, orchify_plugins

    orchify_plugins.register(CostPlugin(max_run_cost=0.50))
    agent = Agent(name='assistant', plugins=['cost'])
    agent.ask('hi')
    print(orchify_plugins.get('cost').total_cost())
'''

import time
from collections import deque
from typing import Any, Dict, Optional

from orchify.budget import cost_of, load_price_table, price_for
from orchify.plugin import Plugin

TOKEN_KEYS = ('prompt_tokens', 'completion_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens', 'total_tokens')


class CostPlugin(Plugin):
    '''Token accounting and spend caps.

    Attributes:
        model (str): Model used when a run does not report one.
        price_table (Optional[Dict[str, Dict[str, float]]]): Explicit price table;
            defaults to the built-in table merged with ``ORCHIFY_PRICE_TABLE``.
        max_run_cost (Optional[float]): USD cap for one run; None disables the cap.
        max_total_cost (Optional[float]): USD cap for everything this plugin has
            seen; None disables the cap.
        run_costs (Dict[str, Dict[str, Any]]): Per-run totals, keyed by run id.
        total_usd (float): Sum of every priced step the plugin has seen.
    '''

    name = 'cost'
    version = '1.0.0'
    description = 'Price LLM usage per step and per run, with optional USD caps.'
    tags = ('cost', 'budget', 'observability')

    def __init__(
        self,
        broker=None,
        *,
        model: str = '',
        price_table: Optional[Dict[str, Dict[str, float]]] = None,
        max_run_cost: Optional[float] = None,
        max_total_cost: Optional[float] = None,
        emit_events: bool = True,
        history: int = 200,
    ):
        '''Create the plugin.

        Args:
            broker (Optional[Broker]): Broker to attach to; defaults to the shared
                one.
            model (str): Fallback model name used when a run carries none.
            price_table (Optional[Dict[str, Dict[str, float]]]): Override price
                table; None uses :func:`orchify.budget.load_price_table`.
            max_run_cost (Optional[float]): Stop a run that costs more than this.
            max_total_cost (Optional[float]): Stop new runs after this total.
            emit_events (bool): Emit ``cost:step`` / ``cost:run`` / ``cost:budget``
                events for dashboards and tests.
            history (int): Number of step records kept in ``records``.
        '''
        super().__init__(broker=broker)
        self.model = model
        self.price_table = price_table
        self.max_run_cost = float(max_run_cost) if max_run_cost is not None else None
        self.max_total_cost = float(max_total_cost) if max_total_cost is not None else None
        self.emit_events = bool(emit_events)
        self.records: deque = deque(maxlen=max(1, int(history)))
        self.run_costs: Dict[str, Dict[str, Any]] = {}
        self.total_usd = 0.0
        self.total_tokens: Dict[str, int] = {key: 0 for key in TOKEN_KEYS}
        self._models: Dict[str, str] = {}
        self._warned_models: set = set()
        self._stopped: set = set()
        self._table: Optional[Dict[str, Any]] = None
        self._prices: Dict[str, Any] = {}

    def bind(self, agent) -> None:
        '''Remember the agent's model as the fallback for its runs.

        Args:
            agent (Agent): Agent this plugin was attached to.
        '''
        super().bind(agent)
        model = getattr(agent, 'model', '') or ''
        if model and not self.model:
            self.model = model

    @Plugin.hook('agent:finish')
    def on_agent_finish(self, event) -> None:
        '''Price one finished LLM step.

        Args:
            event (AgentEvent): The ``agent:finish`` event of one step.
        '''
        payload = getattr(event, 'payload', None) or {}
        run_id = getattr(event, 'run_id', '') or ''
        usage = {key: _num(payload.get(key)) for key in TOKEN_KEYS}
        if not usage['prompt_tokens'] and not usage['completion_tokens']:
            return
        model = self._model_for(run_id)
        price = self._price_for(model)
        usd = cost_of(usage, price)
        record = {
            'run_id': run_id,
            'agent': getattr(event, 'agent_name', '') or '',
            'step': getattr(event, 'step', None) or getattr(event, 'turn', None) or 0,
            'model': model,
            'prompt_tokens': usage['prompt_tokens'],
            'completion_tokens': usage['completion_tokens'],
            'cached_tokens': usage['prompt_cache_hit_tokens'],
            'usd': usd,
            'priced': price is not None,
            'finish_reason': payload.get('finish_reason') or '',
            'ts': time.time(),
        }
        self.records.append(record)
        self._accumulate(run_id, record)
        self.total_usd += usd
        for key, value in usage.items():
            self.total_tokens[key] += value
        if self.emit_events:
            self.event('cost:step', payload=record)
        self._check_caps(run_id, record)

    @Plugin.hook('run:finish')
    def on_run_finish(self, event) -> None:
        '''Finalise the totals for a run and emit ``cost:run``.

        Args:
            event (BaseEvent): The ``run:finish`` event of the run.
        '''
        run_id = getattr(event, 'run_id', '') or ''
        totals = self.run_costs.get(run_id)
        if totals is None:
            return
        payload = getattr(event, 'payload', None) or {}
        status = payload.get('status') or 'finished'
        summary = dict(totals)
        summary.update({'run_id': run_id, 'status': status, 'ts': time.time()})
        self.run_costs[run_id] = summary
        if self.emit_events:
            self.event('cost:run', payload=summary)
        self.log(f'run {run_id} cost ${summary.get("usd", 0.0):.6f} '
                 f'({summary.get("prompt_tokens", 0)} prompt / {summary.get("completion_tokens", 0)} completion)')

    def run_cost(self, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        '''Return the spend of one run, or of every run.

        Args:
            run_id (Optional[str]): Run to inspect; None returns all runs.

        Returns:
            Optional[Dict[str, Any]]: The run's totals, a copy of every run keyed
            by run id, or None when the run is unknown.
        '''
        if run_id is None:
            return {key: dict(value) for key, value in self.run_costs.items()}
        totals = self.run_costs.get(run_id)
        return dict(totals) if totals is not None else None

    def total_cost(self) -> float:
        '''Return the total priced spend in USD.

        Returns:
            float: Summed cost of every step seen since the plugin loaded.
        '''
        return round(self.total_usd, 10)

    def cost_of_run(self, run_id: str) -> float:
        '''Return one run's cost in USD, 0.0 when unknown.

        Args:
            run_id (str): Run to price.

        Returns:
            float: The run's accumulated cost.
        '''
        totals = self.run_costs.get(run_id) or {}
        return float(totals.get('usd') or 0.0)

    def snapshot(self) -> Dict[str, Any]:
        '''Return counters, totals and per-run costs as a plain dict.

        Returns:
            Dict[str, Any]: Snapshot suitable for status endpoints or logs.
        '''
        return {
            'steps': len(self.records),
            'total_usd': self.total_cost(),
            'total_tokens': dict(self.total_tokens),
            'runs': len(self.run_costs),
            'max_run_cost': self.max_run_cost,
            'max_total_cost': self.max_total_cost,
            'model': self.model,
            'last': dict(self.records[-1]) if self.records else None,
        }

    def reset(self) -> None:
        '''Forget every record and total (useful between tests).'''
        self.records.clear()
        self.run_costs.clear()
        self._models.clear()
        self._stopped.clear()
        self._warned_models.clear()
        self._prices.clear()
        self.total_usd = 0.0
        self.total_tokens = {key: 0 for key in TOKEN_KEYS}

    def _model_for(self, run_id: str) -> str:
        '''Resolve the model id for a run.'''
        if run_id in self._models:
            return self._models[run_id]
        model = ''
        context = (getattr(self.broker, 'run_contexts', {}) or {}).get(run_id) or {}
        if isinstance(context, dict):
            model = str(context.get('model') or '')
        model = model or self.model
        if run_id:
            self._models[run_id] = model
        return model

    def _price_for(self, model: str):
        '''Look up a price, warning once per unpriced model.

        The merged price table is resolved once (``load_price_table()`` returns a
        fresh copy, which used to be rebuilt for every step of every run) and each
        model's price is memoised, so a long run prices its steps by dict lookup.
        '''
        try:
            return self._prices[model]
        except KeyError:
            pass
        if self._table is None or (
            self.price_table is not None and self._table is not self.price_table
        ):
            self._table = self.price_table if self.price_table is not None else load_price_table()
            self._prices.clear()
        price = price_for(model, self._table)
        self._prices[model] = price
        if price is None and model not in self._warned_models:
            self._warned_models.add(model)
            self.log(f'no price for model {model!r}; set ORCHIFY_PRICE_TABLE to price it')
        return price


    def _accumulate(self, run_id: str, record: Dict[str, Any]) -> None:
        '''Add one step to the per-run totals.'''
        totals = self.run_costs.setdefault(run_id, {
            'run_id': run_id,
            'model': record['model'],
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'cached_tokens': 0,
            'usd': 0.0,
            'steps': 0,
            'unpriced_steps': 0,
            'started_at': record['ts'],
        })
        totals['prompt_tokens'] += record['prompt_tokens']
        totals['completion_tokens'] += record['completion_tokens']
        totals['cached_tokens'] += record['cached_tokens']
        totals['usd'] = round(totals['usd'] + record['usd'], 10)
        totals['steps'] += 1
        if not record['priced']:
            totals['unpriced_steps'] += 1
        totals['updated_at'] = record['ts']

    def _check_caps(self, run_id: str, record: Dict[str, Any]) -> None:
        '''Stop the run (or announce a global cap) when a limit is crossed.'''
        if self.max_run_cost is not None and run_id and run_id not in self._stopped:
            totals = self.run_costs.get(run_id) or {}
            spend = float(totals.get('usd') or 0.0)
            if spend > self.max_run_cost:
                self._stopped.add(run_id)
                reason = (f'run cost ${spend:.6f} exceeded max_run_cost ${self.max_run_cost:.6f}')
                self.feedback('control:stop', run_id=run_id, payload={'reason': reason})
                if self.emit_events:
                    self.event('cost:budget', payload={'run_id': run_id, 'scope': 'run',
                                                      'spend': spend, 'limit': self.max_run_cost,
                                                      'reason': reason})
        if self.max_total_cost is not None and self.total_usd > self.max_total_cost:
            if 'global' not in self._stopped:
                self._stopped.add('global')
                reason = (f'total cost ${self.total_usd:.6f} exceeded max_total_cost '
                          f'${self.max_total_cost:.6f}')
                if self.emit_events:
                    self.event('cost:budget', payload={'run_id': run_id, 'scope': 'global',
                                                      'spend': self.total_usd,
                                                      'limit': self.max_total_cost,
                                                      'reason': reason})


def _num(value: Any) -> int:
    '''Coerce a usage counter to a non-negative int.'''
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, number)
