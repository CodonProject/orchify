from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_PRICE_TABLE: Dict[str, Dict[str, float]] = {
    'gpt-4o-mini': {'prompt': 0.15, 'completion': 0.60, 'cache_hit': 0.075},
    'gpt-4o': {'prompt': 2.50, 'completion': 10.00, 'cache_hit': 1.25},
    'deepseek-chat': {'prompt': 0.27, 'completion': 1.10, 'cache_hit': 0.07},
    'deepseek-reasoner': {'prompt': 0.55, 'completion': 2.19, 'cache_hit': 0.14},
}


@dataclass
class Price:
    '''USD per 1M tokens for one model.'''

    prompt: float = 0.0
    completion: float = 0.0
    cache_hit: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        '''Return this price entry as a plain dictionary.

        Returns:
            Dict[str, float]: Keys ``prompt``, ``completion`` and ``cache_hit`` in
            USD per million tokens.
        '''
        return {'prompt': self.prompt, 'completion': self.completion, 'cache_hit': self.cache_hit}

def _num(value: Any) -> float:
    '''A finite float for a numeric value, else 0.0 (bools are not numbers).'''
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _parse_price_table(raw: Optional[str]) -> Dict[str, Dict[str, float]]:
    '''Parse ORCHIFY_PRICE_TABLE (inline JSON or a JSON file path); {} on failure.'''
    if not raw or not raw.strip():
        return {}
    text = raw.strip()
    if os.path.exists(text):
        try:
            with open(text, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
    else:
        try:
            data = json.loads(text)
        except ValueError:
            return {}
    if not isinstance(data, dict):
        return {}
    table: Dict[str, Dict[str, float]] = {}
    for model, values in data.items():
        if not isinstance(values, dict):
            continue
        entry: Dict[str, float] = {}
        for key in ('prompt', 'completion', 'cache_hit'):
            if isinstance(values.get(key), (int, float)) and not isinstance(values.get(key), bool):
                entry[key] = float(values[key])
        if entry:
            table[str(model)] = entry
    return table


def load_price_table() -> Dict[str, Dict[str, float]]:
    '''Built-in prices merged with ORCHIFY_PRICE_TABLE (the env wins per model).'''
    table = {model: dict(entry) for model, entry in DEFAULT_PRICE_TABLE.items()}
    for model, entry in _parse_price_table(os.getenv('ORCHIFY_PRICE_TABLE')).items():
        merged = dict(table.get(model) or {})
        merged.update(entry)
        table[model] = merged
    return table


def price_for(model: Optional[str], table: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[Price]:
    '''Longest-prefix price lookup for a model name; None when unknown.'''
    if not model:
        return None
    table = load_price_table() if table is None else table
    needle = str(model).lower()
    best_key: Optional[str] = None
    for key in table:
        lowered = str(key).lower()
        if needle == lowered or needle.startswith(lowered):
            if best_key is None or len(str(key)) > len(str(best_key)):
                best_key = str(key)
    if best_key is None:
        return None
    entry = table.get(best_key, {})
    cache_hit = entry.get('cache_hit', entry.get('prompt', 0.0))
    return Price(prompt=entry.get('prompt', 0.0), completion=entry.get('completion', 0.0), cache_hit=cache_hit)


def cost_of(usage: Optional[Dict[str, Any]], price: Optional[Price]) -> float:
    '''Estimate the USD cost of one usage payload.

    Args:
        usage (Optional[Dict[str, Any]]): Provider usage counters as reported in
            ``prompt_tokens``, ``completion_tokens`` and
            ``prompt_cache_hit_tokens``.
        price (Optional[Price]): Price entry for the model, or ``None`` when the
            model cannot be priced.

    Returns:
        float: The estimated cost in USD, or ``0.0`` when it cannot be priced.
        ``prompt_tokens`` is billed at the cache-miss rate for everything not
        reported as a cache hit, which is how providers report cached prompts
        (``prompt_tokens`` already includes the cached part).
    '''
    if not usage or price is None:
        return 0.0
    prompt = _num(usage.get('prompt_tokens'))
    completion = _num(usage.get('completion_tokens'))
    hit = min(_num(usage.get('prompt_cache_hit_tokens')), prompt)
    miss = max(0.0, prompt - hit)
    return (miss * price.prompt + hit * price.cache_hit + completion * price.completion) / 1_000_000.0


@dataclass
class Budget:
    '''Independent caps for one run. None/0 means "no cap".'''

    tokens: Optional[int] = None
    usd: Optional[float] = None
    seconds: Optional[float] = None
    price_table: Optional[Dict[str, Dict[str, float]]] = None

    def enabled(self) -> bool:
        '''True when at least one cap is set.'''
        return bool(self.tokens) or bool(self.usd) or bool(self.seconds)

    def limits(self) -> Dict[str, Any]:
        '''The configured caps (unset ones are None).'''
        return {
            'tokens': self.tokens or None,
            'usd': self.usd or None,
            'seconds': self.seconds or None,
        }

    def price(self, model: Optional[str]) -> Optional[Price]:
        '''Price for a model under this budget's table.'''
        return price_for(model, self.price_table)

    def cost(self, usage: Optional[Dict[str, Any]], model: Optional[str]) -> float:
        '''Estimated USD for a usage payload.'''
        return cost_of(usage, self.price(model))

    def require_price(self, model: Optional[str]) -> None:
        '''Raise when a USD cap is set but the model has no price.'''
        if not self.usd:
            return
        if self.price(model) is None:
            raise ValueError(
                f"budget_usd is set but model '{model}' has no price entry. "
                "Add one with ORCHIFY_PRICE_TABLE (inline JSON or a JSON file path), "
                f'e.g. {{"{model}": {{"prompt": 1.0, "completion": 2.0, "cache_hit": 0.25}}}}.'
            )

    def violations(
        self,
        usage: Optional[Dict[str, Any]],
        elapsed: float,
        model: Optional[str],
    ) -> List[Dict[str, Any]]:
        '''Cap violations for the current usage/elapsed time, tokens first.'''
        usage = usage or {}
        found: List[Dict[str, Any]] = []
        total = usage.get('total_tokens')
        if self.tokens and _num(total) > self.tokens:
            found.append({'reason': 'tokens', 'limit': self.tokens, 'used': _num(total)})
        if self.usd:
            price = self.price(model)
            if price is not None:
                cost = cost_of(usage, price)
                if cost > self.usd:
                    found.append({'reason': 'usd', 'limit': self.usd, 'used': round(cost, 6)})
        if self.seconds and elapsed > self.seconds:
            found.append({'reason': 'seconds', 'limit': self.seconds, 'used': round(elapsed, 3)})
        return found

    def info(
        self,
        usage: Optional[Dict[str, Any]],
        elapsed: float,
        model: Optional[str],
    ) -> Dict[str, Any]:
        '''Serialisable budget state for event payloads and manager records.'''
        usage = usage or {}
        return {
            'limits': self.limits(),
            'elapsed': round(elapsed, 3),
            'cost_usd': round(self.cost(usage, model), 6),
            'priced': self.price(model) is not None,
            'total_tokens': _num(usage.get('total_tokens')),
        }
