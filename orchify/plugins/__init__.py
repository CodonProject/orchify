'''Opt-in plugins shipped with orchify.

Nothing here is loaded automatically: import the class you want and register it
with the plugin manager (or attach it to an agent), because every one of them
changes request behaviour or adds bookkeeping::

    from orchify.plugins import ContextWindowPlugin, CostPlugin
    from orchify import orchify_plugins

    orchify_plugins.register(ContextWindowPlugin(max_prompt_tokens=16000))
    orchify_plugins.register(CostPlugin(max_run_cost=0.50))

    agent = Agent(name='assistant', plugins=['context_window', 'cost'])

Attributes:
    ContextWindowPlugin (type): Keeps prompts inside a token budget.
    CostPlugin (type): Prices tokens and exposes spend per run.
'''

from orchify.plugins.context import ContextWindowPlugin, TrimResult
from orchify.plugins.cost import CostPlugin

__all__ = ['ContextWindowPlugin', 'CostPlugin', 'TrimResult']
