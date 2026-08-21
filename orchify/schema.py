from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Vote:
    '''Represents an agent's vote in voting mode.'''
    agent_name: str
    vote: Any
    reason: str = ''