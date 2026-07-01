from typing import List, Dict, Any, Optional, Union, Generator

from agentforge.llm.base import LLMInterface
from agentforge.tool import Tool
from agentforge.event import AgentEvent, ToolEvent, EVENT_TYPES
from agentforge.utils import safecode


class Agent:
    def __init__(
        self,
        name: str,
        llm: LLMInterface,
        system_prompt: str = 'You are a helpful assistant.',
        tools: Optional[List[Tool]] = None,
        model: str = 'gpt-4o',
    ):
        self.name = name
        self.code = safecode(length=4)
        self.llm = llm
        self.system_prompt = system_prompt
        self.model = model
        
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        
        self.messages: List[Dict[str, Any]] = [
            {'role': 'system', 'content': system_prompt}
        ]

    def run(self, user_input: str, max_steps: int = 5) -> Generator[Union[AgentEvent, ToolEvent], None, None]:
        ...