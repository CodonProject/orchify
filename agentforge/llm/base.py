from dataclasses import dataclass, field
from typing import Optional, Generator, Literal


@dataclass
class Chunk:
    is_cot: bool
    content: str

    is_assembly_tool: bool = field(default=False)
    assembly_chunk: str    = field(default='')

    total_content: str          = field(default='')
    total_cot_content: str      = field(default='')
    total_tool_call: list[dict] = field(default_factory=list)

    is_cot_end: bool = field(default=False)


@dataclass
class FinalStatus:
    content: str
    reasoning: str
    tool_calls: list[dict] = field(default_factory=list)

    completion_tokens: int = field(default=0)
    prompt_tokens: int = field(default=0)
    prompt_cache_hit_tokens: int = field(default=0)
    prompt_cache_miss_tokens: int = field(default=0)
    total_tokens: int = field(default=0)

    @property
    def completion_token(self) -> int:
        return self.completion_tokens

    @property
    def prompt_token(self) -> int:
        return self.prompt_tokens

    @property
    def total_token(self) -> int:
        return self.total_tokens


@dataclass
class Response:
    current_chunk: Optional[Chunk]
    final_status: Optional[FinalStatus] = field(default=None)

    is_final: bool = field(default=False)


class LLMInterface:
    def request(
        self,
        messages: list[dict],
        model: str = 'gpt-4o',
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        json_format: bool = False,
        max_tokens: Optional[int] = None,
        thinking: Literal['disabled', 'enabled', 'auto'] = 'auto',
        effort: Literal['minimal', 'low', 'medium', 'high', 'xhigh'] = 'medium',
    ) -> Generator[Response, None, None]:
        raise NotImplementedError('Subclasses must implement this method.')