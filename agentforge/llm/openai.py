import json
import requests

from typing import Generator, Optional, Literal

from agentforge.llm.base import Chunk, FinalStatus, Response, LLMInterface


class OpenAICompat(LLMInterface):
    def __init__(self, api_key: str, base_url: str = 'https://api.openai.com/v1'):
        self.api_key  = api_key
        self.base_url = base_url
        self.url = '/chat/completions'
    
    @property
    def full_url(self) -> str:
        return f'{self.base_url}{self.url}'
    
    def _request(
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
    ) -> Generator[Chunk, None, FinalStatus]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Build requests payload safely
        data = {
            'model': model,
            'messages': messages,
            'stream': True,
            'stream_options': {'include_usage': True},
        }

        if tools: data['tools'] = tools

        # Handle temperature and top_p: OpenAI reasoning models do not support them
        is_openai_reasoning = model.startswith('o1') or model.startswith('o3')
        if not is_openai_reasoning:
            data['temperature'] = temperature
            data['top_p'] = top_p

        if json_format:
            data['response_format'] = {'type': 'json_object'}

        if max_tokens is not None:
            data['max_tokens'] = max_tokens

        # Handle thinking / reasoning parameters based on provider and model
        is_deepseek = 'api.deepseek.com' in self.base_url or 'deepseek' in model.lower()

        if is_openai_reasoning:
            if thinking != 'disabled':
                effort_map = {
                    'minimal': 'low',
                    'low': 'low',
                    'medium': 'medium',
                    'high': 'high',
                    'xhigh': 'high'
                }
                data['reasoning_effort'] = effort_map.get(effort, 'medium')
        elif is_deepseek:
            # Map 'auto' to 'adaptive' for DeepSeek models
            ds_thinking_type = 'adaptive' if thinking == 'auto' else thinking
            data['thinking'] = {'type': ds_thinking_type}
        else:
            # Fallback for other providers: only set if explicitly enabled/disabled
            if thinking in ('enabled', 'disabled'):
                data['thinking'] = {'type': thinking}

        try:
            response = requests.post(self.full_url, headers=headers, json=data, stream=True)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json()
            except Exception:
                try:
                    error_detail = response.text
                except Exception:
                    error_detail = ""
            
            error_msg = f"HTTP Error {response.status_code}: {e}"
            if error_detail:
                error_msg += f"Response detail: {error_detail}"
            try:
                response.close()
            except Exception:
                pass
            raise requests.exceptions.HTTPError(error_msg, response=response) from e

        total_content = ""
        total_cot_content = ""
        tool_calls_map = {}

        completion_tokens = 0
        prompt_tokens = 0
        prompt_cache_hit_tokens = 0
        prompt_cache_miss_tokens = 0
        total_tokens = 0

        has_cot = False
        cot_ended = False

        with response:
            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode('utf-8')
                if not line_str.startswith('data: '):
                    continue
                
                raw_data = line_str[6:].strip()
                if raw_data == '[DONE]':
                    break

                try:
                    chunk = json.loads(raw_data)
                except json.JSONDecodeError:
                    continue

                usage = chunk.get('usage')
                if usage:
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    
                    prompt_details = usage.get('prompt_tokens_details', {})
                    cached_tokens = prompt_details.get('cached_tokens')
                    if cached_tokens is not None:
                        prompt_cache_hit_tokens = cached_tokens
                    else:
                        prompt_cache_hit_tokens = usage.get('prompt_cache_hit_tokens', 0)
                    
                    prompt_cache_miss_tokens = usage.get('prompt_cache_miss_tokens')
                    if prompt_cache_miss_tokens is None:
                        prompt_cache_miss_tokens = max(0, prompt_tokens - prompt_cache_hit_tokens)

                choices = chunk.get('choices', [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get('delta', {})

                cot_chunk = delta.get('reasoning_content') or delta.get('reasoning') or ""
                content_chunk = delta.get('content') or ""

                tool_calls_delta = delta.get('tool_calls', [])
                is_assembly_tool = False
                assembly_chunk = ""

                if tool_calls_delta:
                    is_assembly_tool = True
                    for tool_call in tool_calls_delta:
                        idx = tool_call.get('index')
                        if idx is None:
                            continue

                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                'id': '',
                                'type': 'function',
                                'function': {'name': '', 'arguments': ''}
                            }

                        if tool_call.get('id'):
                            tool_calls_map[idx]['id'] = tool_call['id']
                        if tool_call.get('type'):
                            tool_calls_map[idx]['type'] = tool_call['type']

                        func_delta = tool_call.get('function', {})
                        if func_delta.get('name'):
                            tool_calls_map[idx]['function']['name'] += func_delta['name']
                        if func_delta.get('arguments'):
                            arg_part = func_delta['arguments']
                            tool_calls_map[idx]['function']['arguments'] += arg_part
                            assembly_chunk += arg_part

                if cot_chunk:
                    total_cot_content += cot_chunk
                if content_chunk:
                    total_content += content_chunk

                is_cot_end = False
                if cot_chunk:
                    has_cot = True
                elif has_cot and not cot_ended:
                    is_cot_end = True
                    cot_ended = True

                yield_triggered = False

                if cot_chunk:
                    yield Chunk(
                        is_cot=True,
                        content=cot_chunk,
                        total_content=total_content,
                        total_cot_content=total_cot_content,
                        total_tool_call=list(tool_calls_map.values())
                    )
                    yield_triggered = True

                if content_chunk:
                    yield Chunk(
                        is_cot=False,
                        content=content_chunk,
                        total_content=total_content,
                        total_cot_content=total_cot_content,
                        total_tool_call=list(tool_calls_map.values()),
                        is_cot_end=is_cot_end
                    )
                    yield_triggered = True

                if is_assembly_tool and not (cot_chunk or content_chunk):
                    yield Chunk(
                        is_cot=False,
                        content="",
                        is_assembly_tool=True,
                        assembly_chunk=assembly_chunk,
                        total_content=total_content,
                        total_cot_content=total_cot_content,
                        total_tool_call=list(tool_calls_map.values()),
                        is_cot_end=is_cot_end
                    )
                    yield_triggered = True

                if is_cot_end and not yield_triggered:
                    yield Chunk(
                        is_cot=False,
                        content="",
                        total_content=total_content,
                        total_cot_content=total_cot_content,
                        total_tool_call=list(tool_calls_map.values()),
                        is_cot_end=True
                    )

            if has_cot and not cot_ended:
                yield Chunk(
                    is_cot=False,
                    content="",
                    total_content=total_content,
                    total_cot_content=total_cot_content,
                    total_tool_call=list(tool_calls_map.values()),
                    is_cot_end=True
                )

        return FinalStatus(
            content=total_content,
            reasoning=total_cot_content,
            tool_calls=list(tool_calls_map.values()),
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            prompt_cache_hit_tokens=prompt_cache_hit_tokens,
            prompt_cache_miss_tokens=prompt_cache_miss_tokens,
            total_tokens=total_tokens
        )
    
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
        gen = self._request(
            messages=messages,
            model=model,
            tools=tools,
            temperature=temperature,
            top_p=top_p,
            json_format=json_format,
            max_tokens=max_tokens,
            thinking=thinking,
            effort=effort,
        )
        
        while True:
            try:
                chunk = next(gen)
                yield Response(current_chunk=chunk, is_final=False)
            except StopIteration as e:
                final_status = e.value
                yield Response(current_chunk=None, final_status=final_status, is_final=True)
                break
