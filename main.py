from orchify import Agent, OpenAICompat
from orchify.broker import orchify_broker
from orchify.event import BaseEvent

import time

endpoint = OpenAICompat()

agent = Agent(
    name='Agent',
    llm=endpoint,
    system_prompt='',
    thinking='disable'
)

@orchify_broker.hook('*')
def handle_agent_events(event: BaseEvent):
    # 1. 运行生命周期事件
    if event.event_type == 'run:start':
        print(f"\n[Run Start] Agent: {event.agent_name} (#{event.agent_code}), Turn: {event.turn_id}")
        print(f"User Input: {event.payload.get('user_input')}\n")
    
    elif event.event_type == 'run:next':
        print(f"\n[Next Turn] Entering Step {event.turn}...")

    # 2. LLM 推理（CoT思维链）事件 - 如果是支持思维链的模型（如 DeepSeek R1/o1）会触发
    elif event.event_type == 'agent:reason:step':
        # 实时打印思维链内容（不换行）
        print(f"\033[90m{event.content}\033[0m", end="", flush=True)
        
    elif event.event_type == 'agent:reason:finish':
        print("\n[Thinking Finished]")

    # 3. LLM 最终回答输出事件
    elif event.event_type == 'agent:answer':
        # 实时流式输出回答
        print(event.content, end="", flush=True)

    elif event.event_type == 'agent:finish':
        print(f"\n\n[Agent Finished Response] Token Usage: {event.payload}")

    # 4. 工具参数流式组装事件
    elif event.event_type == 'tool:assembly:start':
        print(f"\n[Tool Assembly] Structuring arguments for '{event.tool_name}' ({event.tool_id})...")
        
    elif event.event_type == 'tool:assembly:step':
        # 流式打印正在生成的参数 Json 字符串
        print(f"\033[93m{event.chunk_arg}\033[0m", end="", flush=True)
        
    elif event.event_type == 'tool:assembly:finish':
        print(f"\n[Tool Assembly Finished] Arguments: {event.args}")

    # 5. 实际工具执行事件
    elif event.event_type == 'tool:call:start':
        print(f"[Tool Execution] Calling '{event.tool_name}' with args: {event.args}")
        
    elif event.event_type == 'tool:call:finish':
        print(f"[Tool Result] -> {event.payload.get('result')}")

    # 6. 运行结束
    elif event.event_type == 'run:finish':
        print(f"\n[Run Finished] Total Steps: {event.payload.get('total_steps')}")
        print(f"Total Tokens: {event.payload.get('usage', {}).get('total_tokens')}")


agent.run(user_input='你好')

while orchify_broker.runs:
    time.sleep(1)