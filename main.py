import sys

from orchify import Agent, OpenAICompat
from orchify.broker import orchify_broker
from orchify.event import BaseEvent

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

endpoint = OpenAICompat()

agent = Agent(
    name='Agent',
    llm=endpoint,
    system_prompt='',
    thinking='disable'
)


@orchify_broker.hook('*')
def handle_agent_events(event: BaseEvent):
    '''Print every lifecycle event of the demo run.

    Args:
        event (BaseEvent): The event emitted by the broker.
    '''
    if event.event_type == 'run:start':
        print(f"\n[Run Start] Agent: {event.agent_name} (#{event.agent_code})")
        print(f"  run_id : {event.run_id}   (one execution; feedback targets this)")
        print(f"  turn_id: {event.turn_id}   (one LLM round trip)")
        print(f"User Input: {event.payload.get('user_input')}\n")

    elif event.event_type == 'run:next':
        print(f"\n[Next Turn] Entering Step {event.turn} ({event.turn_id})...")

    elif event.event_type == 'agent:reason:step':
        print(f"\033[90m{event.content}\033[0m", end="", flush=True)

    elif event.event_type == 'agent:reason:finish':
        print("\n[Thinking Finished]")

    elif event.event_type == 'agent:answer':
        print(event.content, end="", flush=True)

    elif event.event_type == 'agent:finish':
        print(f"\n\n[Agent Finished Response] Token Usage: {event.payload}")

    elif event.event_type == 'tool:assembly:start':
        print(f"\n[Tool Assembly] Structuring arguments for '{event.tool_name}' ({event.tool_id})...")

    elif event.event_type == 'tool:assembly:step':
        print(f"\033[93m{event.chunk_arg}\033[0m", end="", flush=True)

    elif event.event_type == 'tool:assembly:finish':
        print(f"\n[Tool Assembly Finished] Arguments: {event.args}")

    elif event.event_type == 'tool:call:start':
        print(f"[Tool Execution] Calling '{event.tool_name}' with args: {event.args}")

    elif event.event_type == 'tool:call:finish':
        print(f"[Tool Result] -> {event.payload.get('result')}")

    elif event.event_type == 'run:finish':
        print(f"\n[Run Finished] run_id: {event.run_id}")
        print(f"Total Steps: {event.payload.get('total_steps')}")
        print(f"Total Tokens: {event.payload.get('usage', {}).get('total_tokens')}")
        print(f"Answer: {event.payload.get('answer')!r}")


run_id = agent.run(user_input='你好')

if not agent.wait(run_id):
    print(f'\n[Main] run {run_id} did not finish before ORCHIFY_RUN_TIMEOUT')

print(f'\n[Main] final answer: {agent.last_answer!r}')
