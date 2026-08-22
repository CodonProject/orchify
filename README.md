# Orchify

A Python framework for building LLM-powered AI agent orchestration systems.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from orchify import Agent, OpenAICompat

llm = OpenAICompat()
agent = Agent(name="MyAgent", llm=llm, system_prompt="You are a helpful assistant.")

agent.run(user_input="Hello!")
```

## With Tools

```python
from orchify import Agent, OpenAICompat, tool

@tool(cache=True)
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 25°C"

llm = OpenAICompat()
agent = Agent(
    name="WeatherBot",
    llm=llm,
    system_prompt="You are a weather assistant.",
    tools=[get_weather]
)

agent.run(user_input="What's the weather in Tokyo?")
```

## Environment Variables

Create `.env` file:

```
API_KEY=your_api_key
BASE_URL=https://api.deepseek.com
MODEL_ID=deepseek-chat
```

## Event Monitoring

```python
from orchify.broker import orchify_broker
from orchify.event import BaseEvent

@orchify_broker.hook('*')
def on_event(event: BaseEvent):
    print(f"{event.event_type}: {event.content}")

agent.run(user_input="Hi!")
```
