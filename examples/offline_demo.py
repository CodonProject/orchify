import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

from orchify import Agent, Tool, console, orchify_manager
from orchify.testing import FakeLLM, final, tool_call


def deploy(env: str) -> str:
    '''Pretend to deploy a release.

    Args:
        env (str): Target environment.

    Returns:
        str: What the deployment did.
    '''
    return f'deployed to {env}'


def main():
    '''Run ask, a tool call, an approval and a question without a provider.

    ``orchify.testing.FakeLLM`` replays a scripted provider, so this example
    needs no API key, and the console answers are scripted instead of typed.
    '''
    agent = Agent(
        name='release-bot',
        llm=FakeLLM([
            [final('', [tool_call('deploy', '{"env": "prod"}')])],
            [final('', [tool_call('ask_user', '{"question": "Which tag?"}')])],
            [final('released v1.0.0')],
        ]),
        system_prompt='Use your tools, ask when you are missing a detail.',
        tools=[Tool(
            func=deploy,
            name='deploy',
            description='Deploy the current release.',
            require_approval=True,
        )],
        enable_ask_user=True,
    )

    answers = iter(['y', 'v1.0.0'])
    loop = console(inp=lambda prompt: next(answers)).start()
    try:
        print(agent.ask('Ship the release.', max_steps=5, timeout=60))
    finally:
        loop.stop()

    print(f'run ids: {[item["run_id"] for item in orchify_manager.history(limit=3)]}')


if __name__ == '__main__':
    main()
