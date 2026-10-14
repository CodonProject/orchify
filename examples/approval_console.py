import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

from orchify import Agent, Tool, console


def run_shell(command: str) -> str:
    '''Pretend to run a shell command on the host.

    Args:
        command (str): Command the agent wants to run.

    Returns:
        str: What the command would have printed.
    '''
    return f'(pretend) ran: {command}'


def main():
    '''Let the operator approve each risky tool call in the terminal.

    ``console()`` runs in its own thread, so the main thread can stay parked on
    ``ask()`` while the approval prompts are printed here.
    '''
    agent = Agent.from_env(
        name='operator',
        system_prompt='Use the shell tool when the user asks for something risky.',
        tools=[Tool(
            func=run_shell,
            name='run_shell',
            description='Run a shell command on the host.',
            require_approval=True,
        )],
    )
    loop = console().start()
    try:
        print(agent.ask('List the files in the current directory.', max_steps=6, timeout=300))
    finally:
        loop.stop()


if __name__ == '__main__':
    main()
