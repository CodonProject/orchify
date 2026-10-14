import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

from orchify import Agent


def main():
    '''Ask one question and print the answer.

    Requires ``API_KEY`` (or ``OPENAI_API_KEY``) and optionally ``BASE_URL``,
    ``MODEL_ID`` in the environment or in a ``.env`` file.
    '''
    agent = Agent.from_env(
        name='assistant',
        system_prompt='Answer in one short sentence.',
    )
    print(agent.ask('Why is the sky blue?', timeout=60))


if __name__ == '__main__':
    main()
