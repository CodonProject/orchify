import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

from orchify import Agent, Tool


def word_count(text: str) -> str:
    '''Count the words in a text.

    Args:
        text (str): Text to count the words of.

    Returns:
        str: The number of words, as text.
    '''
    return f'{len(str(text).split())} words'


def main():
    '''Give an agent a tool and let it decide when to call it.

    The tool schema is derived from the signature and the docstring, so no
    hand-written JSON is needed.
    '''
    agent = Agent.from_env(
        name='assistant',
        system_prompt='Use your tools when they help, then answer briefly.',
        tools=[Tool(
            func=word_count,
            name='word_count',
            description='Count the words in a text.',
        )],
    )
    print(agent.ask('How many words are in "one two three four"?', max_steps=5, timeout=60))


if __name__ == '__main__':
    main()
