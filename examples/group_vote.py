import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

from orchify import Agent, Group


def main():
    '''Ask a panel of agents and print the majority answer.

    A ``'voting'`` group waits for every member (``ORCHIFY_VOTE_TIMEOUT``,
    300 seconds by default) and stores the winning answer on the group.
    '''
    members = [
        Agent.from_env(
            name=f'analyst-{index}',
            system_prompt='Answer with Yes or No first, then one short reason.',
        )
        for index in (1, 2, 3)
    ]
    group = Group(name='panel', agents=members, mode='voting')

    print(group.ask('Is a hotdog a sandwich?', timeout=300))
    print(f'winner: {group.last_winner} with {len(group.last_votes)} votes')


if __name__ == '__main__':
    main()
