'''
Cross-Entropy:

During an agents lifetime, it's experience is stored as separate Episodes, each with its own set of EpisodeSteps.
Each Episode is a sequence of observations the agent receives from the environment, actions it has issued and rewards it
has received for these actions. For every Episode we can calculate the total reward the agent has claimed.

Episodes :
================================
|| o1, a1, r1 || o2, a2, r2 || o3, a3, r3 || o4, a4, r4 || o5, a5, r5 || R: r1+r2+r3+...+r5

|| o1, a1, r1 || o2, a2, r2 || o3, a3, r3 || R: r1+r2+r3

|| o1, a1, r1 || o2, a2, r2 || o3, a3, r3 || o4, a4, r4 || o5, a5, r5 || o6, a6, r6 || R: r1+r2+r3+...+r6

|| o1, a1, r1 || o2, a2, r2 || o3, a3, r3 || o4, a4, r4 || R: r1+r2+r3+r4

=================================
Every cell represents an agent's step in the episode. Due to randomness in the environment and the way the agents selection
of actions some episodes will be better than others.
The Core of Cross-Entropy is to train on these good ones and throw away the bad ones.
Steps of the algorithm:

1. Play N-number of episodes
2. calculate the total reward of every episode and decide on a reward boundary. Usually the 50 - 70th percentile.
3. throw away all episodes with a reward below the boundary
4. train on the remaining 'elite' episodes using observations as the input and issued actions as the desired output
5. repeat from step 1 until we become satisfied with the result


'''

import torch.nn as nn

HIDDEN_SIZE = 128
BATCH_SIZE = 32
PERCENTILE = 70

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    