'''
continuation of laptop...

In practice a policy is a probability distribution over the set of possible actions.
That makes the agents data flow quite simple -> pass the received observation from the environment
to the network and receive a prob. dist. over the set of possible actions as outputs. Then perform
random sampling using prob. dist. to get an action to carry out.

Looking at observation sequences, we end up with different cells, where each cell represents an agent's
step in the episode. (episode ==> a finite number steps the agent has taken)
o = observation received
r = reward received
a = action taken

1 -> [o1, r1, a1], [o2, r2, a2], ....., [oN, rN, aN]  # not uniform in length
2 -> [o1, r1, a1], [o2, r2, a2], ....., [oN, rN, aN]
3 -> [o1, r1, a1], [o2, r2, a2], ....., [oN, rN, aN]
4 -> [o1, r1, a1], [o2, r2, a2], ....., [oN, rN, aN]
....


1 - play a number of episodes using model and environment.
2 - calculate the total reward for the episode. Decide on a reward boundary below which
episodes are thrown away (50th, 70th percentile).
3 - throw episodes below boundary away.
4 - train remaining 'elite' episodes using observations as the input and issued actions as the
desired output.
5 - repeat until satisfied with the result.

'''

from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf
import torch
from tensorboardX import SummaryWriter
from torch.nn import Sequential, Linear, ReLU, Module, Softmax, CrossEntropyLoss

from NNFromScratch.RL_studies.DiscreteWrapper import DiscreteOneHotWrapper

logdir = "logs"

tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

# suppress scientific notation
np.set_printoptions(suppress=True)

'''
constant values
'''
HIDDEN_SIZE = 128
BATCH_SIZE = 16
# episode rewards below this percentile are thrown away
PERCENTILE = 70

'''
Note: cross-entropy succeeds for the CartPole environment, but fails miserably for the FrozenLake env.
This is due to the received rewards and way the percentile boundary fails to find the correct elite episodes
for FrozenLake. Remember that FrozenLake simply rewards the Agent with a 1 once the env is solved. And 0 if the
agent fails to reach the goal. This is not compatible with how the agent learned in CartPole.
See * TODO a detailed conclusion why CE fails in this context *  
'''
ENVIRONMENT = 'FrozenLake-v1'  # 'CartPole-v1'

LEARNING_RATE = 0.01

RENDER_MODE = None  # None || 'human'

'''
Each Episode is a collection of EpisodeStep instances.
'''
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

'''
The model that receives observations and returns actions
'''


class Net(Module):
    def __init__(self, observation_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = Sequential(
            Linear(observation_size, hidden_size),
            ReLU(),
            Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def iterate_batches(environment, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    observation = environment.reset()

    # environment.render()

    softmax = Softmax(dim=1)

    while True:
        observation_v = torch.FloatTensor([observation])
        action_probabilities_v = softmax(net(observation_v))

        # tensor([[0.4840, 0.5160]], grad_fn=<SoftmaxBackward0>)
        # print(action_probabilities_v)

        act_probs = action_probabilities_v.data.numpy()[0]
        # [0.48396415 0.5160358 ]
        # print(act_probs)

        action = np.random.choice(len(act_probs), p=act_probs)
        # 0 || 1
        # print(action)

        if ENVIRONMENT == "CartPole-v1":
            next_observation, reward, is_done, _, _ = environment.step(action)
        elif ENVIRONMENT == "FrozenLake-v1":
            next_observation, reward, is_done, _ = environment.step(action)
        else:
            print("not frozenlake not cartpole? What is it? ")

        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=observation, action=action))

        '''
        episode_steps:  [EpisodeStep(observation=array([ 0.03989746,  0.02226709,  0.02531616, -0.0003232 ], dtype=float32), action=0), EpisodeStep(observation=array([ 0.03989746,  0.02226709,  0.02531616, -0.0003232 ], dtype=float32), action=0), EpisodeStep(observation=array([ 0.03989746,  0.02226709,  0.02531616, -0.0003232 ], dtype=float32), action=1), EpisodeStep(observation=array([ 0.03989746,  0.02226709,  0.02531616, -0.0003232 ], dtype=float32), action=0), EpisodeStep(observation=array([ 0.03989746,  0.02226709,  0.02531616, -0.0003232 ], dtype=float32), action=0), EpisodeStep(observation=array([ 0.03989746,  0.02226709,  0.02531616, -0.0003232 ], dtype=float32), action=0), EpisodeStep(observation=array([ 0.03989746,  0.02226709,  0.02531616, -0.0003232 ], dtype=float32), action=1), EpisodeStep(observation=array([ 0.03989746,  0.02226709,  0.02531616, -0.0003232 ], dtype=float32), action=1), EpisodeStep(observation=array([ 0.03989746,  0.02226709,  0.02531616, -0.0003232 ], dtype=float32), action=0), EpisodeStep(observation=array([ 0.03989746,  0.02226709,  0.02531616, -0.0003232 ], dtype=float32), action=0), EpisodeStep(observation=array([ 0.03989746,  0.02226709,  0.02531616, -0.0003232 ], dtype=float32), action=1)]
        received next_observation:  [-0.06497169 -0.7713521   0.21111907  1.4888651 ]
        received reward:  1.0
        is episode done:  True
        total episode_reward:  12.0
        '''
        # print("received next_observation: ", next_observation)
        # print("received reward: ", reward)
        # print("is episode done: ", is_done)
        # print("total episode_reward: ", episode_reward)
        # print("episode_steps: ", episode_steps)

        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []

            next_observation = environment.reset()

            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []

        observation = next_observation


'''
iterate_batches returns a batch that needs to be filtered out, as determined by the reward percentile.
'''


def filter_batches(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    # print("rewards: ", rewards)

    reward_bound = np.percentile(rewards, percentile)
    # print("reward_bound: ", reward_bound)

    reward_mean = float(np.mean(rewards))

    training_observations = []
    training_actions = []

    for filter_iter_no, feature in enumerate(batch):
        if feature.reward < reward_bound:
            # print("item: %d with reward: %.3f is BELOW reward_bound: %.1f" % (filter_iter_no, feature.reward, reward_bound))
            continue

        # print("item: %d with reward: %.3f is ABOVE reward_bound: %.1f" % (filter_iter_no, feature.reward, reward_bound))
        training_observations.extend(map(lambda step: step.observation, feature.steps))
        training_actions.extend(map(lambda step: step.action, feature.steps))
        # print("training_observations: ", training_observations)
        # print("training_actions: ", training_actions)

    training_observations_v = torch.FloatTensor(training_observations)
    training_actions_v = torch.LongTensor(training_actions)

    return training_observations_v, training_actions_v, reward_bound, reward_mean


if __name__ == "__main__":
    print("main ... ")

    environment = gym.make(ENVIRONMENT, render_mode=RENDER_MODE)

    # environment = gym.wrappers.RecordEpisodeStatistics(environment, new_step_api=True)
    environment = DiscreteOneHotWrapper(environment)

    observation_size = environment.observation_space.shape[0]
    # print("observation_size: ", observation_size)

    n_actions = environment.action_space.n
    # print("action_space: ", environment.action_space.n)

    net = Net(observation_size=observation_size,
              hidden_size=HIDDEN_SIZE,
              n_actions=n_actions
              )

    # batch = iterate_batches(environment, net, BATCH_SIZE)

    '''
    for iter_no, batch in enumerate(batch):
        print("iter_no: ", iter_no)
        print("len(batch): ", len(batch))
    '''

    objective = CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir=".", logdir="./B" + str(BATCH_SIZE) + "_P" + str(PERCENTILE), comment="-cartpole-v1")

    '''
    go through the batches and filter out the below-percentile ones.
    '''
    for iter_no, batch in enumerate(iterate_batches(environment, net, BATCH_SIZE)):
        # print("iter_no: ", iter_no)
        observation_v, actions_v, reward_bound, reward_mean = filter_batches(batch, PERCENTILE)
        '''
        print("observation_v: ", observation_v)
        print("actions_v: ", actions_v)
        print("reward_bound: ", reward_bound)
        print("reward_mean: ", reward_mean)
        '''

        optimizer.zero_grad()

        action_scores_v = net(observation_v)
        # print("action_scores_v: ", action_scores_v)

        loss_v = objective(action_scores_v, actions_v)
        # print("loss_v: ", loss_v)
        loss_v.backward()

        optimizer.step()

        # if RENDER_MODE is None:
        #    print("%d: loss: %.3f, reward_mean: %.1f, reward_bound: %.1f" % (iter_no, loss_v.item(), reward_mean, reward_bound))
        print("%d: loss: %.3f, reward_mean: %.1f, reward_bound: %.1f" % (
        iter_no, loss_v.item(), reward_mean, reward_bound))

        '''
        send the metrics to tensorboard
        '''
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)

        if reward_mean > 199:
            print("Done -> solved with reward_mean: ", reward_mean)
            break
    writer.close()
