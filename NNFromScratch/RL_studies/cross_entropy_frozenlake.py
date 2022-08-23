import random

import gym
import numpy as np
import tensorflow as tf
import torch
from tensorboardX import SummaryWriter
from torch.nn import Softmax, CrossEntropyLoss
from torch.optim import Adam

from NNFromScratch.RL_studies.DiscreteWrapper import DiscreteOneHotWrapper
from NNFromScratch.RL_studies.cross_entropy import Episode, EpisodeStep, Net

logdir = "logs"

tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

np.set_printoptions(suppress=True)

RENDER_MODE = None  # 'human'

HIDDEN_SIZE = 128
# BATCH_SIZE = 16
# tweaked
BATCH_SIZE = 125

# episode rewards below this percentile are thrown away
# PERCENTILE = 70
# tweaked
PERCENTILE = 20

GAMMA = 0.95

# used by gradient descent to define "step size"
LEARNING_RATE = 0.01

'''
This shows the limitations of cross-entropy as a model. It works well for the simple 
left-right cart-pole system. But in FrozenLake, which does have a different reward system, 
cross-entropy cannot converge. Even with some more advanced techniques, such as 
looking into future rewards, 
keeping elite episodes and 
adding a discount factor GAMMA, 
the model does not learn the episodes well.
Maybe some more tweaking of hyper-params may help but also to reach at least 50% successfull episodes at least 5K of iterations are necessary since
successfull episodes are quite rare and must be used longer than when using them for cartPole.
 
'''


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []

    next_observation = env.reset()
    softmax = Softmax(dim=1)

    while True:

        # VERY SLOW! adjust as per warning
        observation_v = torch.FloatTensor([next_observation])

        action_probs_v = softmax(net(observation_v))

        action_probs = action_probs_v.data.numpy()[0]

        action = np.random.choice(len(action_probs), p=action_probs)

        next_observation, reward, is_done, _, _ = env.step(action)

        episode_reward += reward

        # add episode step
        episode_steps.append(EpisodeStep(observation=next_observation, action=action))

        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_observation = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        next_observation = next_observation


def filter_batches(batch, percentile):
    discrete_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))

    # print("discrete_rewards: ", discrete_rewards)

    reward_boundary = np.percentile(discrete_rewards, percentile)
    # reward_mean = np.mean(reward_boundary)

    # print("reward_mean: ", reward_mean)

    training_observations = []
    training_actions = []

    # tweaked
    elite_batches = []

    for feature, discounted_reward in zip(batch, discrete_rewards):
        if discounted_reward > reward_boundary:
            training_observations.extend(map(lambda s: s.observation, feature.steps))
            training_actions.extend(map(lambda s: s.action, feature.steps))
            elite_batches.append(feature)

    return elite_batches, training_observations, training_actions, reward_boundary


if __name__ == "__main__":

    random.seed(12345)

    environment = gym.make("FrozenLake-v1", render_mode=RENDER_MODE, new_step_api=True)

    environment = DiscreteOneHotWrapper(environment)

    net = Net(environment.observation_space.shape[0], HIDDEN_SIZE, environment.action_space.n)

    loss = CrossEntropyLoss()

    optimizer = Adam(params=net.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(logdir="logs", comment="-FrozenLake_tweaked")
    # print("environment.observation_space: ", environment.observation_space)
    # print("environment.action_space: ", environment.action_space)

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env=environment, net=net, batch_size=BATCH_SIZE)):

        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))

        full_batch, observation_v, actions_v, reward_boundary = filter_batches(full_batch + batch, PERCENTILE)
        # print("observation_v: ", observation_v)
        # print("actions_v: ", actions_v)
        # print("reward_boundary: ", reward_boundary)
        # print("reward_mean: ", reward_mean)

        if not full_batch:
            continue

        observation_v = torch.FloatTensor(observation_v)
        actions_v = torch.LongTensor(actions_v)

        full_batch = full_batch[-500:]

        optimizer.zero_grad()

        action_scores_v = net(observation_v)

        loss_v = loss(action_scores_v, actions_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss: %.3f, reward_mean: %.1f, reward_bound: %.1f" % (
            iter_no, loss_v.item(), reward_mean, reward_boundary))

        '''
        send the metrics to tensorboard
        '''
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_boundary, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)

        if reward_mean > 0.8:
            print("Done -> solved with reward_mean: ", reward_mean)
            break
    writer.close()
