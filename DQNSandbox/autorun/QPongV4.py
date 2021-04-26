import torch
import DQNSandbox.wrappers.wrappers as wrappers
from brain import DQN, Experience, ExperienceBuffer
from agency import BasicAgent
from torch.optim import Adam
from loss_functions import calc_loss
import numpy as np
import time
import sys
from JSONCoder import fromJSON
from domain_layer import Training

'''
    Q-Learning with Bellman equation approach to store the experience inside an experience buffer
    and learn from the most successful episodes.  
    1. Inititalize Q(s,a) with random weights, an epsilon value of 1.0 and
    an empty experience buffer to store observations
    2. With a probability € select a random action a
    else action a = argmax a Q(s,a) , the epsilon greedy algorithm
    3.execute action a and observe the reward r and new state s'
    4. store transition (s, r, a, d, s') in experience buffer
    5. sample a random mini-batch of transitions from the replay buffer
    6. for every transition in the buffer, calculate the target Y = r if the episode has ended
    else Y = r - GAMMA arg max a' Q'(s',a')
    7. calculate the loss L = (Q(s,a)-Y)**2
    8. update Q(s,a) using a SDG algorithm to minimize loss
    9. every N steps copy the weights from Q to Q' target
'''

def doProcessing(args):

    print("received param args: %s" % args)

    # args is a list of n args
    assert args[0]
    assert args[1]

    training = fromJSON(args[1], Training)

    print("learningRate: %s" % training.hyperParameters.learningRate)
    print("gamma: %s" % training.hyperParameters.gamma)

    DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
    GAMMA = 0.98  # used in bellmann equation
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.1
    EPSILON_DECAY_RATE = 10 ** 5
    LEARNING_RATE = 1e-4

    BATCH_SIZE = 64  # play around with this parameter
    SYNC_TARGET_FRAMES = 10000

    REPLAY_SIZE = 20000
    REPLAY_START_SIZE = 5000
    MEAN_REWARD_BOUND = 19.5

    LOG_TO_SUMMARY_WRITER = True

    env_name = DEFAULT_ENV_NAME

    cuda = "cuda"
    device = torch.device(cuda if cuda else "cpu")
    #callLog("device: %s with type: %s is available: %s" % (device, device.type, torch.cuda.is_available()))
    #sendMessage("device: %s with type: %s is available: %s" % (device, device.type, torch.cuda.is_available()))
    env = wrappers.make_env(env_name)
    action = env.reset()

    if LOG_TO_SUMMARY_WRITER == True:
        print("create the SummaryWriter")
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    # create out model networks
    net = DQN(env.observation_space.shape, env.action_space.n, device=device)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n, device=device)

    experience_buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = BasicAgent(env, experience_buffer)

    optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
    epsilon = EPSILON_START
    best_mean_reward = None
    total_rewards = []

    frame_idx = 0
    ts_frame = 0
    ts = time.time()

    #callLog("starting training loop at: %s" % ts)
    #sendMessage("starting training loop at: %s" % ts, callback)

    while True:
        # start the training loop
        frame_idx += 1

        # calculate the epsilon greedy algorithm value to define exploration vs. exploitation
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_RATE)

        # now play epsiode - ask the agent to do next step and receive the reward as result of this step
        reward = agent.play_step(net, epsilon=epsilon, device=device)

        if reward is not None:
            # add the received reward to the array
            total_rewards.append(reward)
            # calculate the speed in fps
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            # update ts_frame and timestamp ts
            ts_frame = frame_idx
            ts = time.time()
            # calculate the mean reward
            mean_reward = np.mean(total_rewards[-100:])
            print("Frames: %d, done: %d games, mean reward: %.3f, epsilon: %.2f, speed: %.2f fps" % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed
            ))

            '''
            callLog("Frames: %d, done: %d games, mean reward: %.3f, epsilon: %.2f, speed: %.2f fps" % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed
            ))
            '''
            '''
            sendMessage("Frames: %d, done: %d games, mean reward: %.3f, epsilon: %.2f, speed: %.2f fps" % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed
            ), callback)
            '''

            if LOG_TO_SUMMARY_WRITER:
                print("log to tensorboard")
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("mean_reward", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)

            # env.render(mode="human")

            '''
            everytime our mean_reward for the last 100 episodes reaches a max -best_mean_reward - we save the model
            '''
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), env_name + "-best.dat")
                if best_mean_reward is not None:
                    print("best_mean_reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > MEAN_REWARD_BOUND:
                print("solved in %.d frames!" % frame_idx)
                break
        if len(experience_buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
        optimizer.zero_grad()
        batch = experience_buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        optimizer.step()
    print("Process Stopped ... ")
    if LOG_TO_SUMMARY_WRITER:
        writer.close()


import argparse
import json

def reassembleJSON(args):

    training = Training()
    for key in args:
        print(key)


    print("assembled Training: " % Training)
    return "sdkflsdf"

if __name__ == "__main__":
    print("__main__ called in QPongV4.py exec script")

    ''' 
    # good argument processing - based on documentation
    # setup the ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('JSON', metavar='json', type=str, help='JSON to parse')
    args = parser.parse_args()
    json_str = args.JSON
    print(json_str)
    '''
    list = ['QPongV4.py', '{id:', '1234-ID2,', 'version:', '999.9999.9999,', 'env:', 'env2,', 'file:', 'QPongV4.py,', 'path:', 'C:/Users/info/Documents/Development/PycharmProjects/SelfDeepQLearning/DQNSandbox/autorun,', 'modelParams:', '{model:', 'model2,', 'optimizer:', 'optimizer2,', 'loss:', 'loss2},', 'log:', '{log:', 'log2},', 'hyperParameters:', '{epochs:', 'epochs2,', 'learningRate:', '0.00001,', 'gamma:', '0.98,', 'epsilon_start:', '1.0,', 'epsilon_final:', '0.1,', 'epsilon_decay_rate:', '10**5}}']

    args = list # sys.argv
    print(args)
    print(len(args))
    argsDoubles = reassembleJSON(args)
    #print(argsDoubles)


    doProcessing("test")