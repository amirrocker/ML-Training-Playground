'''
For more information on the gym environments --
https://github.com/openai/universe/blob/master/doc/protocols.rst
-------------------------------------------------------------------
'''
import gym
import torch
import random
import collections
from torch.autograd import variable
import numpy as np
from collections import namedtuple, deque
from .agent import BaseAgent
from .common import utils
from .actions import *

HYPERPARAMETERS = {
    'pong' : {
        'env_name' : 'PongNoFrameskip-v4',
        'stop_reward' : 18.0,
        'run_name' : '',
        'replay_size' : '',
        'replay_initial' : '',
        'target_net_sync' : '',
        'epsilon_frames' : '',
        'epsilon_start' : '',
        'epsilon_final' : '',
        'learning_rate' : '',
        'gamma' : '',
        'batch_size' : ''
    }
}