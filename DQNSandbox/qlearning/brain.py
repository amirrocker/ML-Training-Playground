import torch.nn as nn
import collections
import numpy as np
import torch


class LinearNN(nn.Module):
    def __init__(self, input_shape, n_actions, device):
        super(LinearNN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            nn.ReLU()
        )

        if device.type == 'cuda':
            self.linear.cuda(device=device)




class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, device):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        if device.type == 'cuda':
            self.conv.cuda(device=device)

        conv_out_size = self._get_conv_out(input_shape, device)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        if device.type == 'cuda':
            self.fc.cuda(device=device)

    def _get_conv_out(self, input_shape, device):
        zeros = torch.zeros(1, *input_shape, device=device).to(device=device)
        o = self.conv(zeros)
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


'''
Each experience capsulates a number of observations made at this timestep. It is the basic memory data structure.  
'''
Experience = collections.namedtuple("Experience", field_names=[
    "state", "action", "reward", "done", "next_state"
])

'''
The buffer represents the collected memory of the agent.
'''


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, new_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(new_states)
