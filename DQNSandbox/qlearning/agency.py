import numpy as np
import torch
import torch.nn as nn

from brain import Experience


class BasicAgent(nn.Module):
    def __init__(self, env, experience_buffer):
        self.env = env
        self.experience_buffer = experience_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.2, device="cpu"):
        done_reward = None

        # calculate and decide on result whether we do exploration vs. exploitation
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a, device=device).to(device=device)
            q_values_v = net(state_v)
            act_v, _ = torch.max(q_values_v, dim=1)
            action = int(act_v.item())
        # now execute the calculated action
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        next_state = new_state
        experience = Experience(self.state, action, reward, is_done, next_state)
        self.experience_buffer.append(experience)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward




