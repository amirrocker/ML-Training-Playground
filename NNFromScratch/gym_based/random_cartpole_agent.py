import random

import gym

environment = "CartPole-v1"

class Environment:

    def __init__(self):
        self.steps_left = 10

    def get_observation(self):
        return [0.0, 0.0, 0.0]

    def get_actions(self):
        return [0, 1]

    def is_done(self):
        return self.steps_left == 0

    def action(self, action):
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()

class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env):
        current_observation = env.get_observation()
        print("current_observation: ", current_observation)
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward

if __name__ == "__main__":
    env = gym.make(id=environment, new_step_api=True)
    # set reward
    total_reward = 0
    total_steps = 0

    env.reset()
    # a Discrete -> either 0 : left or 1 : right
    print(env.action_space)
    # a Box[4, ] dtype=float32
    print(env.observation_space)

    s = env.step(0)
    print(s)

    sample_action_space = env.action_space.sample()
    print(sample_action_space)

    sample_action_space2 = env.action_space.sample()
    print(sample_action_space2)

    environment = Environment()
    agent = Agent()
    while not environment.is_done():
        agent.step(environment)
        print("agent.total_reward: ", agent.total_reward)


