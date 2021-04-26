import gym
from _collections import namedtuple, deque
from .agent import BaseAgent

# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])

class ExperienceSource:
    '''
    Simple n-step experience source(buffer) using single or multiple environments
    Every experience contains n entries of Experiences
    '''
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        '''
        Create a simple experience source
        :param env:
        :param agent:
        :param steps_count:
        :param steps_delta:
        :param vectorized:
        '''
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)

        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool =  [env]

        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def __iter__(self):
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []
        for env in self.pool:
            obs = env.reset()
            # if the environment is vectorized all its output is lists of results.
            # Details to be found here: https://github.com/openai/universe/blob/master/doc/env_semantics.rst
            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)
            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_state())

        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    actions[idx] = self.pool[0].action_space.sample() # assume that all envs are from the same family
                else:
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                states_actions, new_agent_states = self.agent(states_input=states_input, agent_states=agent_states)
                for idx, action in enumerate(states_actions):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]
            grouped_actions = _group_list(actions, env_lens)

            global_ofs = 0
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    next_state_n, r_n, is_done, _ = env.step(action_n)
                else:
                    next_state, r, is_done, _ = env.step(action_n[0])