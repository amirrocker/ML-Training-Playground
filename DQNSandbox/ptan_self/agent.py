import torch
import copy
import numpy as np
import torch.nn.functional as F

from ptan_self.actions import ProbabilityActionSelector

from qlearning.brain import DQN, ExperienceBuffer, Experience


class BaseAgent:
    def initial_state(self):
        return None
    def __call__(self, states, agent_states):
        '''
        convert observations and states into actions tot take
        :param states: list of env states to process
        :param agent_states: list of ststes with the same length as observations
        '''
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError

def default_states_preprocessor(states):
    '''
    convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    '''
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)

'''

'''
def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)


class DQNAgent(BaseAgent):
    '''
    a memoryless DQN agent which calculates Q values
    from the observations and converts them into actions using
    action_selector
    '''
    def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=default_states_preprocessor):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    @torch.no_grad()
    def __call__(self, states, agent_states = None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states
        

class TargetNet:
    '''
    Wrapper around model which provides copy of it instead of trained weights
    We periodically sync the weights over to the target net
    @See stabilizing training p.358 ff.
    '''
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        '''
        Blend params of target net with params from the model
        :param alpha:
        '''
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()

        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1-alpha) * v
        self.target_model.load_state_dict(tgt_state)


class PolicyAgent(BaseAgent):
    '''
    Policy agent gets action probabilities from the model and samples action from it
    '''
    # TODO unity code - only one selector differs
    def __init__(self, model, action_selector=ProbabilityActionSelector(), device="cpu", apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        '''
        Return the actions from a given list of states
        :param states: a list of states
        :return: list of states
        '''
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states

class ActorCriticAgent(BaseAgent):
    '''
    Policy agent which returns policy and value tensors from observations. Values are stored in agent's state
    and could be reused for rollouts calculations by ExperienceSource.
    '''
    def __init__(self, model, action_selector=ProbabilityActionSelector(), device="cpu", apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        '''
        Return actions from given list of states
        :param states: list of states
        :param agent_states:
        :return:
        '''
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v, values_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        agent_states = values_v.data.squeeze().cpu().numpy().tolist()
        return np.array(actions), agent_states