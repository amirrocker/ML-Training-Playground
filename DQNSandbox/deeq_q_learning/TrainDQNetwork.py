'''
We implement the slightly modified Q-learning algorithm.

1. Init Q(s,a) with some initial approximation using random weights
2. By interacting with the environment obtain the tuple of (s,a,r,s')
3. Calculate loss: L=(Qs,a - r)**2 if episode has ended 
else L = (Q(s,a) - (r - sigma max(a') € A Q(s', a') )**2
otherwise
4. Update Q(s,a) using SGD algorithm by minimizing the loss with respect to 
the model parameters.
5. Repeat from step 2 until converged

Now we go one step further and add the experience buffer as well as loss:

1. Initialize Q(s,a) with a random weight, an epsilon value of 1.0 ( € <- 1.0 ) and 
an empty ExperienceBuffer to store the transitions.
2. With probability € select a random action a 
else a = arg max a(Q(s,a)) --> this defines the epsilon greedy algorithm
3. Execute action a and observe the reward r and next_state s'
4. store transition (s, a, r, s') in ExperienceBuffer aka ReplayBuffer
5. Sample a random mini-batch of transitions from the replay buffer
6. for every transition in the buffer, calculate the target Y = r if the episode has ended
at this step or Y = r - sigma max a' € A Q' s', a' otherwise
7. calculate loss: L = (Q(s,a) - Y)**2
8. Update Q(s,a) using a SDG algorithm by minimizing the loss in respect to model parameters
9. Every N steps copy the weights from Q to Q't 
10. Repeat from step 2 until converged

let's kick some atari a**.... 
'''

from DQNSandbox.wrappers import wrappers as wrappers
from DQNSandbox.deeqQ_chapter_6.lib import dqn_model as dqn_model
import torch
import torch.nn as nn
import collections
import time
import numpy as np
#from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

'''
logging traceback - we need a better solution for that since we dont want to keep 
having to add this traceback / callback
'''
# import traceback
import sys
import warnings

def warn_with_traceback(message, category, filename, lineno, file=None, line=None ):
	log = file if hasattr(file, 'write') else sys.stderr
	log.write("")

warnings.showwarning = warn_with_traceback


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
BATCH_SIZE = 32

MEAN_REWARD_BOUND = 19.5
GAMMA = 0.99
BATCH_SIZE = 32
SYNC_TARGET_FRAMES = 1000

# max capacity of the buffer
REPLAY_SIZE = 10000
# count of frames to wait before starting training
REPLAY_START_SIZE = 10000
# learning rate used in Adam optimizer
LEARNING_RATE = 1e-4

EPSILON_DECAY_LAST_FRAME = 10**5  # the decay rate of our epsilon greedy
EPSILON_START = 1.0
EPSILON_FINAL = 0.02  # 2 percent random behaviour - 98 percent 'learned' behavior

'''
Experience is the datastructure to keep our 'memories'
'''
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])

'''
A helper class that allows for experience management
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
		states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
		return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)

class Agent:
	def __init__(self, env, experience_buffer):
		self.env = env
		self.experience_buffer = experience_buffer
		self._reset()

	def _reset(self):
		self.state = self.env.reset()
		self.total_reward = 0.0

	def play_step(self, net, epsilon=0.0, device="cpu"):
		done_reward = None
		if np.random.random() < epsilon:
			action = self.env.action_space.sample()
		else:
			state_a = np.array([self.state], copy=False)
			state_v = torch.tensor(state_a).to(device)
			q_vals_v = net(state_v)
			act_v, _ = torch.max(q_vals_v, dim=1)
			action = int(act_v.item())
		# now execute the action
		new_state, reward, is_done, _ = self.env.step(action)
		self.total_reward += reward
		next_state = new_state
		experience = Experience(self.state, action, reward, is_done, next_state )
		self.experience_buffer.append(experience)
		self.state = new_state
		if is_done:
			done_reward = self.total_reward
			self._reset()
		return done_reward

def calc_loss(batch, net, tgt_net, device="cpu"):
	# get the states, actions, rewards, dones and next_states from the received batch
	states, actions, rewards, dones, next_states = batch

	'''
	print("[calc_loss] calculating loss for Number of states: %d, actions: %d, rewards: %d, dones: %d, next_states: %d" %
	      (len(states), len(actions), len(rewards), len(dones), len(next_states))
	      )
	'''

	states_v = torch.tensor(states).to(device)
	next_states_v = torch.tensor(next_states).to(device)
	actions_v = torch.LongTensor(actions).to(device)
	rewards_v = torch.tensor(rewards).to(device)
	done_mask = torch.ByteTensor(dones).to(device)

	state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
	# print("[calc_loss] state_action_values: ", state_action_values)
	next_state_values = tgt_net(next_states_v).max(1)[0]
	# print("[calc_loss] next_state_values: ", next_state_values)
	next_state_values[done_mask] = 0.0
	next_state_values = next_state_values.detach()
	expected_state_action_values = next_state_values * GAMMA + rewards_v
	return nn.MSELoss()(state_action_values, expected_state_action_values.float())


if __name__ == "__main__":

	cuda = "cuda"
	env_name = DEFAULT_ENV_NAME

	env = wrappers.make_env(env_name)
	device = torch.device(cuda)
	# writer = SummaryWriter(comment="-"+env_name)

	net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
	tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

	experience_buffer = ExperienceBuffer(REPLAY_SIZE)
	agent = Agent(env, experience_buffer)

	optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
	epsilon = EPSILON_START

	best_mean_reward = None
	total_rewards = []

	frame_idx = 0
	ts_frame = 0
	ts = time.time()

	while True:
		# start the training loop
		frame_idx += 1

		# calculate epsilon greedy - exploration vs exploitation - decrease until epsilon > EPSILON_FINAL
		epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

		# now play one episode - ask the agent to do one next step and receive the reward as result of this step
		reward = agent.play_step(net, epsilon=epsilon, device=device)

		if reward is not None:
			# add the received reward to the array
			total_rewards.append(reward)
			# calculate the speed in fps
			speed = (frame_idx - ts_frame) / (time.time() - ts)
			# update ts_frame and timestep ts
			ts_frame = frame_idx
			ts = time.time()
			# calculate the mean reward
			mean_reward = np.mean(total_rewards[-100:])
			print("%d: done %d games, mean reward: %.3f, epsilon: %.2f, speed: %.2f fps " %
			      (frame_idx, len(total_rewards), mean_reward, epsilon, speed)
			)

			# add the values to the tensorboard
			writer.add_scalar("epsilon", epsilon, frame_idx)
			writer.add_scalar("speed", speed, frame_idx)
			writer.add_scalar("mean_reward", mean_reward, frame_idx)
			writer.add_scalar("reward", reward, frame_idx)

			'''
			every time our mean_reward for the last 100 epsidodes reaches a maximum - best_mean_reward - we save the model
			'''
			if best_mean_reward is None or best_mean_reward < mean_reward:
				# save the weights to a model file
				torch.save(net.state_dict(), env_name+"-best.dat")
				if best_mean_reward is not None:
					print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
				best_mean_reward = mean_reward
			if mean_reward > MEAN_REWARD_BOUND:
				print("solved in %d frames! " % frame_idx)
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
	writer.close()




