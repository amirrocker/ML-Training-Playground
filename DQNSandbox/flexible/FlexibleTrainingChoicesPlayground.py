'''
What we want to achieve:
We want to have a flexible system that allows for choosing different files based on a number of parameters
passed along as payload. E.g. imagine we have 5 modeling algorithms we need to use for training. And 5 loss functions,
5 optimizers and so on. And now we want to be able to define which combination of parameters to be executed based on the
payload given to the function.

Given the simple json structure that defines the payload like so:
The payload should be structured using two categories:
- file dependent parameters
- function dependant parameters

file dependent parameters help to decide which file is to be used. 


{
    "id": "9753-2211-4711",
    "trainingID": "TID-1234-1234-1234",
    "createdAt": "2020.01.09 23:34:18",

    # the address part of the service ? How do we handle routing? Is it necessary to handle it here? Or better encapsuled
    # inside the gateway?
    "serviceID": "23454-ServiceID",
    "serviceName": "23454-ServiceName",
    "serviceIP": "23454-ServiceIP",
    "serviceQueue": "23454-ServiceQueue-975",

    "trainings": [
        {
              # Each training needs to have an id to be identified.
              "id": "1",

              # the atari environment to train on.
              # ["PongNoFrameskip-v4", "PongNoFrameskip-v6", "Stickman-v0"] # look up the possible values here:
              # http:// ???  TODO look up where again
              "env": "PongNoFrameskip-v4",

              # Each training uses a file that is called. This file must conform to a certain interface like
              # Runnable, Callable or Executable so a specific function name can be invoked on that file.
              "file": "file1",

              # Training Sessions use an agency to aquire the defined Agent to handle the lifecycle of a
              # Training.
              "agent" : "BasicAgent"

              # create out model networks
              # net = DQN(env.observation_space.shape, env.action_space.n, device=device)
              # this parameter defines which type of neural net to use.
              "model": "DQN",

              # optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
              "optimizer": "optimizer1",

              # loss_t = calc_loss(batch, net, tgt_net, device)
              "loss": "calc_loss",  # [mean_squared_error, mean_abs_error, hinge, sparse_categorical_crossentropy]

              "log": "Some Log String",
              "epochs": "32",
              "gamma":".99",
              "epsilon_start":"", // the exploration algorithm start value
              "epsilon_final":"", // the exploration algorithm end value
              "epsilon_decay_rate":"", // the exploration algorithm decay value
              "batch_size":"",
              "sync_target_frames":"",
              "replay_size":"",
              "replay_start_size":"",
              "mean_reward_bound":"",


              "learningRate": "learningRate"
        },
        ....
    ]
}

visualizes the different options necessary.
Each selection needs to adhere to above parameters.
Question: How can we correlate the above parameters with their corresponding files or functions?

Solution 1:
using the payload parameters as simple variable values. That would lead to a solution where the message receiver

TODO JSON does not quite match the params for DQNSandbox, the first file to use as "prototype template"
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
GAMMA = 0.99  # used in bellmann equation
EPSILON_START = 1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY_RATE = 10**5
LEARNING_RATE = 0.00001

BATCH_SIZE = 32  # play around with this parameter
SYNC_TARGET_FRAMES = 10000

REPLAY_SIZE = 10000
REPLAY_START_SIZE = 1000
MEAN_REWARD_BOUND = 19.5

env_name = DEFAULT_ENV_NAME  # could use argparse !

cuda = "cuda"
device = torch.device(cuda if cuda else "cpu")
#callLog("device: %s with type: %s is available: %s" % (device, device.type, torch.cuda.is_available()))
sendMessage("device: %s with type: %s is available: %s" % (device, device.type, torch.cuda.is_available()), callback)
env = wrappers.make_env(env_name)
action = env.reset()
# writer = SummaryWriter()

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

'''


# entry point into the algo....
def runFlexChoices():
    print("We are running flex choices...")

