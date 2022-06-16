# import qlearning.DQNSandbox as sandbox

import multiprocessing
import time
from threading import current_thread

import numpy as np
# from rx.core import Observer
import requests
import rx
import torch
from rx import operators as ops
from rx.scheduler import ThreadPoolScheduler
from torch.optim import Adam

import wrappers as wrappers
from agency import BasicAgent
from brain import DQN, ExperienceBuffer
from loss_functions import calc_loss

# using pykson which should actually be pygson but well....

# from qlearning.MessageClient import sendMessageOnClient

print("Runner runs .... ")

# app = Flask(__name__)
# api = Api(app)

''' 
def startTrainingProcess(observer, scheduler):
    observer.on_next("Alpha")
    observer.on_next("Beta")
    observer.on_next("Gamma")
    observer.on_next("Delta")

    observer.on_completed()
'''


# multiprocessing

def intense_calculation(value, connection):  # TODO rename the method
    print("starting sleep :: value: %s " % value)
    # time.sleep(random.randint(5, 20) )
    # doProcessing(connection)
    print("intense calculcating .... ")
    return value

# calculate the number of cpu's to use for ThreadScheduler
optimal_thread_count = multiprocessing.cpu_count()
pool_scheduler = ThreadPoolScheduler(optimal_thread_count)

'''

class Processing(Resource):
    def post(self):
        print("processing get ...")

        # xs = Observable.create(range(10))
        # d = xs.filter(
        #         lambda x: x%2
        #     ).subscribe(print)

        # source = create(startTrainingProcess)
        #
        # source.subscribe(
        #     on_next=lambda i: print("received {0}: ".format(i)),
        #     on_error=lambda i: print("received {0}: ".format(i)),
        #     on_completed=lambda: print("On Complete called .... ")
        # )

        # create first process
        # rx.of("alpha", "beta", "gamma", "delta", "epsilon").pipe(
        rx.of("training 1", "training 2").pipe(
            ops.map(lambda s: intense_calculation(s)),
            ops.subscribe_on(pool_scheduler)
        ).subscribe(
            on_next=lambda s: print("Process 1: {0} {1}".format(current_thread().name, s)),
            on_error=lambda e: print("Error e: {0}".format(e)),
            on_completed=lambda: print("Process 1 On Complete")
        )


        # doProcessing()
        return "Test String its running"

# api.add_resource(Processing, "/processing/<process_id>", methods=['POST'])
'''


class Session:
    def __init__(self, payload):
        self.payload = payload

    def getPayload(self):
        return self.payload

sessions = []


'''
Instead of using a route based approach where we would have to implement something
like a round-robin rotating algorithm, we opt to use a publish-subscribe approach
since that gives us flexibility out of the box and horizontal scalability.

We run this script as a scheduled task once at 
9:00 in the morning and listen for events for 10 hours. After that the service shuts down.
These are first test values - we need to make sure we find a good balance (performance/run time/cost)
as well as a good mechanism to manage these scripts. 

@app.route('/processing', methods=['POST'])
def flaskRunSessions():
    runSessions("testing processing with browser")
'''

'''
We removed the flask route annotation and added a message param 
which must be a json element containing the sessions to run.
@TODO Add the sessions json to jms message 
'''


def runSessions(trainingSession, connection):
    # payload = message #request.get_json()
    # trainingSession = createMockSessions(payload=message)

    '''
    we pass something like this: 
    
    {
        "id":"1234",
        "createdAt":"2020-04-18",
        "trainings":[
            {
                "id":1,
                "env":"env1",
                "file":"file1",
                "modelParams": {
                    "model":"model1",
                    "optimizer":"optimizer1",
                    "loss":"loss1"
                },
                "log":{
                    "log":"log1"
                },
                "hyperparams":{
                    "epochs":"epochs1",
                    "learningRate":"learningRate1"
                }
            }
        ]
    }
    which is deserialized into the corresponding classes.
    '''

    rx.of("training 1", "training 2").pipe(
        # rx.from_list(trainingSession.trainings).pipe(
        ops.map(lambda s: intense_calculation(s, connection)),
        ops.subscribe_on(pool_scheduler)
    ).subscribe(
        on_next=lambda s: print("Process 1: {0} {1}".format(current_thread().name, s)),
        on_error=lambda e: print("Error e: {0}".format(e)),
        on_completed=lambda: print("Process 1 On Complete")
    )
    # return jsonify({'quarks': payload})


# Processing()

def callLog(msg):
    message = "msg: %s" % format(msg)
    url = "http://localhost:8080/log"
    params = {"message" : message}
    requests.post(url=url, data=params)

def sendMessage(msg, callback):
    message = "msg: %s" % format(msg)

    callback.sendMessageOnClient(message)

    #connection.send(body=message, destination="training-session-start-topic")
    # conn.send(body = 'message to active mq 1', destination="training-session-start-topic")
    #url = "http://localhost:8080/log"
    #params = {"message" : message}
    #requests.post(url=url, data=params)

def doProcessing(callback):
    '''
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

    DEFAULT_ENV_NAME = "Pong-v0"
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

    env_name = DEFAULT_ENV_NAME  # could use argparse !

    cuda = "cuda"
    device = torch.device(cuda if cuda else "cpu")
    # callLog("device: %s with type: %s is available: %s" % (device, device.type, torch.cuda.is_available()))
    # sendMessage("device: %s with type: %s is available: %s" % (device, device.type, torch.cuda.is_available()), callback)
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

    # callLog("starting training loop at: %s" % ts)
    # sendMessage("starting training loop at: %s" % ts, callback)

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

            # sendMessage("Frames: %d, done: %d games, mean reward: %.3f, epsilon: %.2f, speed: %.2f fps" % (
            #    frame_idx, len(total_rewards), mean_reward, epsilon, speed
            # ), callback)

            if LOG_TO_SUMMARY_WRITER == True:
                print("log to tensorboard")
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("mean_reward", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)

            env.render(mode="human")

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
    # writer.close()


if __name__ == "__main__":
    # app.run(port="5002", debug=True)
    # TODO put FLask back in !
    doProcessing(None)
