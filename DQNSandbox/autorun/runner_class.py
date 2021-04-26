import rx
#import multiprocessing
from threading import current_thread
#from rx.scheduler import ThreadPoolScheduler
from rx import operators as ops
from pykson import Pykson
from domain_layer import TrainingSession, Training, HyperParameter, Log, ModelParameter


# calculate the number of cpu's to use for ThreadScheduler
#optimal_thread_count = multiprocessing.cpu_count()
#pool_scheduler = ThreadPoolScheduler(optimal_thread_count)

class SessionProcessor:

    def fromJSON(self, payload, clazz):
        payload.strip(" \n\r\t")
        assert len(payload) != 0, 'No valid JSON payload received'
        training_session = Pykson().from_json(payload, clazz)
        return training_session

    '''
    map the data from JSON to Objects. 
    for each training node a Training object is created and
    sent downstream.  
    '''
    def runSessions(self, trainingSession):

        trainingSession = self.fromJSON(trainingSession, TrainingSession)

        #path = 'C:/Users/info/Documents/Development/PycharmProjects/SelfDeepQLearning/DQNSandbox/qlearning'

        #rx.of("training 1", "training 2").pipe(
        rx.from_list(trainingSession.trainings).pipe(
            ops.map(lambda s: run_training(s)),
            #ops.subscribe_on(pool_scheduler)
        ).subscribe(
            on_next=lambda s: print("on_next: {0} {1}".format(current_thread().name, s)),
            on_error=lambda e: print("on_error: {0}".format(e)),
            on_completed=lambda: print("on_complete")
        )
        print("SessionProcessor runSessions called")


# multiprocessing
'''
Run each training node as a spawned os process. This is a first prototype
version and no serious thoughts have yet been put into
- security
- deployment/containerization
- performance
for now it is simply about getting that thing to run :))) 
'''

def run_training(value): # TODO rename this asap

    assert len(value.id) != 0, 'invalid value argument in run_training'
    assert len(value.file) != 0, 'invalid value.file argument in run_training'
    assert len(value.path) != 0, 'invalid value.path argument in run_training'
    print("value.file: %s" % value.file)
    print("value.path: %s" % value.path)
    #model = ModelFileExecutor(value.path)
    #model.execute(value)

    print("intense calculcating .... ")
    return value


'''
This is the OLD back channel to the broker for all responses during training.
It goes against the idea of decoupled services to pass in a callback dependency! duh

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

'''
