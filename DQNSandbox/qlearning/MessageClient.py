import stomp
import time
import sys
from qlearning.runner import runSessions

'''
Check here for activeMQ console: 
http://localhost:8161/admin
'''


# check the min requirements
if not sys.version_info > (2, 7):
    # you are using a 10-year old python installation!!! Update! Duh....!
    print("you are using a 10-year old python installation!!! Update! Duh....!")
elif not sys.version_info >= (3, 5):
    # well, she is using an ok version. Be kind, at least its 3.5 ;)
    print("please update your relatively up to date installation - you are using 3.5.")

class MyListener(stomp.ConnectionListener):
    def on_error(self, headers, body):
        #super().on_error(headers, body)
        print('received error "%s" ' % body)

    def on_message(self, headers, body):
        #super().on_message(headers, body)
        print('received message headers: "%s" and message: "%s" ' % (headers, body))
        runSessions(body, self)

    def sendMessageOnClient(self, message):
        conn.send(body=message, destination="training-session-update-topic")
        print("Message has been sent...")


hosts = [('localhost', 61613)]

conn = stomp.Connection(host_and_ports=hosts)
conn.set_listener('', MyListener())
conn.connect('admin', 'admin', wait=True, headers={'client-id':'clientname'})
conn.subscribe(destination="training-session-start-queue", id=1,  ack='auto', headers={'subscription-type':'MULTICAST', 'durable-subscription-name':'training-session-start-topic'} )


registerMessage = '''{
  "id": "9753-2211-4711",
  "serviceID": "23454-ServiceID",
  "serviceName": "23454-ServiceName",
  "serviceIP": "23454-ServiceIP",
  "serviceQueue": "23454-ServiceQueue-975",
  "env": "env2",
  "file": "file2",
  "model": "model2",
  "optimizer": "optimizer2",
  "loss": "loss2",
  "log": "log2",
  "epochs": "epochs2",
  "learningRate": "LearningRate2"
}'''



conn.send(body = registerMessage, headers={"DocumentType":"de.amirrocker.training.BasicServiceBundle"}, destination="training-service-register-topic")





#conn.send(body = 'message to active mq 1', destination="training-session-start-topic")

#conn.send(body = "message to active mq 2", destination="training-session-start-topic")

time.sleep(36000)

conn.disconnect()