import stomp
import time
import sys
from qlearning.runner import runSessions

class MyListener(stomp.ConnectionListener):
    def on_error(self, headers, body):
        #super().on_error(headers, body)
        print('received error "%s" ' % body)

    def on_message(self, headers, body):
        #super().on_message(headers, body)
        print('received message headers: "%s" and message: "%s" ' % (headers, body))
        runSessions(body)


hosts = [('localhost', 61613)]

conn = stomp.Connection(host_and_ports=hosts)
conn.set_listener('', MyListener())
conn.connect('admin', 'admin', wait=True, headers={'client-id':'clientname'})
conn.subscribe(destination="training-session-start-topic", id=1,  ack='auto', headers={'subscription-type':'MULTICAST', 'durable-subscription-name':'training-session-start-topic'} )


#conn.send(body = 'message to active mq 1', destination="training-session-start-topic")

#conn.send(body = "message to active mq 2", destination="training-session-start-topic")

time.sleep(36000)

conn.disconnect()
