import getopt
import sys
import socket
import getapt
import threading
import subprocess

'''
Netcat is the swiss army knife of networking. To have it available if not installed
or deactivated, we use our own netcat impl.
'''

# globals
listen = False
command = False
upload = False
execute = ""
target = ""
upload_destination = ""
port = 0


# main functions, for handling command line input

def usage():
    print("BHP Netcat Tool")
    print("")
    print("usage: replacing_netcat.py -t target_host -p port")
    print("-l --listen      - listen on [host]:[port] for incoming connections")
    print(" TODO add more usage text .....")
    sys.exit(0)


def main():
    global listen
    global port
    global upload_destination
    global execute
    global command
    global target

    # if no valid host is found
    if not len(sys.argv[1:]):
        usage()

    # read in cmd line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hle:t:p:cu",
                                   ["help", "listen", "execute", "target", "port", "command", "upload"])
    except getopt.GetoptError as err:
        print(str(err))
        usage()

    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
        elif o in ("-l", "--listen"):
            listen = True
        elif o in ("-e", "--execute"):
            execute = a
        elif o in ("-c", "--commandshell"):
            command = True
        elif o in ("-u", "--upload"):
            upload_destination = a
        elif o in ("-t", "--target"):
            target = a
        elif o in ("-p", "--port"):
            port = a
        else:
            assert False, "unhandled option"

    # listen or send data from stdin?
    if not listen and len(target) and port > 0:
        buffer = sys.stdin.read()
        print("client_sender(buffer) not yet impl.")
        # client_sender(buffer)

    if listen:
        print("server_loop not yet impl.")
        # server_loop()

main()