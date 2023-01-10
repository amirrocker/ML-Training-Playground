import getopt
import sys
import socket
import getopt
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


def server_loop():
    global target

    # if no target is defined, listen on all interfaces
    if not len(target):
        target = "0.0.0.0"

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((target, port))
    server.listen(5)

    while True:
        client_socket, addr = server.accept()
        # spin off a thread to handle the new client
        # TODO find out how non-blocking clients work in python
        client_thread = threading.Thread(
            target=client_handler,
            args=(client_socket, )
        )
        client_thread.start()

def main():
    global listen
    global port
    global upload_destination
    global execute
    global command
    global target

    print("starting main .... ")

    # if no valid host is found
    if not len(sys.argv[1:]):
        print("no valid host found. calling usage()")
        usage()

    # read in cmd line options
    try:
        print("read cmd line: sys.argv: %s" % sys.argv)
        opts, args = getopt.getopt(sys.argv[1:], "hle:t:p:cu:",
                                   ["help", "listen", "execute", "target", "port", "command", "upload"])
    except getopt.GetoptError as err:
        print(str(err))
        usage()

    print("opts: %s" % str(opts))

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
            port = int(a)
        else:
            assert False, "unhandled option"

    # listen or send data from stdin?
    if not listen and len(target) and port > 0:
        print("len(target): %s" % len(target))
        buffer = sys.stdin.read()
        print("buffer from stdin: %s" % buffer)
        client_sender(buffer)

    if listen:
        print("server_loop is starting to listen on %s:%s" % (target, port))
        server_loop()

main()

def client_sender(buffer):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        #connect to target host
        client.connect((target, port))

        if len(buffer):
            client.send(buffer)

            while True:
                recv_len = 1
                response = ""

                while recv_len:
                    data = client.recv(4096)
                    recv_len = len(data)
                    response += data
                    print("building response: ", response)

                    if recv_len < 4096:
                        print("recv_len < 4096 -> break out")
                        break

                print("final response: ", response)

                buffer = raw_input("")
                buffer += "\n"

                # send it off to client
                client.send(buffer)
    except:
        print("except caught error - close and clean up")
        client.close()

def run_command(command):
    # trim newline
    command = command.rstrip()

    # run it
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    except:
        output = "Failed to execute command. /r/n"

    # send output back to client
    return output

# callback handler for client
def client_handler(client_socket):
    global upload
    global execute
    global command

    # check for upload
    if len(upload_destination):
        file_buffer = ""
        # keep reading until none is available
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            else:
                file_buffer += data

        try:
            file_descriptor = open(upload_destination, "wb")
            file_descriptor.write(file_buffer)
            file_descriptor.close()

            client_socket.send("Successfully saved file to %s/r/n" % upload_destination)
            print("Successfully saved file to %s/r/n" % upload_destination)
        except:
            client_socket.send("Failed to save file to %s/r/n" % upload_destination)
            print("except in client handler")

    if len(execute):
        # run command
        output = run_command(execute)
        client_socket.send(output)

    if command:
        while True:
            # show a simple prompt
            client_socket.send("<BHP:#> ")
            # receive until we hit enter key (see a linefeed)
            cmd_buffer = ""
            while "\n" not in cmd_buffer:
                cmd_buffer += client_socket.recv(1024)

            # send back command ouput
            response = run_command(cmd_buffer)

            # send back response
            client_socket.send(response)