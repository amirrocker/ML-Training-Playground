import os

'''
first version accepts 
    - hardcoded absolute path
    - security implications of calling os.system() - could be secured and sandboxed on an os level using docker
    - performance implications  

A simple and naive first implementation of a process spawning
executor class that receives a path prefix and the file to execute.
'''


class ModelFileExecutor:

    def __init__(self, path):
        self.path = path

    def execute(self, file):
        url = self.path + '/' + file
        assert len(url) != 0, 'No valid url was found'
        print("exec url: %s" % url)
        os.system('python ' + url)


'''
if __name__ == "__main__":
    PATH = 'C:/Users/info/Documents/Development/PycharmProjects/SelfDeepQLearning/DQNSandbox/qlearning'
    print("path: %s" % PATH)
    ModelFileExecutor(PATH).execute('TestDynamicExec.py')

    print("main ModelFileExecutor script")
'''
