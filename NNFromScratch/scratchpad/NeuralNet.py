'''
given what we have done so far inside the layer.py file and the experiments and concepts we looked at
in the scratchpad, now a formalized approach could be formulated.
This is a first version sketch of a simple net.
'''
# from layers import Layer_Dense

import numpy as np

'''
always remember: multiply by weight to add non-linearity and add some bias 
'''


class NeuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # set the different nodes or layers ?
        print("__init__ called")
        # these set the number of nodes we want to create
        self.input_nodes = inputNodes
        self.hidden_nodes = hiddenNodes
        self.output_nodes = outputNodes
        # next initialize the starting random weight values
        # wih -> weightInputHidden

        self.weightInputHidden_simple = np.random.rand(self.input_nodes, self.hidden_nodes)
        print(self.weightInputHidden_simple)

        '''
        the np.random.normal samples a normal distribution.
        parameters are:
        center of the distribution - 0.0
        the std deviation (or range?) of the distribution - self.hidden_nodes - 0.5 -> clamps values from -.5 to .5
        the shape of the desired matrix (if we want a matrix instead of a simple scalar)
        '''
        self.weightInputHidden = np.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                                  (self.hidden_nodes, self.input_nodes))
        print("self.weightInputHidden: %s" % self.weightInputHidden)

    def train(self):
        # train the network
        print("train called")

    def query(self):
        # ask the network to predict or classify
        print("query the network")


# like usual my lazy 'unit' testing section :)

print("create the net...")
net = NeuralNetwork(inputNodes=3, hiddenNodes=4, outputNodes=3, learningRate=0.000001)
print(net.hidden_nodes)
