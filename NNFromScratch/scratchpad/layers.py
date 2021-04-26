import numpy as np
import math
from data_creation import create_data
import matplotlib.pyplot as plt

# a very simple layer object - it may grow with time.
class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 *  np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) # 1, n_neurons since bias is per neuron not per weight

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases


'''
ReLU as can be seen in the print of the result leads to values being dropped 
and potentially lost when trying to calc the error.
'''
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


'''
Instead of ReLU try out Softmax which is a non-linear rather than linear such as ReLU

First as tried out in exponentation_sketchpad.py use exponentiation and normalization
to implement a Softmax activation function:
softmax steps:
Input -> Exponentiate -> Normalize -> Output

'''


class Activation_Softmax:
    def forward(self, inputs, axis=1):
        '''
        :param inputs:
        :param axis:
        :return:
        # wexp = np.subtract(inputs, max(inputs))
        exp_values = np.exp(np.subtract(inputs, max(inputs)))
        self.output = exp_values / np.sum(exp_values, axis=axis, keepdims=True)
        '''
        exp_values = np.exp(inputs - np.max(inputs, axis=axis, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=axis, keepdims=True)



'''
Now after all the testing concepts and trying out the different moving parts we
should be ready to try to assemble them into a simple, but working, neural network. 
we use the created spiral data from cs231 :    
https://cs231n.github.io/neural-networks-case-study/
'''
X, y = create_data(samples=100, classes=3)

'''
looking at the created data, we should see a non-linear distribution of sample points.
from MatplotlibLibrary import ConfigurableScatter

print("X: %s" % X)
ConfigurableScatter(X[:, 0], X[:, 1]).withLabels(y, "Created Test data", "X", "Y")
'''




inputLayer = Layer_Dense(2, 5)
activation_relu = Activation_ReLU()
inputLayer.forward(X)
activation_relu.forward(inputLayer.output)

hiddenLayer1 = Layer_Dense(5, 5)
activation_softmax = Activation_Softmax()

hiddenLayer1.forward(inputLayer.output)
#print("MiddleLayer::output: %s" % hiddenLayer1.output)
activation_softmax.forward(hiddenLayer1.output)

print("hiddenLayer1::output: %s" % hiddenLayer1.output)
print("hiddenLayer1 activation_softmax::output: %s" % activation_softmax.output)









