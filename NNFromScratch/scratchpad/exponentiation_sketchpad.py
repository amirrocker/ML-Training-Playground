import math
from math import fsum
import numpy as np

layer_output = [4.8, 1.21, 2.385]

# Euler: 2.71828182846
E = math.e

exp_values = []

for output in layer_output:
    exp_values.append(E**output)

print("exponentation values: %s" % exp_values)



# formalize in function:

def exponential(inputs):
    E = math.e
    exp_values = np.exp(inputs)  # using numpy rather than doing it ourselves
    print("exponentation values: %s" % exp_values)
    return exp_values

print("function exponentation values: %s" % exponential(layer_output))

# normalization


U = [1, 2]
# y = U / (sum(U))

y1 = U[0] / (U[0]+U[1])
y2 = U[1] / (U[0]+U[1])
print(y1)
print(y2)

firstInput = U[0]
secInput = U[1]

def normalize(input):
    return input / fsum(U)

print(normalize(firstInput))
print(normalize(secInput))

# different approach using regular sum
exp_values = exponential(layer_output)
sum_exponent_values = sum(exp_values)

'''
normalized_values = []

for normalized_value in exp_values:
    normalized_values.append( normalized_value / sum_exponent_values )
# the prob. distr. over our values-
print("normalized_value: %s" % normalized_values)

# sanity check to see it really adds up :)
print("sum should be 1: %s" % sum(normalized_values))


With the above switch to numpy we can rewrite like so: 
'''

norm_values = exp_values / np.sum(exp_values)
norm_values2 = exp_values / sum_exponent_values
print(norm_values)
print(norm_values2)


### lets look at np.sum real quick :

matrix = [[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12]]

print(np.sum(matrix)) # this is NOT what we want. A single scalar is not desired result. instead we want 3 scalars, one for each row
# try axis param
print(np.sum(matrix, axis=1)) # much better, but still not quite the right shape. Again one result per row, now we have three results in one row
# use the keepdims=True parameter of the sum function
print(np.sum(matrix, axis=1, keepdims=True)) # awesome - now we are getting there...

'''
Note: code was copied over from 'layers.py' so ref and packages may need to be adjusted once uncommented.


    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

outputLayer = Layer_Dense(5, 3)
outputLayer.forward(hiddenLayer1.output)
print("OutputLayer::output: %s" % outputLayer.output)


# simple short test section:
activationReLU = Activation_ReLU()

inputLayerReLU = Layer_Dense(4, 5)
print("original random inputLayer::weights: %s" % inputLayerReLU.weights)
inputLayerReLU.forward(X)
print("layer forward -> inputLayer::output: %s" % inputLayerReLU.output)
activationReLU.forward(inputLayerReLU.output)
print("activation forward -> activationReLU::output: %s" % activationReLU.output)

# test softmax
layer_output = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]

activation_softmax = Activation_Softmax()
activation_softmax.forward(layer_output)
print("softmax output: %s" % activation_softmax.output)
'''

'''
looking at these print values, from our softmax activation function, we see that we are 
running into a exponentiation problem. Why do we suddenly see scientific notation values ? 
Well, if we keep calculating large exponential numbers we will at some point fail with an overflow. 
so overflow protection is necessary: 
v = u - max(u) 
this clamps the value down to a range from x=-inf and y=0 to x = 0 and y = 1  

# test the concept
vector_to_clamp = [0.3, 1.3, 2.3]
matrix_1x3_to_clamp = [ [1],
                        [2],
                        [3]]
matrix_2x3_to_clamp = [ [1, 4],
                        [2, 5],
                        [3, 6]]
matrix_3x3_to_clamp = [ [1, 4, 4],
                        [-2, 5, 8],
                        [-3, 6, 7]]

max_value = max(vector_to_clamp)
max_value_matrix = max(matrix_1x3_to_clamp)
max_value_matrix_2x3 = max(matrix_2x3_to_clamp)
max_value_matrix_3x3 = max(matrix_3x3_to_clamp)
print(max_value)
print(max_value_matrix)
print(max_value_matrix_2x3)
print("3x3: %s" % max_value_matrix_3x3)

result_3x3 = np.subtract(matrix_3x3_to_clamp, max(matrix_3x3_to_clamp))
print(result_3x3)


# a 'layer' without exponentiation - three nodes
result = np.subtract(vector_to_clamp, max(vector_to_clamp))
print(result)

# and a 'layer' with exponentiation
#wexp = np.subtract(vector_to_clamp, max(vector_to_clamp))
activation_softmax.forward(vector_to_clamp, 1)
print("activation_softmax.forward() -> activation_softmax.output: %s" % activation_softmax.output)

'''