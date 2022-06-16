# nn from scratch
# check out sentdex


# this little thing models a single neuron on the input layer.
import numpy as np

inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2.0

input = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
print("input: {}".format(input))
# the output is the value that is being sent to the next layer neurons.

# now imagine you would model a single output layer neuron.
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + bias
print("output: {}".format(output))

#### simple so far.

#### now code three neurons with 4 inputs & weights per neuron and one bias value per neuron - bias is per neuron not per weight

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

'''
output = [inputs[0] * weights[0][0] + inputs[1] * weights[0][1] + inputs[2] * weights[0][2] + inputs[3] * weights[0][3] + bias[0],
            inputs[0] * weights[1][0] + inputs[1] * weights[1][1] + inputs[2] * weights[1][2] + inputs[3] * weights[1][3] + bias[1],
            inputs[0] * weights[2][0] + inputs[1] * weights[2][1] + inputs[2] * weights[2][2] + inputs[3] * weights[2][3] + bias[2]]

print("neuron_one: {}".format(output))
'''

### now let's rewrite it to be a bit more like 'acceptable' programming.

layer_outputs = []
for neuron_weight, neuron_bias in zip(weights, biases):
    print("neuron_weight: {}".format(neuron_weight))
    print("neuron_bias: {}".format(neuron_bias))
    for n_input, weight in zip(inputs, neuron_weight):
        print("n_input: {}, weight: {}".format(n_input, weight))
        neuron_output = n_input * weight
    neuron_output += neuron_bias
    print("neuron_output: {}, neuron_bias: {}".format(neuron_output, neuron_bias))
    layer_outputs.append(neuron_output)
print("layer outputs: {}".format(layer_outputs))

### how to calculate a dot product
# mul 2 matrices element wise with each other

m1 = [1, 2, 3, 4, 5, 6]
m2 = [6, 5, 4, 3, 2, 1]
m6x2 = [[6, 5, 4, 3, 2, 1],
        [6, 5, 4, 3, 2, 1]]  # or 2x6 matrix - TODO decide which

# dot[[a, b, c], [d, e, f]] = [a*d]+[b*e]+[c*f]
result_all = 0
for v1, v2 in zip(m1, m2):
    result = v1 * v2
    result_all += result
withBias = (result_all+bias)
print("result all: {}".format(withBias))

result_np = np.dot(m1, m2)+bias
assert result_np == withBias, 'not the same result'

### now, how about a matrix and vector dot product

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

dotResult = np.dot(weights, inputs) + biases
print("dotResult: {}".format(dotResult))

# or manually:
manual_output = [((weights[0][0] * inputs[0]) + (weights[0][1] * inputs[1]) + (weights[0][2] * inputs[2]) + (
        weights[0][3] * inputs[3])) + biases[0],
                 ((weights[1][0] * inputs[0]) + (weights[1][1] * inputs[1]) + (weights[1][2] * inputs[2]) + (
                         weights[1][3] * inputs[3])) + biases[1],
                 ((weights[2][0] * inputs[0]) + (weights[2][1] * inputs[1]) + (weights[2][2] * inputs[2]) + (
                         weights[2][3] * inputs[3])) + biases[2]
                 ]
print("manual_output: {}".format(manual_output))

'''
short look at random:
'''
matrix_4x3 = np.random.randn(4, 3)
print("matrix_4x3: %s" % matrix_4x3)

# create an empty matrix
print(np.zeros((1, 4)))

'''
since we will be using a number of numpy features and my numpy fitness is a bit like my regular fitness, never good enough :)
let's take a short look at indexing in numpy
'''
random_matrix = np.random.rand(2, 3)
print("random_matrix: %s" % random_matrix)

zeros = np.zeros((2, 3))
print("zeros: %s" % zeros)

# slicing and dicing those babies
sliced_to_cubes = random_matrix[:, 2]
print("sliced_to_cubes: %s" % sliced_to_cubes)

sliced_to_slices = matrix_4x3[1:2, :]
print("sliced_to_slices: %s" % sliced_to_slices)

sliced_to_reverse = matrix_4x3[::-2, ::-1]
print("sliced_to_reverse: %s" % sliced_to_reverse)

sliced_to_keep_shape = matrix_4x3[:, [1]]  # all rows, one column
print("sliced_to_keep_shape: %s" % sliced_to_keep_shape)

# TODO I have to look this up to make sure this is correct
sliced_to_odd_numbers = matrix_4x3[:, 0:3:2]
print("sliced_to_odd_numbers: %s" % sliced_to_odd_numbers)

# to preserve dimensionality also possible
sliced_to_keep_shape_too = matrix_4x3[:, np.newaxis, :]
print("sliced_to_keep_shape_too: %s" % sliced_to_keep_shape_too)

from numpy.random import randn

# okay - lets create some more random data
array_one = np.array(randn(3, 3, 3) * 10, dtype=np.int64)
array_two = np.array(randn(3, 3, 3) * 10, dtype=np.int64)
print("array_one: %s" % array_one)
print("array_two: %s" % array_two)

'''
Pandas - no not the cuddly animals - but the python data management library.

lets start with a regular DataFrame, the main type of pandas along with Series.
'''
import pandas as pd

dataFrame = pd.DataFrame(np.random.randn(7, 3) * 10,
                         columns=["aaa", "bbb", "ccc"],
                         index=list("agbfced")
                         )
print("pandas dataFrame: %s" % dataFrame)
