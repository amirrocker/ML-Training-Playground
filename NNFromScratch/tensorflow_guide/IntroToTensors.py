# based on
# https://www.tensorflow.org/guide/tensor

'''
Focus will be on indexing, especially multi-axis indexing.
'''

import numpy as np
import tensorflow as tf


'''
lets create a number of tensors
and start with a single Scalar, a Constant
rank-0-tensor
'''
tensor0_of_rank_0 = tf.constant(4)
tensor1_of_rank_0 = tf.constant(2.0)
print(tensor0_of_rank_0)  # what dtype is this constant ?
print(tensor1_of_rank_0)  # what dtype is this constant ?

# a rank_1 tensor
tensor0_of_rank_1 = tf.constant([2.0, 3.4, 1.8, 0.23563123], dtype=tf.float32)
print(tensor0_of_rank_1)

# a rank_2 tensor
tensor0_of_rank_2 = tf.constant([
    [1, 2],
    [3, 4],
    [5, 6],
])
print(tensor0_of_rank_2)

'''
tensors are built like so:
scalar = []
vector = [1, 2, 3]
matrix_1x3x2 = [  
    [1, 2], 
    [3, 4],
    [5, 6],
]
matrix_4x64x28 = [ # its dimension - row - columns d x r x c
    [
        [ # Alpha
           [0..28], #0
            .
            .
            .
           [0..28], #63 
        ],
        [ # Red
           [0..28], #0
            .
            .
            .
           [0..28], #63 
        ],
        [ # Green
           [0..28], #0
            .
            .
            .
           [0..28], #63 
        ],
        [ # Blue
           [0..28], #0
            .
            .
            .
           [0..28], #63 
        ],
    ]
]
'''
# this image simply has a height of 2 pixels only
tensor_4_rank_0 = tf.constant(
    [
        [[0, 1, 2, 3, 4, 5, 6],
         [7, 8, 9, 10, 11, 12, 13]],

        [[14, 15, 16, 17, 18, 19, 20],
         [21, 22, 23, 24, 25, 26, 27]],

        [[28, 29, 30, 31, 32, 33, 34],
         [35, 36, 37, 38, 39, 40, 41]],

        [[42, 43, 44, 45, 46, 47, 48],
         [49, 50, 51, 52, 53, 54, 55]],
    ]
)
print(tensor_4_rank_0)  # shape = (4, 2, 7)

# convert tensor to numpy array
numpyArray_from_array = np.array(tensor_4_rank_0)
print("numpyArray_from_array: ")
print(numpyArray_from_array)

numpyArray_from_numpy = tensor_4_rank_0.numpy()
print("numpyArray_from_numpy: ")
print(numpyArray_from_numpy)

# basic math on tensors
a = tf.constant([[1, 2, 3, 4]])
b = tf.constant([[3, 4, 5, 6]])

a_plus_b = tf.add(a, b)
print("a plus b: ")
print(a_plus_b)

a_times_b = tf.multiply(a, b)
print("a times b: ")
print(a_times_b)

# matmul
# a_matmul_b = tf.matmul(a, b)
# print("a matmul b: ")
# print(a_matmul_b)

# Variables
'''
A variable is the recommended way to represent shared, persistent state the program manipulates.
(see: https://www.tensorflow.org/guide/variable )
'''
# a variable from existing tensor
a_variable = tf.Variable(tensor0_of_rank_0)

# b variable with tensor initialization
b_variable = tf.Variable([False, True, None, False, False, True])
print("b_variable: ", b_variable)