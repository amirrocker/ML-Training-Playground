# https://towardsdatascience.com/recurrent-neural-networks-rnn-explained-the-eli5-way-3956887e8b75

# https://www.simplilearn.com/tutorials/deep-learning-tutorial/rnn

# finally settled on
# https://www.guru99.com/rnn-tutorial.html

import numpy as np
import tensorflow as tf

nInputs = 4
nNeurons = 6
nTimesteps = 2

# three batches of 0..9 sequences
xBatch = np.array([
    [[0, 1, 2, 3, 4], [9, 8, 7, 6]],  # batch 1
    [[0, 1, 2, 3, 4], [9, 8, 7, 6]],  # batch 2
    [[0, 1, 2, 3, 4], [9, 8, 7, 6]],  # batch 3
])


# define data placeholder
# X = tf.placeholder(tf.float32, [None, nTimesteps, nInputs])

@tf.function
def functionX(X):


print("X: %d" % (X))
