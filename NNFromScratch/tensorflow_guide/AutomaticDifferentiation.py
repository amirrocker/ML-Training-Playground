import tensorflow as tf

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2  # y = x pow 2
    dy_dx = tape.gradient(y, x)
    dy_dx.numpy()
    print(dy_dx)

# this may still take some time to discover the concepts of
# AutoDiff
# see https://en.wikipedia.org/wiki/Automatic_differentiation
# and https://en.wikipedia.org/wiki/Backpropagation
