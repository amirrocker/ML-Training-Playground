# https://www.youtube.com/watch?v=BmZJDptVYB0&t=17s

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(tf.__version__)
# tf2 has no logging package anymore.
# https://stackoverflow.com/questions/55318626/module-tensorflow-has-no-attribute-logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# some play data
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38])
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100])

for i, c in enumerate(celsius_q):
    print("celsius: %s -> fahrenheit: %s" % (c, fahrenheit_a[i]))

# this is a very simple use case - learn celsius and "predict" fahrenheit
# a single neuron in a one element Dense layer

l0 = tf.keras.layers.Dense(input_shape=[1], units=4)
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])

# now the model needs to be compiled with a loss function and an Optimizer function
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# now train the model
history = model.fit(celsius_q, fahrenheit_a, epochs=1000, verbose=True)
print("Finished training model...")

plt.xlabel("Epoch Number")
plt.ylabel("Loss Value")
plt.plot(history.history['loss'])
plt.show()

celsius_v = np.array([38, 22, 15, 0, -10, 8, -40])
# result = model.predict(celsius_v)
result = model.predict([100.0])
result2 = model.predict([20.0])
print("Degrees celsius: %s --> predicted Fahrenheit: %s" % (100.0, result))
print("Degrees celsius: %s --> predicted Fahrenheit: %s" % (20.0, result2))

# print out the layer variables - weight values:
print("These are the layer1 variables: {}".format(l0.get_weights()))
print("These are the layer2 variables: {}".format(l1.get_weights()))
print("These are the layer3 variables: {}".format(l2.get_weights()))
