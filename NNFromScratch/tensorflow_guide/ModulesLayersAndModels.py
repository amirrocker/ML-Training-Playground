# https://www.tensorflow.org/guide/intro_to_modules

'''
Any tensorflow model is abstractly:
- a function that computes something (weights) on tensors ( forward pass )
- some variables that can be updated in response to training ( weights? )
'''

import tensorflow as tf


class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.trainable_variable = tf.Variable(3.0, trainable=True, name="do train me")
        self.non_trainable_variable = tf.Variable(6.0, trainable=False, name="do not train me")

    def __call__(self, x):
        return self.trainable_variable * x + self.non_trainable_variable


def runIt():
    simple_module = SimpleModule("simpleModule")
    result = simple_module(tf.constant(10.0))
    print("result: {}", result)


runIt()

'''
Modules:
let's create a two-layer Dense model, consisting of modules:
'''


class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name)
        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name="w")
        self.b = tf.Variable(tf.zeros([out_features]), name="b")

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)  # send y into the activation function


class DenseSequential(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)

        self.dense_1 = Dense(in_features=3, out_features=3, name="dense_1")
        self.dense_2 = Dense(in_features=3, out_features=2, name="dense_2")

    def __call__(self, x):
        y = self.dense_1(x)
        return self.dense_2(y)


# Congrats, its a model. you can be so proud.
my_model = DenseSequential("DenseSequential")

predicted = my_model(tf.constant([[2.0, 3.0, 4.0]]))
print("predicted results: ", predicted)

print("submodules: ", my_model.submodules)

for var in my_model.variables:
    print(var, "\n")

'''
Creating Variables lazily, kinda
'''


class FlexibleDenseModule(tf.Module):
    def __init__(self, out_features, name=None):
        super().__init__(name=name)
        self.is_built = False
        self.out_features = out_features

    def __call__(self, x):
        # create when required
        if not self.is_built:
            self.w = tf.Variable(tf.random.normal([x.shape[-1]], self.out_features, name="w"))
            self.b = tf.Variable(tf.zeros([self.out_features]), name="b")
            self.is_built = True
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)
