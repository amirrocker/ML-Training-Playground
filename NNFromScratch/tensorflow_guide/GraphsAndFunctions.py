# based on source
# https://www.tensorflow.org/guide/intro_to_graphs

# running tensorboard
# https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#vii_Creating_Callback_Object

import tensorflow as tf


# def a regular python func
def a_regular_function(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


func_to_use_a_tf_graph = tf.function(a_regular_function)

# create some data, pref. tensors
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

orig_function_value = a_regular_function(x1, y1, b1).numpy()
print(orig_function_value)

tf_function_value = func_to_use_a_tf_graph(x1, y1, b1).numpy()
print(tf_function_value)

assert orig_function_value == tf_function_value


# above code could also be using a outside Tf.Function like so:
def innerFunc(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


@tf.function
def outerFunc(x):
    y = tf.constant([[2.0], [3.0]])
    b = tf.constant(4.0)
    return innerFunc(x, y, b)


result = outerFunc(tf.constant([[1.0, 2.0]])).numpy()
print(result)


# converting python funcs to graphs:

def simple_relu(x):
    if tf.greater(x, 0):
        return x
    else:
        return 0


tf_simple_relu = tf.function(simple_relu)
print("if branch: ", tf_simple_relu(4).numpy())
print("if branch: ", tf_simple_relu(tf.constant(3)).numpy())
print("else branch: ", tf_simple_relu(tf.constant(-3)).numpy())


# Output of graph-generating AutoGraph:
# uncomment to show created graph code
# print(tf.autograph.to_code(simple_relu))
# and the graph itself:
# uncomment to show json representation of graph
# print(tf_simple_relu.get_concrete_function(tf.constant(1)).graph.as_graph_def())
# or use tensorboard to show the graph
# http://localhost:6006/#graphs&run=run2%5Ctrain


# polymorphism using parametrised graph functions:

@tf.function
def custom_relu(x):
    return tf.maximum(0., x)


print(custom_relu(tf.constant([[1., 5.], [3., 8.]])))
print(custom_relu(tf.constant([[-2., -3.]])))
print(custom_relu(tf.constant([[2., -4.]])))

# does NOT create a new graph since the signature matches, in other words the shape is used to support polymorphism
print(custom_relu(tf.constant([[3.334, 0.7], [3., 8.]])))
print(custom_relu(tf.constant([[-5., 1.3]])))
print(custom_relu(tf.constant([[0.345, -16.]])))

# show all polymorphic signatures in tf.function
# print(custom_relu.pretty_printed_concrete_signatures())

tf.config.run_functions_eagerly(False)


# graph execution vs. eager execution
# Note: If you would like to print values in both eager and graph execution, use tf.print instead.
@tf.function
def get_MSE(y_label, y_pred):
    # print("calculate MSE")
    tf.print("calculate MSE using tf.print")
    sq_diff = tf.pow(y_label - y_pred, 2)
    return tf.reduce_mean(sq_diff)


y_label = tf.random.uniform([5], maxval=10, dtype=tf.int32)
y_pred = tf.random.uniform([5], maxval=10, dtype=tf.int32)
print("y_label: ", y_label)
print("y_pred: ", y_pred)

error = get_MSE(y_label, y_pred)
error = get_MSE(y_label, y_pred)
error = get_MSE(y_label, y_pred)

'''
# Error handling when using tf.function and non-strict execution 
Non-strict execution

Graph execution only executes the operations necessary to produce the observable effects, which includes:

- The return value of the function
- Documented well-known side-effects such as:
    - Input/output operations, like tf.print
    - Debugging operations, such as the assert functions in tf.debugging
    - Mutations of tf.Variable

This behavior is usually known as "Non-strict execution", and differs from eager execution, 
which steps through all of the program operations, needed or not.

In particular, runtime error checking does not count as an observable effect. 
If an operation is skipped because it is unnecessary, it cannot raise any runtime errors.

In the following example, the "unnecessary" operation tf.gather is skipped during graph execution, 
so the runtime error InvalidArgumentError is not raised as it would be in eager execution. 
Do not rely on an error being raised while executing a graph.
check Non-strict execution here: 
https://www.tensorflow.org/guide/intro_to_graphs
'''

# Best Practices
# https://www.tensorflow.org/guide/intro_to_graphs#tffunction_best_practices

TBD!
