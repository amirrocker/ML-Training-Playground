import logging
# helper libraries
import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

np.set_printoptions(suppress=True)

# disable loading bar
tfds.disable_progress_bar()

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)

'''
# variables section 
'''
CONV_INPUT_NUMBER_OF_FILTERS = 32
CONV_HIDDEN_NUMBER_OF_FILTERS = 64
DENSE_HIDDEN_NEURONS = 128
DENSE_OUTPUT_NEURONS = 10

KERNEL_SHAPE_CONV = (3, 3)
KERNEL_SHAPE_MAX_POOLING = (2, 2)
INPUT_SHAPE = (28, 28, 1)
STRIDES = 2

'''
# Function section 
'''


# preprocess and prepare data
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


def defineModel():
    model = tf.keras.Sequential([
        # CONV_INPUT_NEURONS = 32 ? 32 filters ? check that again.
        tf.keras.layers.Conv2D(CONV_INPUT_NUMBER_OF_FILTERS, KERNEL_SHAPE_CONV, padding="same", activation=tf.nn.relu,
                               input_shape=INPUT_SHAPE),
        tf.keras.layers.MaxPooling2D(KERNEL_SHAPE_MAX_POOLING, strides=STRIDES),
        tf.keras.layers.Conv2D(CONV_HIDDEN_NUMBER_OF_FILTERS, KERNEL_SHAPE_CONV, padding="same", activation=tf.nn.relu,
                               input_shape=INPUT_SHAPE),
        tf.keras.layers.MaxPooling2D(KERNEL_SHAPE_MAX_POOLING, strides=STRIDES),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(DENSE_HIDDEN_NEURONS, activation=tf.nn.relu),
        tf.keras.layers.Dense(DENSE_OUTPUT_NEURONS, activation=tf.nn.softmax)
    ])
    return model


def plotImage(image):
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()


'''
Plot a number of images on the same plot
'''


def plotImages(dataset):
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(train_dataset.take(25)):
        image = image.numpy().reshape((28, 28))
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])
    plt.show()


def plot_result_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    print("predicted_label: {}".format(class_names[predicted_label]))
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_result_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# end function section

# using tf dataset api
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

print("dataset containing 28x28 arrays of Ints (0-255) loaded")

# each image (array) is mapped to a label
class_names = metadata.features['label'].names
print(class_names)
print(len(metadata.features))
# or alternate
print("classnames: {}".format(class_names))

# now we can explore the data
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("num_train_examples: {}".format(num_train_examples))
print("num_test_examples: {}".format(num_test_examples))

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

print("train_dataset after normalize: {}".format(train_dataset))
print("test_dataset after normalize: {}".format(test_dataset))

''' Plot a single image
for image, label in test_dataset.take(5):
    break
image = image.numpy().reshape((28, 28))
plotImage(image)
'''

# plotImages(train_dataset)

# cache to speed up training -> is set inside the training section
# train_dataset = train_dataset.cache()
# test_dataset = test_dataset.cache()

'''
## this was all preprocessing. Most of the time it is not this easy.
## See Data Preprocessing for more.

## creating a model in tensor flow consists of
- define model 
- compile model
- train model
- evaluate model
- query model
'''

BATCH_SIZE = 16
EPOCHS = 5

HIDDEN_LAYER_NEURONS = 512

# define the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
model = defineModel()

# compile the model
# loss => how far is the model away from the label data.
# optimizer => the algorithm to minimize loss.
# metrics =>
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# train the model
# batch => tells the model to "feed" the model 32 images per batch to update the model variables
# epoch => the number of training iterations over the training_dataset
#       => 5 x 60000 features to train = 300000 iterations


train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

# setup tensorboard
tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

# now we start training
model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE),
          callbacks=[tb_callback])

# Evaluate model
# loss =>
# accuracy =>
print("Evaluation: ")
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples / BATCH_SIZE))
print("test_loss: {}".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))

# query model
# use the model to make predictions (classification) on images:

for test_images, test_labels in test_dataset.take(1):
    i = 15
    # for i, (image, label) in enumerate(train_dataset.take(25)):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_result_image(i, predictions, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_result_value_array(i, predictions, test_labels)

    # lets plot multiple images with predictions
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for index in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * index + 1)
        plot_result_image(index, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * index + 2)
        plot_result_value_array(index, predictions, test_labels)

    plt.show()

    # grab single image from test_dataset
    img_test_dataset = test_images[0]
    print(img_test_dataset.shape)
    img_test_dataset = np.array([img_test_dataset])
    print(img_test_dataset.shape)

    prediction_single = model.predict(img_test_dataset)
    print(prediction_single)

    plot_result_value_array(0, prediction_single, test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)

    np.argmax(prediction_single[0])
    plt.show()
