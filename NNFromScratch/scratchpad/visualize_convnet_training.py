# https://keras.io/examples/vision/visualizing_what_convnets_learn/

# Setup
import numpy as np
import tensorflow as tf
from tensorflow import keras

# dimens input image
img_width, img_height = 180, 180

# target layer to visualize the filters from
# see 'model.summary()' for list layer names
layer_name = "conv3_block4_out"

# Build feature extraction model
# Build a ResNet50V2 model loaded with pre-trained ImageNet weights
model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)

layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)


# setup gradient ascent process
def computeLoss(inputImage, filterIndex):
    activation = feature_extractor(inputImage)
    # avoid border artefacts
    filterActivation = activation[:, 2:-2, 2:-2, filterIndex]
    return tf.reduce_mean(filterActivation)


@tf.function
def gradient_ascent_step(img, filterIndex, learningRate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = computeLoss(img, filterIndex)
        # print("loss: {}".format(loss))
    # compute gradients
    gradients = tape.gradient(target=loss, sources=img)
    # print("gradients: {}".format(gradients))

    # Normalize gradients using l2_normalization
    gradients = tf.math.l2_normalize(gradients)

    img += learningRate * gradients
    return loss, img


# setup end-to-end visualization:
'''
# process:
# start from random neutral image
# repeatedly apply gradient ascent step function 
# convert resulting input image back to displayable form 
# by 
# - normalizing it  
# - center cropping
# - clamp to [0, 255] range 
'''

import matplotlib.pyplot as plt


def plotImage(image):
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def initializeImage():
    # we start from neutral image
    # tf.random.uniform : Outputs random values from a uniform distribution.
    neutral_img = tf.random.uniform((1, img_width, img_height, 3))
    # as most models ResNet50V2 expects input from [-1, 1]
    # scale the random inputs to [-0.125, +0.125]
    return (neutral_img - 0.5) * 0.25


def visualizeFilter(filterIndex):
    # run gradient ascent for 20 steps
    iterations = 30
    learningRate = 10.0
    img = initializeImage()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filterIndex, learningRate)
        # print("loss: {}".format(loss))

    # Decode the resulting input image
    img = deprocessImage(img[0].numpy())
    return loss, img


def deprocessImage(img):
    # normalize array : center on 0., ensure a variance of 0.15
    img -= img.mean()
    img /= img.std() * 1e-5
    img *= 0.15

    # center crop
    img = img[25:-25, 25:-25, :]

    # clip to [0,1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


# from IPython.display import Image, display

# loss, img = visualizeFilter(0)
# keras.preprocessing.image.save_img("0.png", img)

# display(Image("0.png"))


def displayStitchedFilters():
    # compute image inputs
    allImages = []
    numberFilters = 64
    for filterIndex in range(numberFilters):
        # print("processing filter %d" % (filterIndex,))
        loss, img = visualizeFilter(filterIndex)
        # print("loss: %d" % (loss))
        allImages.append(img)

    # create a black background image with enough space
    # to tile the filter images 8x8 with 120x120 px and 5px margin
    margin = 5
    n = 8
    croppedWidth = img_width - 25 * 2
    croppedHeight = img_height - 25 * 2

    width = n * croppedWidth + (n - 1) * margin
    height = n * croppedHeight + (n - 1) * margin
    # simple black image array
    stitchedFilters = np.zeros((width, height, 3))

    # debug save the template image
    keras.preprocessing.image.save_img("stitched.png", stitchedFilters)

    display(Image("stitched.png"))

    # now fill the template with saved filters
    '''
    for i in range(n):
        for j in range(n):
            img = allImages[i * n + j]
            stitchedFilters[
                (croppedWidth + margin) * i : (croppedWidth + margin) * i + croppedWidth,
                (croppedHeight + margin) * j : (croppedHeight + margin) * j + croppedHeight,
                :,
            ] = img

    keras.preprocessing.image.save_img("stitchedFilters.png", stitchedFilters)
    # display(Image("stitchedFilters.png"))
    plotImage(Image("stitchedFilters.png"))
    '''


# displayStitchedFilters()

# https://www.delftstack.com/howto/python/python-display-image/

# only method to work so far
# from PIL import Image
# im = Image.open("stitchedFilters.png")
# im.show()

# that also works
import matplotlib.image as mpimg

img = mpimg.imread('stitchedFilters.png')
imgplot = plt.imshow(img)
plt.show()
