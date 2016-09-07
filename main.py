# Technique used: softmax regression

# This is the data that will be used for testing, training and validating.
# Each data point has two parts: An image of a handwritten digit and a label.
# Each image is 28x28 pixels, that means it can be divided in a vector of
# 784 numbers.

# The resulting data is a TENSOR with a [n, m] shape, where
# n is the numbers of images given and m the pixels in each image (in this
# case, 784). Each tensor entry describes the intensity of the pixel
# (between 1 and 0). The labels are each a vector of size 10 (since
# we're trying to find digits) - also called one-hot vectors. The first
# position represent a 0, the second a 1 and so on. All the values are 0,
# besides the digit used. Example: 5 is [0, 0, 0, 0, 0, 1, 0, 0, 0, 0].
from input_data import mnist
import tensorflow as tf

# Implementing the regression

# x is a symbolic variable (placeholder), allowing us to input any number
# of MNIST images, each flattened into a 784-dimensional vector.
# It is a 2d tensor of floats, with a [None, 784] shape
# (None means that the dimension can have any length)
x = tf.placeholder(tf.float32, [None, 784])

# W is the weight and b is the bias.
# For machine learning, the model parameters are usually variables.
# They are odifiable tensors full of zeros
# W has a shape of [784, 10] because we want to multiply
# the 784-dimensional image vectors by it to produce
# 10-dimensional vectors of evidence for the difference classes.
# b has a shape of [10] so we can add it to the output.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
