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