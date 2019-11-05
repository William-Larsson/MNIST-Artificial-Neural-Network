import numpy as np
from random import random


class Network:
    """
    A class representing a neuron in the neural network.
    """

    def __init__(self, label, size):
        """
        Construct a neuron with a number it should specialize on.
        Also create an array of random weights for between the neuron and following node.
        :param label: The number the specialize on.
        :param size: The number of pixels in an image.
        """
        self.label = label
        self.weights = np.array([0.1 * random() - 0.05 for _ in range(size)])

    def calculate_error(self, label, a):
        """
        Calculate the error between the desired output y and the activation a of the current neuron.
        :param label: The label representing the real number of the current image.
        :param a: The activation function output.
        :return: The error value.
        """
        y = 1 if self.label == label else -1
        return y - a

    def calculate_new_weight(self, error, img, learning_rate):
        """
        Calculates the new weight between the neuron and the output node.
        :param error: The error between the desired output y and the activation function
        :param img: An image array with all its pixels
        :param learning_rate: Balance how much the weight is updated
        :return: Updated weight between the neuron and the following node.
        """
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + img[i] * error * learning_rate

    def activation_function(self, dot):
        """
        Maps the neurons inputs to corresponding output.
        :param dot: The dot product of the neurons weights and the pixel values.
        :return: A Value representing how active the neuron should be for this particular input.
        """
        return np.tanh(dot)

    def dot_product(self, pixels):
        """
        Calculate the dot product between all pixels and weights for an image.
        :param pixels: Array of pixels in an image
        :return: The dot product sum
        """
        return np.dot(self.weights, pixels)
