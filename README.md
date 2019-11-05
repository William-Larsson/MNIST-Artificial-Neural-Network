## MNIST ANN
This an implementation of a neural network that is supposted to classify images from a subset of the MNIST dataset. This is done with Python 3.7 in combination with and NumPy in order to improve performance. 

## What I learned
* The fundamentals of supervised learning.
* The advantages and drawbacks of ANN:s.
* Creating an ANN model from scratch and getting familiar with all the components needed for it to work.
* Increasing my knowledge of Python and object-oriented programing .
* Working with NumPy for the first time and exploring its capabilites.

## How to run the model
The model needs four parameters in order to work. This could be the provided .txt file which represent a subset of the full MNIST dataset.

System could therefore be run using the following shell command:
> python3 digits.py training-images.txt training-labels.txt validation-images.txt validation-labels.txt

The system should in theory work with the full MNIST dataset as well, but this has not been tested.
