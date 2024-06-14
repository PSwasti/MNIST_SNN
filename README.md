# MNIST_SNN
# A 728,32,16,10 SNN
# MNIST (Modified National Institute of Standards and Technology) database is a large database of handwritten digits used for training various image processing systems.
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
# A Sequential model is used where each layer has exactly one input tensor (MD array with an arbitrary number of dimensions) and one output tensor.
from tensorflow.keras.layers import Input, Dense, Flatten
# Flatten(input_shape=(28, 28)) gave an error so import Input also.
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# Loading the data from MNIST Dataset (already there)
x_train, x_test = x_train / 255.0, x_test / 255.0
# The computation of high numeric values is very complex, so we normalize the values to range from 0 to 1.
model = Sequential([Input(shape=(28, 28)), Flatten(), Dense(32, activation='relu'), Dense(10, activation='softmax')])
# Each neuron in the Dense layer is connected to every neuron in the preceding layer.
# Relu replaces any negative values in the output with zero, Softmax transforms the output of a neural network into a probability distribution.
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# Adam is an Optimization algo, sparse_categorical_crossentropy is a loss function, metric is for performance and accuracy gives proportion of correct o/p (predictions) over total o/p
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
# 5 passes, and 20% of the set for validation
