import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import pickle
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten

# Equal to pixel size, change to 48
img_size = 48

img_size_flat = img_size * img_size

img_shape = (img_size, img_size)

img_shape_full = (img_size, img_size, 1)

num_channels = 1

num_classes = 6


def build_cnn_model():
    # Start construction of the Keras Sequential model.
    model = Sequential()

    model.add(InputLayer(input_shape=(img_size_flat,)))

    model.add(Reshape(img_shape_full))

    # First convolutional layer with ReLU-activation and max-pooling.
    model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                     activation='relu', name='layer_conv1'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Second convolutional layer with ReLU-activation and max-pooling.
    model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                     activation='relu', name='layer_conv2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    model.add(Flatten())

    # First fully-connected / dense layer with ReLU-activation.
    model.add(Dense(128, activation='relu'))

    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(num_classes, activation='softmax'))

    return model


def compile_model(model, optimizer, loss='categorical_crossentropy', metrics=['accuracy']):
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model

def save_data():
    return

def train(model, X, Y, epoch, batch_size):
    model.train()
    return model 

if __name__ == '__main__':
    print('Train models.')
