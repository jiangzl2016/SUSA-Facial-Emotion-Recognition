import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import pickle
from Utils import load_pd_data
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam

def build_cnn_model():
    img_size = 96
    img_size_flat = img_size * img_size
    img_shape = (img_size, img_size, 1)
    #num_channels = 1
    num_classes = 9
    # Start construction of the Keras.
    model = Sequential()
    model.add(InputLayer(input_shape=(img_size_flat,)))
    model.add(Reshape(img_shape))

    #model.add(Dropout(0.5, input_shape=(48, 48, 1)))
    model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                     activation='relu', name='layer_conv1'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(kernel_size=5, strides=1, filters=64, padding='same',
                     activation='relu', name='layer_conv3'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
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
    model.fit(x = X, y = Y, epochs = epoch, batch_size = batch_size)
    return model

def evaluate(model, X, Y):
    result = model.evaluate(X, Y)
    for name, value in zip(model.metrics_names, result):
        print(name, value)
    return model

def predict(model, X_):
    y_pred = model.predict(x = X_)
    cls_pred = np.argmax(y_pred, axis = 1)
    return y_pred, cls_pred

if __name__ == '__main__':
    print('Load in Data ...')
    #train_X, train_Y , _, _ = load_pkl_data('./imagelist.pkl')
    train_X, train_Y, _, _ = load_pd_data('../data/pixel_kaggle.pd')
    epochs = 10
    batch_size = 128
    model = build_cnn_model()
    optimizer = Adam(lr = 1e-3)
    model = compile_model(model, optimizer)
    model = train(model, train_X, train_Y, epochs, batch_size)
    # evaluate on training set itself
    model = evaluate(model, train_X, train_Y)
    # predict also on training set itself
    y_pred, cls_pred = predict(model, train_X[0:9])
    print(cls_pred)






