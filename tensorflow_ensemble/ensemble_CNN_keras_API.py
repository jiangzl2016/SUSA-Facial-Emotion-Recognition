import numpy as np
import time
from Utils import load_pd_data
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten

from tensorflow.python.keras.optimizers import Adam
from Utils import next_batch
class CNN_model(object):

    def __init__(self, image_size, num_channels, num_classes):
        self.model = None
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.X = None
        self.Y = None
    def load_train_data(self, X, Y):
        self.X = X
        self.Y = Y

    def build_model(self):
        img_size_flat = self.image_size * self.image_size
        img_shape = (self.image_size, self.image_size, self.num_channels)
        model = Sequential()
        model.add(InputLayer(input_shape=(img_size_flat,)))
        model.add(Reshape(img_shape))

        # model.add(Dropout(0.5, input_shape=(48, 48, 1)))
        model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                         activation='elu', name='layer_conv1'))
        model.add(MaxPooling2D(pool_size=2, strides=2))

        model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                         activation='elu', name='layer_conv2'))
        model.add(MaxPooling2D(pool_size=2, strides=2))

        model.add(Conv2D(kernel_size=5, strides=1, filters=64, padding='same',
                         activation='elu', name='layer_conv3'))
        model.add(MaxPooling2D(pool_size=2, strides=2))

        model.add(Flatten())

        model.add(Dense(128, activation='elu'))
        model.add(Dense(32, activation='elu'))
        # Last fully-connected / dense layer with softmax-activation
        # for use in classification.
        model.add(Dense(self.num_classes, activation='softmax'))

        self.model = model


    def compile_model(self, optimizer, loss_metric = 'categorical_crossentropy', metrics = ['accuracy']):
        if self.model is not None:
            #self.optimizer = optimizer
            #self.loss_metric = loss_metric
            #self.metrics = ['accuracy']
            self.model.compile(optimizer=optimizer,
                          loss= loss_metric,
                          metrics=metrics)

    def train(self, model_name = None, epochs = 10, batch_size = 128, save_option = False):
        if self.model is not None:
            self.model.fit(x=self.X, y=self.Y, epochs= epochs, batch_size= batch_size)
            if save_option is True:
                path = './models/'
                self.model.save(path + 'model_{}.h5py'.format(model_name))

    def evaluate_train(self):
        if self.model is not None:
            result = self.model.evaluate(self.X, self.Y)
            for name, value in zip(self.model.metrics_names, result):
                print(name, value)

    def evaluate_test(self, X_val, Y_val):
        if self.model is not None:
            result = self.model.evaluate(X_val, Y_val)
            for name, value in zip(self.model.metrics_names, result):
                print(name, value)

    def predict(self, X_test):
        if self.model is None:
            return None, None
        y_pred = self.model.predict(x=X_test)
        cls_pred = np.argmax(y_pred, axis=1)
        return y_pred, cls_pred


def CNN_ensemble(num_networks, train_X, train_Y, batch_size = 128):
    image_size = 96
    num_channels= 1
    num_classes = 9
    list_of_model = []
    for i in range(num_networks):
        # for j in range(epochs..)
        X_batch, Y_batch = next_batch(train_X, train_Y, batch_size)
        model = CNN_model(image_size, num_channels, num_classes)
        model.load_train_data(X_batch, Y_batch)
        model.build_model()
        optimizer = Adam(lr=1e-3)
        model.compile_model(optimizer)
        model.train(epochs=5)
        model.evaluate_train()
        if model is None:
            print ('missing model')
        else:
            y_pred, cls_pred = model.predict(train_X)
            print(cls_pred)



if __name__ == '__main__':
    print('Load in Data ...')
    train_X, train_Y, _, _ = load_pd_data('../data/pixel_kaggle.pd')
    epochs = 10
    batch_size = 128
    num_networs = 3
    CNN_ensemble(3, train_X, train_Y, batch_size)