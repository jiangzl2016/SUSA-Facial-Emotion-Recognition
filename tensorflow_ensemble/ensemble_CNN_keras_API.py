'''
The model created ensemble of CNNs through averaging prediction results.
Author: Zhongling
'''

import numpy as np
import time
import os
from Utils import load_pd_data
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from keras.models import load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_accuracy
from Utils import next_batch

'''
The class create CNN model objects. Include build, compile, train, save and evaluate.
Users could also define other model objects and modify ensemble function to train and
save other models.
'''
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

    def train(self, model_name = None, epochs = 10, batch_size = 128):
        if self.model is not None:
            self.model.fit(x=self.X, y=self.Y, epochs= epochs, batch_size= batch_size)

    def save(self, filepath):
        if self.model is not None:
            self.model.save(filepath + ".h5py")
            model_json = self.model.to_json()
            with open(filepath + ".json", "w") as json_file:
                json_file.write(model_json)
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

'''
The function trains and saves n networks, currently all CNNs. This function could be extended
into other models as well.
'''
def CNN_ensemble(image_size, num_networks, num_classes, train_X, train_Y,  epochs = 5, batch_size = 128):
    num_channels= 1
    list_of_model = []
    for i in range(num_networks):
       # for j in range(epochs):
        #X_batch, Y_batch = next_batch(train_X, train_Y, batch_size)
        model = CNN_model(image_size, num_channels, num_classes)
        model.load_train_data(train_X, train_Y)  #######################
        model.build_model()
        optimizer = Adam(lr=1e-3)
        model.compile_model(optimizer)
        model.train(epochs= epochs, batch_size=batch_size)
        #model.evaluate_train()
        model_name = "CNN_model_{0}".format(i)
        model.save("./model/" + model_name)    #####save the model here
        if model is None:
            print ("missing model_{0}".format(i))

'''
The function loads n models and make predictiona using average of n networks.
'''
def ensemble_predict(number_of_models, num_classes, test_set, filepath= '../model/', pattern= ".h5py"):
    all_files = os.listdir(filepath)
    if len(all_files) == 0:
        print('No Models in the directory. Please first train and save models.')
    else:
        predictions = np.zeros((test_set.shape[0], num_classes))
        for (i,filename) in enumerate(all_files):
            count = 0
            if pattern in filename:
                model = load_model(filepath + filename)
                preds = model.predict(test_set)
                predictions += preds
                #cls_preds = np.argmax(preds, axis=1)
        predictions = predictions / num_networks
        return predictions


if __name__ == '__main__':
    print('Load in Data ...')
    train_X, train_Y, _, _ = load_pd_data('../data/pixel_nocomplex_train.pd')
    test_X, test_Y, _, _ = load_pd_data('../data/pixel_nocomplex_test.pd')
    epochs = 10
    batch_size = 128
    num_networks = 3
    num_classes = 8
    image_size = 96
    CNN_ensemble(image_size, num_networks, num_classes, train_X, train_Y, epochs, batch_size)
    preds = ensemble_predict(num_networks, num_classes, test_X)
    preds_cls = np.argmax(preds, axis=1)
    print("The prediction for first 100 images are: ")
    print(preds[0:99])
    # measure accuracy
    acc = np.sum(preds_cls == test_Y) / len(preds_cls)
    print("The accuracy of ensemble model is: {0}".format(acc))
