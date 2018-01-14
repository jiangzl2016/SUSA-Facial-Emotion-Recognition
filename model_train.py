"""
Creates conv or resnet architecture based upon whether or not arg number is even or odd

Class weighting for model is train
"""

import pandas as pd
import numpy as np
import keras
import h5py
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout,Flatten
from keras.optimizers import SGD, Adam
from keras.activations import relu, tanh, elu
from keras.backend import clear_session
from keras.models import load_model


from kerasresnet.resnet import ResnetBuilder as resbuilder

def normalize(data, mean, range=255):
	return (data - mean)/range

parser = argparse.ArgumentParser()
parser.add_argument('--number', type=int, default=None)
parser.add_argument('--splitnum', type=int, default=None)
args = parser.parse_args()

# Hyper parameters for training
# -----------------------------
EPOCHS = 40
PATIENCE = 10
LEARNING_RATE = 1e-4
CSV_FILENAME = "exp_{}/model_{}/".format(args.splitnum,args.number)
FINAL_SAVENAME = "final_model.h5py"
active = 'relu'

# Data
# ------------------------------
datanpz = np.load("train.npz")
dataX, dataY = datanpz['X'],datanpz['Y']

# Split Data
# ------------------------------
"""Explecting split num e [0,4]"""
assert args.splitnum < 5, "splitnum has to be in [0, 4]"
TRAIN_SPLIT = 0.8 # this is explicit declaration, but hard-coded into implementation
lower_test_idx = int(1/5*args.splitnum*len(dataX))
upper_test_idx = int(1/5*(args.splitnum+1)*len(dataX))

print("lower test:",lower_test_idx,"\tupper test:",upper_test_idx)

test_data_x  = dataX[lower_test_idx:upper_test_idx]
test_data_y  = dataY[lower_test_idx:upper_test_idx]

train_data_x = np.concatenate((dataX[:lower_test_idx],dataX[upper_test_idx:]))
train_data_y = np.concatenate((dataY[:lower_test_idx],dataY[upper_test_idx:]))

# Normalize Data
# ------------------------------
"""Find mean of train data, normalize train+test
To recover mean, just recalculate from dir name"""
mean_param = np.mean(np.mean(train_data_x))
train_data_x = normalize(train_data_x,mean_param)
test_data_x = normalize(test_data_x,mean_param)

emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

train_data_x = np.reshape(train_data_x, (-1, 48, 48, 1))
test_data_x = np.reshape(test_data_x, (-1, 48, 48, 1))

train_data_y = keras.utils.to_categorical(train_data_y, num_classes=7)
test_data_y = keras.utils.to_categorical(test_data_y, num_classes=7)

model = keras.models.Sequential()

# CREATE MODEL
# ------------
# Creates model based on 2-parity of args.number
IN_SHAPE = (48, 48, 1)
if args.number % 2 == 0:
	# our vanilla convolutional approach
	CSV_FILENAME += 'convolutional.log'
	model.add(Conv2D(32, (5,5), strides=(1,1),activation=active,padding='valid', input_shape=IN_SHAPE))
	model.add(Conv2D(64, (3,3), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(64, (3,3), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(128, (3,3), strides=(1,1),activation=active,padding='valid'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'))
	model.add(Conv2D(128, (3,3), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(64, (2,2), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(64, (2,2), strides=(1,1),activation=active,padding='valid'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'))
	model.add(Conv2D(64, (1,1), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(64, (1,1), strides=(1,1),activation=active,padding='valid'))

	model.add(Flatten())
	model.add(Dense(128,activation='tanh'))
	model.add(Dense(32,activation='tanh'))
	model.add(Dense(7,activation='softmax'))
	
elif args.number % 2 == 1:
	# 18 layer resnet
	CSV_FILENAME += 'resnetty.log'
	model = resbuilder.build_resnet_18((1, 48, 48), 7)

# SET UP CLASS WEIGHTING
# -----------------------
"""Use majority class to weight the other classes"""
values = []
for i in range(7):
	values.append(np.mean(train_data_y[:,i] == 1))
print(values)
majority = np.max(values)
values = majority/values
print(values)
values = values+np.random.randn((7))/5
print(values)
# values = [7 if i == (args.number % 7) else 1 for i in range(7)]
cweights = {i:values[i] for i in range(len(values))}

opt = Adam(lr=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# early_stopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, 
#                     patience=PATIENCE, verbose=0, mode='auto')
# checkpoint = keras.callbacks.ModelCheckpoint(FILEPATH, monitor='val_loss', 
            # verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=PERIOD)
csv_logger = keras.callbacks.CSVLogger(CSV_FILENAME, separator=',', append=False)

try:
	model.fit(train_data_x, train_data_y, validation_data=(test_data_x,test_data_y), 
      epochs=EPOCHS, batch_size=32, callbacks=[csv_logger])
except KeyboardInterrupt:
	print("hi we saving you're model to exp_{}/model_{}/final_model.h5py".format(args.splitnum,args.number))

model.save(CSV_FILENAME+FINAL_SAVENAME)