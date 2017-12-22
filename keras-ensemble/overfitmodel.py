import matplotlib
import pandas as pd
import numpy as np
import keras
import h5py
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout,Flatten
from keras.optimizers import SGD
from keras.activations import relu, tanh, elu
from keras.backend import clear_session
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--number', type=int, default=None)
args = parser.parse_args()  

train_data_x = pd.read_pickle('normalized_fer2013.pkl')
train_data_y = pd.read_pickle('normalized_fer2013_labels.pkl').astype(int)
test_data_x = pd.read_pickle('normalized_test_fer2013.pkl')
test_data_y = pd.read_pickle('normalized_test_fer2013_labels.pkl').astype(int)

train_data_x = train_data_x.as_matrix().reshape((-1,48,48,1))
test_data_x = test_data_x.as_matrix().reshape((-1,48,48,1))

emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
positive_emotes = [3, 5]
neutral = [6]
negative_emotes = [0, 1, 2, 4]

def lump_labels(label):
    if label in negative_emotes + neutral:
        return 0
    elif label in positive_emotes:
        return 1
    else:
        return 2
    

test_data_y2 = test_data_y.apply(lump_labels)

train_data_y2 = train_data_y.apply(lump_labels)

train_data_y = train_data_y.as_matrix()
test_data_y = test_data_y.as_matrix()

train_data_y2 = train_data_y2.as_matrix()
test_data_y2 = test_data_y2.as_matrix()

train_data_y = keras.utils.to_categorical(train_data_y, num_classes=7)
test_data_y = keras.utils.to_categorical(test_data_y, num_classes=7)

train_data_y2 = keras.utils.to_categorical(train_data_y2, num_classes=2)
test_data_y2 = keras.utils.to_categorical(test_data_y2, num_classes=2)

clear_session()
model = keras.models.Sequential()

EPOCHS = 1000
PATIENCE = 20
PERIOD = 100
FILEPATH = "model_{}/".format(args.number) + "weights.epoch-{epoch:02d}-val_loss-{val_loss:.2f}-train_loss-{loss:.2f}.hdf5"
CSV_FILENAME = "model_{}/train.log".format(args.number)

if args.number % 3 == 0:
	active = 'elu'
else:
	active = 'relu'

if args.number % 2 == 0:
	model.add(Dropout(0.5,input_shape=(48,48,1)))
	model.add(Conv2D(8, (5,5), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(32, (5,5), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(64, (5,5), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(64, (5,5), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(32, (5,5), strides=(1,1),activation=active,padding='valid'))

	model.add(Flatten())
	model.add(Dense(128,activation='elu'))
	model.add(Dense(2,activation='softmax'))

else:
	model.add(Dropout(0.5,input_shape=(48,48,1)))
	model.add(Conv2D(16, (20,20), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(32, (5,5), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(32, (5,5), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(32, (5,5), strides=(1,1),activation=active,padding='valid'))
	model.add(Conv2D(32, (5,5), strides=(1,1),activation=active,padding='valid'))

	model.add(Flatten())
	if args.number % 5 == 0:
		model.add(Dense(128,activation='elu'))
		model.add(Dense(32,activation='elu'))
	model.add(Dense(2,activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                            patience=PATIENCE, verbose=0, mode='auto')
checkpoint = keras.callbacks.ModelCheckpoint(FILEPATH, monitor='val_loss', 
                    verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=PERIOD)
csv_logger = keras.callbacks.CSVLogger(CSV_FILENAME, separator=',', append=False)

model.fit(train_data_x, train_data_y2, validation_data=(test_data_x,test_data_y2), 
          epochs=EPOCHS, batch_size=32, callbacks=[early_stopping, checkpoint, csv_logger])

model.save("model_{}/final_model.h5py".format(args.number))