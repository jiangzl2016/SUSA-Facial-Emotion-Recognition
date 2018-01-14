# Facial-Emotion-Recognition -SUSA Fall 2017

This repository contains the work completed to classify facial emotions of a dataset of images provided by a Kaggle competition released in 2013. It provides different models that we have attempted, written in Keras and Tensorflow. Our final model's validation accuracy acheived on this dataset was 65% using an ensemble model of max-pooling convolutional neural networks, along with ResNet-like architectures. Currently we are still updating and trying more models.

## Data

This whole dataset is not provided due to size limits.
But a small portion of the data is provided in data folder, only for testing purposes. 

Training data provided as "PublicUse" in the (Kaggle Competition data)[https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data]

Data used to evaluate the model after validation & hyper parameter tuning. Provided as "PrivateUse".


## Code

The code provided is how we trained our ensemble models with a 4-fold cross validation on 28,000 training images (48x48x3).

***splitscript.sh***

This is a bash script to be run to train our models on all 4 cross-validation folds of the provided data. 

***model_train.py***

Script to train either a Max-pooling CNN or ResNet-like architecture depending on flags passed. Saves models to exp\_{fold\_num}/.



