#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 08:59:43 2017

@author: shubham
"""

import numpy as np
import logging
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plot
import time


def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  
    return [x_train, y_train, x_test, y_test]

def build_model(layers):
    activationFunction = "tanh"
    dropout = 0.2
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(dropout))
    logging.info("Activation function: " + activationFunction + " , Dropout: " + str(dropout))
    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation(activationFunction))
    model.compile(loss="mse", optimizer="rmsprop")
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def plot_results(predicted_data, true_data):
    fig = plot.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plot.plot(predicted_data, label='Prediction')
    plot.legend()
    plot.show()
   
epochs = 10
batchSize = 512
validationSplit = 0.05
logging.getLogger().setLevel(logging.INFO)
startTime = time.time()
X_train, y_train, X_test, y_test = load_data('sin.csv', 50, True)
logging.info("Data Loaded")

model = build_model([1, 50, 100, 1])
logging.info("Bulding LSTM with 2 hidden layers. 1st with 50 cells and 2nd with 100 cells.")

model.fit(
    X_train,
    y_train,
    batch_size=batchSize,
    nb_epoch=epochs,
    validation_split=validationSplit)

logging.info("Batch size: " + str(batchSize) + ", number of epochs: " + str(epochs))
predicted = predict_point_by_point(model, X_test)
plot_results(predicted, y_test)
endTime = time.time()
logging.info("Total time taken: " + str(endTime - startTime) + "s")
