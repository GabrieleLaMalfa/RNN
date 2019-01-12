# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 09:13:10 2018

@author: Gabriele
"""

# LSTM for Anomaly Detection

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def gaussian_pdf(x, mean, variance):

    p_x = (1 / (2 * np.pi * variance) ** .5) * np.exp(-(x - mean) ** 2 / (2 * variance))

    return p_x


def series_to_matrix(series,
                     k_shape,
                     striding=1):
    res = np.zeros(shape=(int((series.shape[0] - k_shape) / striding) + 1,
                          k_shape)
                   )
    j = 0
    for i in range(0, series.shape[0] - k_shape + 1, striding):
        res[j] = series[i:i + k_shape]
        j += 1

    return res


"""
 3 modes are possible and need do be specified in 'mode' variable:
    'train': all data is reserved to train;
    'train-test': split between train and test, according to non_train_percentage;
    'validation': data is split among train, test and validation: their percentage is chosen according to the percantge
                  of data that has not been included in train (1-non_train_percentage) and assigned to validation
                  proportionally to val_rel_percentage.
"""
def generate_batches(filename, 
                     window, 
                     mode='train-test', 
                     non_train_percentage=.7, 
                     val_rel_percentage=.5,
                     normalize=False,
                     time_difference=False):

    data = pd.read_csv(filename, delimiter=',', header=0)
    data = (data.iloc[:, 0]).values

    # normalize dataset (max-min method)
    if normalize is True:
        
        data = (data-np.min(data))/(np.max(data)-np.min(data))
        
    # if the flag 'time-difference' is enabled, turn the dataset into the variation of each time 
    #  step with the previous value (loose the firt sample)
    if time_difference is True:
        
        data = data[:-1] - data[1:]
        
    if mode == 'train':

        y = data[window:]
        x = series_to_matrix(data, window, 1)[:-1]

        return x, y

    elif mode == 'train-test':

        train_size = int((1 - non_train_percentage) * np.ceil(len(data)))
        y_train = data[window:train_size]
        x_train = series_to_matrix(data, window, 1)[:train_size - window]
        y_test = data[train_size:]
        x_test = series_to_matrix(data, window, 1)[train_size:]

        return x_train, y_train, x_test, y_test

    elif mode == 'validation':

        # split between train and validation+test
        train_size = int((1 - non_train_percentage) * np.ceil(len(data)))
        y_train = data[window:train_size]
        x_train = series_to_matrix(data, window, 1)[:train_size - window]

        # split validation+test into validation and test
        validation_size = int(val_rel_percentage * np.ceil(len(data) * non_train_percentage))
        y_val = data[train_size:train_size + validation_size]
        x_val = series_to_matrix(data, window, 1)[train_size - window:train_size + validation_size - window]
        y_test = data[train_size + validation_size:]
        x_test = series_to_matrix(data, window, 1)[train_size + validation_size - window:-window]

        return x_train, y_train, x_val, y_val, x_test, y_test


def gaussian_anomaly_detection(input_, mean, variance, threshold):

    anomaly = False

    p_x = gaussian_pdf(input_, mean, variance)

    if p_x < threshold:

        anomaly = True

    return anomaly, p_x


def lstm_exp(filename, 
             num_units, 
             window, 
             batch_size=3, 
             l_rate=.01,
             non_train_percentage=0.5, 
             training_epochs=10, 
             l_rate_test=.1,
             val_rel_percentage=.7,
             normalize=False, 
             time_difference=False):
    
    # clear computational graph
    tf.reset_default_graph()

    # define LSTM features: time steps/hidden layers
    batch_size = batch_size  # length of LSTM networks (n. of LSTM)
    num_units = num_units  # hidden layer in each LSTM
    # size of each input
    window = window

    # create input,output pairs
    X, Y, X_val, Y_val, X_test, Y_test = generate_batches(filename=filename, 
                                                          window=window, 
                                                          mode='validation',
                                                          non_train_percentage=non_train_percentage,
                                                          val_rel_percentage=val_rel_percentage,
                                                          normalize=normalize,
                                                          time_difference=time_difference)
    

    # final dense layerdeclare variable shapes: weights and bias
    weights = tf.Variable(tf.random_normal([num_units, batch_size]))
    bias = tf.Variable(tf.random_normal([1, batch_size]))

    # placeholders (input)
    x = tf.placeholder("float", [None, batch_size, window])
    y = tf.placeholder("float", [None, batch_size])  # dims of target

    # define layers
    lstm_layer = [tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True) for _ in range(batch_size)]
    cells = tf.contrib.rnn.MultiRNNCell(lstm_layer)
    outputs, _ = tf.nn.dynamic_rnn(cells, x, dtype="float32")

    # prediction
    y_hat = tf.matmul(tf.reshape(outputs, shape=(batch_size, num_units)), weights) + bias
    y_hat = tf.transpose(tf.reduce_sum(y_hat, axis=1, keepdims=True))
    y_hat = tf.nn.sigmoid(y_hat)

    # calculate loss and optimization algorithm
    loss = tf.losses.mean_squared_error(labels=y, predictions=y_hat)
    opt = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss)

    # estimate error as the distance (L1) between prediction and target
    error = tf.abs(y_hat - y)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # train phase
        epochs = training_epochs
        plot_y = list()
        plot_y_hat = list()
        list_validation_error = list()
        list_test_error = list()
        for e in range(epochs + 1):

            iter_ = 0
            print("Epoch ", e + 1)

            # train
            if e < epochs - 2:

                while iter_ < int(np.floor(X.shape[0] / batch_size)):
                    batch_x = X[np.newaxis, iter_ * batch_size:batch_size * (iter_ + 1)]
                    batch_y = Y[np.newaxis, iter_ * batch_size:batch_size * (iter_ + 1)]

                    sess.run(opt, feed_dict={x: batch_x, y: batch_y})

                    iter_ = iter_ + 1

            # validation
            elif e == epochs - 2:

                print(" Validation epoch:")

                iter_val_ = 0
                while iter_val_ < int(np.floor(X_val.shape[0] / batch_size)):

                    batch_val_x = X_val[np.newaxis, iter_val_ * batch_size:batch_size * (iter_val_ + 1)]
                    batch_val_y = Y_val[np.newaxis, iter_val_ * batch_size:batch_size * (iter_val_ + 1)]

                    # estimate validation error and append it to a list
                    list_validation_error.append(sess.run(error, feed_dict={x: batch_val_x, y: batch_val_y}))

                    iter_val_ += 1

            # test
            elif e == epochs - 1:
                                
                print(" Test epoch:")
                iter_test_ = 0
                while iter_test_ < int(np.floor(X_test.shape[0] / batch_size)):

                    batch_test_x = X_test[np.newaxis, iter_test_ * batch_size:batch_size * (iter_test_ + 1)]
                    batch_test_y = Y_test[np.newaxis, iter_test_ * batch_size:batch_size * (iter_test_ + 1)]

                    pred_y = sess.run(y_hat, feed_dict={x: batch_test_x, y: batch_test_y})
                    plot_y_hat.append(pred_y)
                    plot_y.append(batch_test_y)
                    list_test_error.append(sess.run(error, feed_dict={x: batch_test_x, y: batch_test_y}))

                    iter_test_ += 1

    dict_results = {"Number_of_units": num_units, "Window_size": window,
                    "Batch_size": batch_size, "Learning_rate": l_rate,
                    "Y": plot_y, "Y_HAT": plot_y_hat,
                    "X_train": X, "Y_train": Y, "X_test": X_test, "Y_test": Y_test,
                    "Validation_Errors": list_validation_error, "Test_Errors": list_test_error}

    return dict_results
