# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 09:13:10 2018

@author: Gabriele
"""

# LSTM for Anomaly Detection

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
                     stride=1,
                     mode='train-test', 
                     non_train_percentage=.7, 
                     val_rel_percentage=.5,
                     normalize=False,
                     time_difference=False,
                     td_method=None):

    data = pd.read_csv(filename, delimiter=',', header=0)
    data = (data.iloc[:, 0]).values

    # normalize dataset (max-min method)
    if normalize is True:
        
        data = (data-np.min(data))/(np.max(data)-np.min(data))
                
    # if the flag 'time-difference' is enabled, turn the dataset into the variation of each time 
    #  step with the previous value (loose the firt sample)
    if time_difference is True:
        
        if td_method is None:
            
            data = data[1:] - data[:-1]
        
        else:
            
            data = td_method(data+1e-5)
            data = data[1:] - data[:-1]
        
    if mode == 'train':

        y = series_to_matrix(data[window:], 1, stride)
        x = series_to_matrix(data, window, stride)
        
        if stride == 1 or window == 1:
            
            x = x[:-1]

        return x, y

    elif mode == 'train-test':

        train_size = int(np.ceil((1 - non_train_percentage) * len(data)))
        train = data[:train_size]; test = data[train_size:]
        
        y_train = series_to_matrix(train[window:], 1, stride)
        x_train = series_to_matrix(train, window, stride)       
        
        y_test = series_to_matrix(test[window:], 1, striding=1)
        x_test = series_to_matrix(test, window, striding=1)
        
        if stride == 1 or window == 1:
            
            x_train = x_train[:-1]; x_test = x_test[:-1]

        return x_train, y_train, x_test, y_test

    elif mode == 'validation':

        # split between train and validation+test
        train_size = int(np.ceil((1 - non_train_percentage) * len(data)))
        train = data[:train_size]
        
        y_train = series_to_matrix(train[window:], 1, stride)
        x_train = series_to_matrix(train, window, stride)

        # split validation+test into validation and test: no stride is applied
        validation_size = int(val_rel_percentage * np.ceil(len(data) * non_train_percentage))
        val = data[train_size:validation_size+train_size]; test = data[validation_size+train_size:]

        y_val = series_to_matrix(val[window:], 1, striding=1)
        x_val = series_to_matrix(val, window, striding=1)
        
        y_test = series_to_matrix(test[window:], 1, striding=1)
        x_test = series_to_matrix(test, window, striding=1)
        
        if stride == 1 or window == 1:
            
            x_train = x_train[:-1]; x_test = x_test[:-1]; x_val = x_val[:-1]

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
             stride=1,
             batch_size=3, 
             l_rate=.01,
             non_train_percentage=0.5, 
             training_epochs=10, 
             l_rate_test=.1,
             val_rel_percentage=.7,
             normalize=False, 
             time_difference=False,
             td_method=None):
    
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
                                                          stride=stride,
                                                          mode='validation',
                                                          non_train_percentage=non_train_percentage,
                                                          val_rel_percentage=val_rel_percentage,
                                                          normalize=normalize,
                                                          time_difference=time_difference,
                                                          td_method=td_method)
    

    # suppress second axis on Y values (the algorithms expects shapes like (n,) for the prediction)
    Y = Y[:,0]; Y_val = Y_val[:,0]; Y_test = Y_test[:,0]
    
    # if the dimensions mismatch (somehow, due tu bugs in generate_batches function,
    #  make them match)
    mismatch = False
    
    if len(X) > len(Y):
        
        X = X[:len(Y)]
        mismatch = True
    
    if len(X_val) > len(Y_val):
        
        X_val = X_val[:len(Y_val)]
        mismatch = True
    
    if len(X_test) > len(Y_test):
        
        X_test = X_test[:len(Y_test)]
        mismatch = True
    
    if mismatch is True: 
        
        print("Mismatched dimensions due to generate batches: this will be corrected automatically.")
        
    print("Datasets shapes: ", X.shape, Y.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

    # final dense layer: declare variable shapes: weights and bias
    weights = tf.get_variable('weights', 
                              shape=[num_units, batch_size, batch_size], 
                              initializer=tf.truncated_normal_initializer())
    bias = tf.get_variable('bias', 
                           shape=[1, batch_size], 
                           initializer=tf.truncated_normal_initializer())

    # placeholders (input)
    x = tf.placeholder("float", [None, batch_size, window]) # (batch, time, input)
    y = tf.placeholder("float", [None, batch_size])  # (batch, output)

    # define the LSTM cells
    cell = tf.nn.rnn_cell.LSTMCell(num_units, 
                                   forget_bias=1.,
                                   state_is_tuple=True,
                                   activation=tf.nn.tanh,
                                   initializer=tf.contrib.layers.xavier_initializer())
    
    initial_state = cell.zero_state(1, tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(cell, 
                                   x,
                                   initial_state=initial_state,
                                   dtype="float32")

    # dense layer: prediction
    y_hat = tf.tensordot(tf.reshape(outputs, shape=(batch_size, num_units)), weights, 2) + bias
#    y_hat = tf.nn.sigmoid(y_hat)
    
    # calculate loss (L2, MSE, huber, hinge or sMAPE, leave uncommented one of them) and optimization algorithm
#    loss = tf.nn.l2_loss(y-y_hat)
    loss = tf.losses.mean_squared_error(y, y_hat)
#    loss = tf.losses.huber_loss(y, y_hat, weights=.2)
#    loss = tf.losses.hinge_loss(y, y_hat)
#    loss = (200/batch_size)*tf.reduce_mean(tf.abs(y-y_hat))/tf.reduce_mean(y+y_hat)
    opt = tf.train.GradientDescentOptimizer(learning_rate=l_rate).minimize(loss)

    # estimate error as the difference between prediction and target
    error = y - y_hat

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
