# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 13:56:22 2019

@author: Emanuele
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scistats
from sklearn import mixture as mixture
import tensorflow as tf
import sys as sys

sys.path.append('../utils')
import utils_dataset as utils
import best_fit_distribution as bfd


def cnn_lstm(filename, 
             sequence_len,
             stride,
             batch_size,
             cnn_kernels,  # (kernel_size, stride, number_of_filters)
             cnn_activations,  # tf activations (list)
             cnn_pooling,
             lstm_params,
             lstm_activation,  # tf activations (list)
             dense_activation,  # tf activation 
             l_rate,
             non_train_percentage, 
             training_epochs, 
             l_rate_test,
             val_rel_percentage,
             normalize, 
             time_difference,
             td_method,
             stop_on_growing_error=False,
             stop_valid_percentage=1.,
             auxiliary_loss = None,
             l_rate_auxiliary = 1e-3):
    
    # training settings
    epochs = 250
    stop_on_growing_error = True  # early-stopping enabler
    stop_valid_percentage = 1.  # percentage of validation used for early-stopping
    
    # reset computational graph
    tf.reset_default_graph()
                  
    # define input/output pairs
    input_ = tf.placeholder(tf.float32, [None, sequence_len, batch_size])  # (batch, input, time)
    target = tf.placeholder(tf.float32, [None, batch_size])  # (batch, output)
    
    weights_conv = bias_conv = list()
    
    for i in range(len(cnn_kernels)):
        
        weights_conv.append(tf.Variable(tf.truncated_normal(shape=[cnn_kernels[i][0],
                                                                      cnn_kernels[i][2],
                                                                      1])) for _ in range(batch_size))
    
        bias_conv[i] = tf.Variable(tf.zeros(shape=[batch_size]))
     
    input_stacked = list(input_)
    for j in range(len(cnn_kernels)):
        
        # stack one input for each battery of filters
        input_stacked.append(tf.stack([input_stacked[j]]*cnn_kernels[j][2], axis=3))
           
        layer_conv = [tf.nn.conv1d(input_stacked[j][:,:,i,:],
                                   filters = weights_conv[j], 
                                   stride = cnn_activations[1], 
                                   padding = 'SAME') for i in range(batch_size)]
    
        # squeeze and stack the input of the lstm
        layer_conv = tf.squeeze(tf.stack([l for l in layer_conv], axis=-2), axis=-1)
        layer_conv = tf.add(layer_conv, bias_conv[j])
                  
        # non-linear activation before lstm feeding                
        layer_conv = cnn_activations[j](layer_conv)    

    # reshape the output so it can be feeded to the lstm (batch, time, input)
    number_of_lstm_inputs = layer_conv.get_shape().as_list()[1]
    layer_conv_flatten = tf.reshape(layer_conv, (-1, batch_size, number_of_lstm_inputs))
    
    # define the LSTM cells
    cells = [tf.contrib.rnn.LSTMCell(lstm_params[i],                                   
                                     forget_bias=1.,
                                     state_is_tuple=True,
                                     activation=lstm_activation[i],
                                     initializer=tf.contrib.layers.xavier_initializer()) for i in range(len(lstm_params))]

    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)    
    outputs, _ = tf.nn.dynamic_rnn(multi_rnn_cell, 
                                   layer_conv_flatten,
                                   dtype="float32")
      
    # final dense layer: declare variable shapes: weights and bias
    weights_dense = tf.get_variable('weights', 
                              shape=[lstm_params[-1], batch_size, batch_size], 
                              initializer=tf.truncated_normal_initializer())
    bias_dense = tf.get_variable('bias', 
                           shape=[1, batch_size], 
                           initializer=tf.truncated_normal_initializer())
    
    # dense layer: prediction
    y_hat = tf.tensordot(tf.reshape(outputs, shape=(batch_size, lstm_params[-1])), weights_dense, 2) + bias_dense

    # activation of the last, dense layer
    y_hat = dense_activation(y_hat)
    
    # estimate error as the difference between prediction and target
    error = target - y_hat
    
    # calculate loss
    loss = tf.nn.l2_loss(error)
    
    # optimization
    opt = tf.train.GradientDescentOptimizer(learning_rate=l_rate).minimize(loss)
    
    # extract train and test
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches( filename=filename, 
                                                                                 window=sequence_len,
                                                                                 stride=stride,
                                                                                 mode='validation', 
                                                                                 non_train_percentage=.5,
                                                                                 val_rel_percentage=.5,
                                                                                 normalize = normalize,
                                                                                 time_differene = time_difference,
                                                                                 td_method = td_method)
    
    # suppress second axis on Y values (the algorithms expects shapes like (n,) for the prediction)
    y_train = y_train[:,0]; y_valid = y_valid[:,0]; y_test = y_test[:,0]
    
    # if the dimensions mismatch (somehow, due tu bugs in generate_batches function,
    #  make them match)
    mismatch = False
    
    if len(x_train) > len(y_train):
        
        x_train = x_train[:len(y_train)]
        mismatch = True
    
    if len(x_valid) > len(y_valid):
        
        x_valid = x_valid[:len(y_valid)]
        mismatch = True
    
    if len(x_test) > len(y_test):
        
        x_test = x_test[:len(y_test)]
        mismatch = True
    
    if mismatch is True: 
        
        print("Mismatched dimensions due to generate batches: this will be corrected automatically.")
        
    print("Datasets shapes: ", x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape)
    
    # train the model
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        
        # train
        last_error_on_valid = np.inf
        current_error_on_valid = .0
        e = 0
        
        while e < epochs:
            
            print("epoch", e+1)
            
            iter_ = 0
            
            while iter_ < int(np.floor(x_train.shape[0] / batch_size)):
        
                batch_x = x_train[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                batch_y = y_train[np.newaxis, iter_*batch_size: (iter_+1)*batch_size]
                
                sess.run(opt, feed_dict={input_: batch_x,
                                                 target: batch_y})   
    
                iter_ +=  1

            if stop_on_growing_error:

                current_error_on_valid = .0
                
                # verificate stop condition
                iter_val_ = 0
                while iter_val_ < int(stop_valid_percentage * np.floor(x_valid.shape[0] / batch_size)):
                    
                    batch_x_val = x_valid[iter_val_*batch_size: (iter_val_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                    batch_y_val = y_valid[np.newaxis, iter_val_*batch_size: (iter_val_+1)*batch_size]
                    
                    # accumulate error
                    current_error_on_valid +=  np.abs(np.sum(sess.run(error, feed_dict={input_: batch_x_val, 
                                                                                        target: batch_y_val})))

                    iter_val_ += 1
                 
                print("Past error on valid: ", last_error_on_valid)
                print("Current total error on valid: ", current_error_on_valid)
                
                if current_error_on_valid > last_error_on_valid:
            
                    print("Stop learning at epoch ", e, " out of ", epochs)
                    e = epochs
                        
                last_error_on_valid = current_error_on_valid                

            
            e += 1

        # validation
        errors_valid = np.zeros(shape=(len(x_valid), batch_size))
        iter_ = 0
        
        while iter_ < int(np.floor(x_valid.shape[0] / batch_size)):
    
            batch_x = x_valid[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
            batch_y = y_valid[np.newaxis, iter_*batch_size: (iter_+1)*batch_size]
                
            errors_valid[iter_] = sess.run(error, feed_dict={input_: batch_x,
                                                                     target: batch_y})

            iter_ +=  1        
        
        ###########################################################################
        # TEST WITH DYNAMIC ERROR'S FUNCTION FITTING
        ###########################################################################
        # estimate mean and deviation of the errors' vector
        #  since we have a batch size that may be different from 1 and we consider
        #   the error of each last batch_y, we need to cut off the zero values
        n_mixtures = 1
        
        iter_ = 0
        
        while iter_ < int(np.floor(x_test.shape[0] / batch_size)):
    
            batch_x = x_test[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
            batch_y = y_test[np.newaxis, iter_*batch_size: (iter_+1)*batch_size]
                
            predictions[iter_] = sess.run(prediction, feed_dict={input_: batch_x,
                                                                 target: batch_y}).flatten()
    
            errors_test[iter_] = batch_y - predictions[iter_]
         
            for i in range(batch_size):
                
                # evaluate Pr(Z=1|X) for each gaussian distribution
                num = np.array([w*scistats.norm.pdf(predictions[iter_, i]-batch_y[:,i], mean, std) for (mean, std, w) in zip(means_valid, stds_valid, weights_valid)])
                den = np.sum(num)                
                index = np.argmax(num/den)                
                gaussian_error_statistics[iter_, i] = scistats.norm.pdf(predictions[iter_, i]-batch_y[:,i], means_valid[index], stds_valid[index])
                anomalies[iter_, i] = (True if (gaussian_error_statistics[iter_, i] < threshold[index]) else False)
            
            iter_ +=  1
        
        anomalies = np.argwhere(anomalies.flatten() == True)            
    
    errors_test = errors_test.flatten()
    predictions = predictions.flatten()
    
    # plot results
    fig, ax1 = plt.subplots()

    # plot data series
    ax1.plot(y_test[:int(np.floor(x_test.shape[0] / batch_size))*batch_size], 'b', label='index')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Index Value')

    # plot predictions
    ax1.plot(predictions[:int(np.floor(x_test.shape[0] / batch_size))*batch_size], 'r', label='prediction')
    ax1.set_ylabel('Prediction')
    plt.legend(loc='best')

    # highlights anomalies
    for i in anomalies:
        
        if i <= len(y_test):
            
            plt.axvspan(i, i+1, color='yellow', alpha=0.5, lw=0)
        
    fig.tight_layout()
    plt.show()
    
    print("Total test error:", np.sum(np.abs(errors_test)))
    
    # plot reconstructed signal
    tot_y = 0.
    tot_y_hat = 0.
    recovered_plot_y = np.zeros(shape=len(predictions[:int(np.floor(x_test.shape[0] / batch_size))*batch_size])+1)
    recovered_plot_y_hat = np.zeros(shape=len(predictions[:int(np.floor(x_test.shape[0] / batch_size))*batch_size])+1)
    for i in range(1, len(recovered_plot_y)):
        
        recovered_plot_y[i] = tot_y
        recovered_plot_y_hat[i] = tot_y_hat
        
        tot_y += y_test[i-1]
        tot_y_hat += predictions[i-1] 
                
    fig, ax1 = plt.subplots()

    # plot data series
    print("\nReconstruction:")
    ax1.plot(recovered_plot_y, 'b', label='index')
    ax1.set_xlabel('RECONSTRUCTION: Date')
    ax1.set_ylabel('Space Shuttle')

    # plot predictions
    ax1.plot(recovered_plot_y_hat, 'r', label='prediction')
    ax1.set_ylabel('RECONSTRUCTION: Prediction')
    plt.legend(loc='best')

    fig.tight_layout()
    plt.show()
    
    # errors on test
    print("\nTest errors:")
    plt.hist(np.array(errors_test).ravel(), bins=30) 

    # performances
    target_anomalies = np.zeros(shape=int(np.floor(x_test.shape[0] / batch_size))*batch_size)
    
    # caveat: define the anomalies based on absolute position in test set (i.e. size matters!)
    # train 50%, validation_relative 50%
    target_anomalies[520:540] = 1
    
    # real values
    condition_positive = np.argwhere(target_anomalies == 1)
    condition_negative = np.argwhere(target_anomalies == 0)
    
    # predictions
    predicted_positive = anomalies
    predicted_negative = np.setdiff1d(np.array([i for i in range(len(target_anomalies))]), 
                                      predicted_positive,
                                      assume_unique=True)
    
    # precision
    precision = len(np.intersect1d(condition_positive, predicted_positive))/len(predicted_positive)
    
    # fall-out
    fall_out = len(np.intersect1d(predicted_positive, condition_negative))/len(condition_negative)
    
    # recall
    recall = len(np.intersect1d(condition_positive, predicted_positive))/len(condition_positive)
    
    print("Anomalies: ", condition_positive.T)
    print("Anomalies Detected: ", predicted_positive.T)
    print("Precision: ", precision)
    print("Fallout: ", fall_out)
    print("Recall: ", recall)

    # top-n distributions that fit the test errors.
    top_n = 10
    cols = [col for col in bfd.best_fit_distribution(np.array(errors_test).ravel(), top_n=top_n)]
    top_n_distr = pd.DataFrame(cols, index=['NAME', 'PARAMS', 'ERRORS'])
    print("\n\nTop distributions: NAME ERRORS PARAM ", top_n_distr)
    
    file_ptr = np.loadtxt('../__tmp/__tmp_res.csv', dtype=object)
    for i in range(top_n):
        
        file_ptr = np.append(file_ptr, top_n_distr[i]['NAME'])
    
    np.savetxt('../__tmp/__tmp_res.csv', file_ptr, fmt='%s')
    
    # save sMAPE of each model
    sMAPE_error_len = len(np.array(errors_test).ravel())
    sMAPE_den = np.abs(np.array(predictions).ravel()[:sMAPE_error_len])+np.abs(np.array(y_test).ravel()[:sMAPE_error_len])
    perc_error = np.mean(200*(np.abs(np.array(errors_test).ravel()[:sMAPE_error_len]))/sMAPE_den)
    print("Percentage error: ", perc_error)
    
    file_ptr = np.loadtxt('../__tmp/__tmp_err.csv', dtype=object)
    file_ptr = np.append(file_ptr, str(perc_error))
    np.savetxt('../__tmp/__tmp_err.csv', file_ptr, fmt='%s')                   
