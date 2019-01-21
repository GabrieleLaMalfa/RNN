# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 13:56:22 2019

@author: Emanuele
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scistats
from sklearn import mixture as mixture
import tensorflow as tf

import utils_dataset as utils


if __name__ == '__main__':
    
    # reset computational graph
    tf.reset_default_graph()
        
    batch_size = 5
    sequence_len = 20
    stride = 3
    learning_rate = 5e-3
    epochs = 25
    
    # define convolutional layer(s)
    kernel_size = 3
    number_of_filters = 100  # number of convolutions' filters for each LSTM cells
    stride_conv = 1
    
    # define lstm elements
    number_of_lstm_units = 75  # number of hidden units in each lstm
    
    
    # define input/output pairs
    input_ = tf.placeholder(tf.float32, [None, sequence_len, batch_size])  # (batch, input, time)
    target = tf.placeholder(tf.float32, [None, batch_size])  # (batch, output)
    
    weights_conv = [tf.Variable(tf.truncated_normal(shape=[kernel_size,
                                                           number_of_filters,
                                                           1])) for _ in range(batch_size)]
    
    bias_conv = tf.Variable(tf.zeros(shape=[batch_size]))
    
    # stack one input for each battery of filters
    input_stacked = tf.stack([input_]*number_of_filters, axis=3)
       
    layer_conv = [tf.nn.conv1d(input_stacked[:,:,i,:],
                               filters=weights_conv[i], 
                               stride=stride_conv, 
                               padding='SAME') for i in range(batch_size)]
    
    # squeeze and stack the input of the lstm
    layer_conv = tf.squeeze(tf.stack([l for l in layer_conv], axis=-2), axis=-1)
    layer_conv = tf.add(layer_conv, bias_conv)
              
    # non-linear activation before lstm feeding                
    layer_conv = tf.nn.leaky_relu(layer_conv)    

    # reshape the output so it can be feeded to the lstm (batch, time, input)
    number_of_lstm_inputs = layer_conv.get_shape().as_list()[1]
    layer_conv_flatten = tf.reshape(layer_conv, (-1, batch_size, number_of_lstm_inputs))
        
    # define the LSTM cells
    cell = tf.nn.rnn_cell.LSTMCell(number_of_lstm_units, 
                                   forget_bias=1.,
                                   state_is_tuple=True,
                                   activation=tf.nn.tanh,
                                   initializer=tf.contrib.layers.xavier_initializer())
    
    initial_state = cell.zero_state(1, tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(cell, 
                                   layer_conv_flatten,
                                   initial_state=initial_state,
                                   dtype="float32")
        
    # dense layer extraction
    # final dense layer: declare variable shapes: weights and bias
    weights_dense = tf.get_variable('weights', 
                              shape=[number_of_lstm_units, batch_size, batch_size], 
                              initializer=tf.truncated_normal_initializer())
    bias_dense = tf.get_variable('bias', 
                           shape=[1, batch_size], 
                           initializer=tf.truncated_normal_initializer())
    
    output_lstm = outputs[:, -1, :]
    
    # dense layer: prediction
    prediction = tf.tensordot(tf.reshape(outputs, shape=(batch_size, number_of_lstm_units)), weights_dense, 2) + bias_dense
    
#    # (optional) exponential decay of the predictions
#    decay = tf.constant(np.array([2**(-i) for i in range(batch_size)], dtype='float32')[::-1])
#    prediction_with_decay = prediction*decay
    prediction_with_decay = prediction

    # loss evaluation
    # calculate loss (L2, MSE, huber, hinge, sMAPE: leave uncommented one of them)
    loss = tf.nn.l2_loss(target-prediction_with_decay)
#    loss = tf.losses.mean_squared_error(target, prediction_with_decay)
#    loss = tf.losses.huber_loss(target, prediction_with_decay, delta=.25)
#    loss = tf.losses.hinge_loss(target, prediction_with_decay)
#    loss = (200/batch_size)*tf.reduce_mean(tf.abs(target-prediction_with_decay))/tf.reduce_mean(target+prediction_with_decay)
    
    # optimization algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)        
    
    # extract train and test
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches(
                                                             filename='data/space_shuttle_marotta_valve.csv', 
                                                             window=sequence_len,
                                                             stride=stride,
                                                             mode='validation', 
                                                             non_train_percentage=.5,
                                                             val_rel_percentage=.5,
                                                             normalize=True,
                                                             time_difference=True,
                                                             td_method=None)
    
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
        for e in range(epochs):
            
            print("epoch", e+1)
            
            iter_ = 0
            
            while iter_ < int(np.floor(x_train.shape[0] / batch_size)):
        
                batch_x = x_train[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                batch_y = y_train[np.newaxis, iter_*batch_size: (iter_+1)*batch_size]
                
                sess.run(optimizer, feed_dict={input_: batch_x,
                                               target: batch_y})   
    
                iter_ +=  1

        # validation
        errors_valid = np.zeros(shape=(len(x_valid), batch_size))
        iter_ = 0
        
        while iter_ < int(np.floor(x_valid.shape[0] / batch_size)):
    
            batch_x = x_valid[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
            batch_y = y_valid[np.newaxis, iter_*batch_size: (iter_+1)*batch_size]
                
            errors_valid[iter_] = sess.run(prediction-batch_y, feed_dict={input_: batch_x,
                                                                          target: batch_y})

            iter_ +=  1
        
        # estimate mean and deviation of the errors' vector
        #  since we have a batch size that may be different from 1 and we consider
        #   the error of each last batch_y, we need to cut off the zero values
        n_mixtures = 1
        errors_valid = errors_valid[:iter_].flatten()
        gaussian_mixture = mixture.GaussianMixture(n_components=n_mixtures)
        gm = gaussian_mixture.fit(errors_valid.reshape(-1, 1))
        means_valid = gm.means_[:,0]
        stds_valid = gm.covariances_[:,0,0]**.5  # square it since it is the cov matrix
        weights_valid = gm.weights_
                
        # test
        predictions = np.zeros(shape=(int(np.floor(x_test.shape[0] / batch_size)), batch_size))
        y_test = y_test[:x_test.shape[0]]

        # anomalies' statistics
        gaussian_error_statistics = np.zeros(shape=(len(predictions), batch_size))
        errors_test = np.zeros(shape=(len(predictions), batch_size))
        threshold = [scistats.norm.pdf(mean-2.5*std, mean, std) for (mean, std) in zip(means_valid, stds_valid)]
        anomalies = np.array([np.array([False for _ in range(batch_size)]) for _ in range(len(y_test))])
        
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
    print("\nTest errors' histogram:")
    plt.hist(np.array(errors_test).ravel(), bins=30)

    # performances
    target_anomalies = np.zeros(shape=int(np.floor(x_test.shape[0] / batch_size))*batch_size)

    
    # caveat: define the anomalies based on absolute position in test set (i.e. size matters!)
    target_anomalies[500:580] = 1
    
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
