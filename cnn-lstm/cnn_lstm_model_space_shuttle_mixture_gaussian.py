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

import utils_topix as utils


if __name__ == '__main__':
    
    # reset computational graph
    tf.reset_default_graph()
        
    batch_size = 5
    sequence_len = 50
    stride = 2
    learning_rate = 1e-3
    
    # define input/output pairs
    input_ = tf.placeholder(tf.float32, [batch_size, sequence_len])
    target = tf.placeholder(tf.float32, [batch_size, 1])
    
    # expand input to be supported by conv1d operation
    input_ = tf.expand_dims(input_, -1)
    
    # define convolutional layer(s)
    kernel_size = 10
    number_of_channels = 1
    number_of_filters = 30
    
    weights_conv = tf.Variable(tf.truncated_normal(shape=[kernel_size, 
                                                          number_of_channels,
                                                          number_of_filters]))
    bias_conv = tf.Variable(tf.zeros(shape=[number_of_filters]))
    
    layer_conv = tf.nn.conv1d(input_, filters=weights_conv, stride=1, padding='SAME')    
    layer_conv = tf.nn.relu(layer_conv)
    
    # flatten the output
    dims = layer_conv.get_shape()
    number_of_elements = dims[2:].num_elements()
    layer_conv_flatten = tf.reshape(layer_conv, [batch_size, sequence_len, number_of_elements])
    
    # define lstm layer(s)
    number_of_lstm_layers = 3
    
    cell_lstm = tf.contrib.rnn.BasicLSTMCell(number_of_filters)
    layer_lstm = tf.contrib.rnn.MultiRNNCell([cell_lstm for _ in range(number_of_lstm_layers)])
    outputs, states = tf.nn.dynamic_rnn(layer_lstm, layer_conv_flatten, dtype=tf.float32)
    
    # dense layer extraction
    output_lstm = outputs[:, -1, :]
    weights_dense = tf.Variable(tf.truncated_normal(shape=[number_of_filters, 1]))
    bias_dense = tf.Variable(tf.zeros(shape=[1]))
    
    layer_dense = tf.matmul(output_lstm, weights_dense) + bias_dense
    prediction = layer_dense  # linear activation
    
    # loss evaluation
    loss = tf.losses.mean_squared_error(labels=target[-1], predictions=prediction[-1])
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)        
    
    # extract train and test
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches(
                                                             filename='data/space_shuttle_marotta_valve.csv', 
                                                             window=sequence_len,
                                                             stride=stride,
                                                             mode='validation', 
                                                             non_train_percentage=.3,
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
        
    print("Datasets shapes: ", X.shape, Y.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)
    
    # train the model
    epochs = 25
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        
        # train
        for e in range(epochs):
            
            print("epoch", e+1)
            
            iter_ = 0
            
            while iter_ < int(np.floor(x_train.shape[0] / batch_size)):
        
                batch_x = x_train[iter_*batch_size: (iter_+1)*batch_size, :, np.newaxis]
                batch_y = y_train[iter_*batch_size: (iter_+1)*batch_size, np.newaxis]
                
                sess.run(optimizer, feed_dict={input_: batch_x,
                                               target: batch_y})
    
                iter_ +=  1

        # validation
        errors_valid = np.zeros(shape=len(x_valid))
        iter_ = 0
        
        while iter_ < int(np.floor(x_valid.shape[0] / batch_size)):
    
            batch_x = x_valid[iter_*batch_size: (iter_+1)*batch_size, :, np.newaxis]
            batch_y = y_valid[iter_*batch_size: (iter_+1)*batch_size, np.newaxis]
                
            errors_valid[iter_] = sess.run(prediction-batch_y, feed_dict={input_: batch_x,
                                                                          target: batch_y})[-1]

            iter_ +=  1
        
        # estimate mean and deviation of the errors' vector
        #  since we have a batch size that may be different from 1 and we consider
        #   the error of each last batch_y, we need to cut off the zero values
        n_mixtures = 2
        errors_valid = errors_valid[:iter_]
        gaussian_mixture = mixture.GaussianMixture(n_components=n_mixtures)
        gm = gaussian_mixture.fit(errors_valid.reshape(-1, 1))
        means_valid = gm.means_[:,0]
        stds_valid = gm.covariances_[:,0,0]**.5  # square it since it is the cov matrix
        weights_valid = gm.weights_
                
        # test
        predictions = np.zeros(shape=y_test.shape)
        y_test = y_test[:x_test.shape[0]]

        # anomalies' statistics
        errors_test = np.zeros(shape=len(y_test))
        threshold = [scistats.norm.pdf(mean-3.*std, mean, std) for (mean, std) in zip(means_valid, stds_valid)]
        anomalies = np.array([False for _ in range(len(y_test))])
        
        iter_ = 0
        
        while iter_ < int(np.floor(x_test.shape[0] / batch_size)):
    
            batch_x = x_test[iter_*batch_size: (iter_+1)*batch_size, :, np.newaxis]
            batch_y = y_test[iter_*batch_size: (iter_+1)*batch_size, np.newaxis]
                
            predictions[iter_*batch_size:(iter_+1)*batch_size] = sess.run(prediction, feed_dict={input_: batch_x,
                                                                                                 target: batch_y}).flatten()
            
            for i in range(batch_size):
                
                # evaluate Pr(Z=1|X) for each gaussian distribution
                num = np.array([w*scistats.norm.pdf(predictions[(iter_*batch_size)+i]-batch_y[i], mean, std) for (mean, std, w) in zip(means_valid, stds_valid, weights_valid)])
                den = np.sum(num)
                
                index = np.argmax(num/den)
                errors_test[(iter_*batch_size)+i] = scistats.norm.pdf(predictions[(iter_*batch_size)+i]-batch_y[i], means_valid[index], stds_valid[index])
                anomalies[(iter_*batch_size)+i] = (True if (errors_test[(iter_*batch_size)+i] < threshold[index]) else False)
            
            iter_ +=  1
        
        anomalies = np.argwhere(anomalies == True)
            
            
    # plot results
    fig, ax1 = plt.subplots()

    # plot data series
    ax1.plot(y_test[:int(np.floor(x_test.shape[0] / batch_size))*batch_size], 'b', label='index')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('TOPIX')

    # plot predictions
    ax1.plot(predictions[:int(np.floor(x_test.shape[0] / batch_size))*batch_size], 'r', label='prediction')
    ax1.set_ylabel('Change Point')
    plt.legend(loc='best')

    # highlights anomalies
    for i in anomalies:
        
        if i <= len(y_test):
            
            plt.axvspan(i, i+1, color='yellow', alpha=0.5, lw=0)
        
    fig.tight_layout()
    plt.show()               
