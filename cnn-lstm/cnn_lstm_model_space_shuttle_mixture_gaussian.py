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
    stride = 2
    learning_rate = 1e-2
    epochs = 10
    
    # define convolutional layer(s)
    kernel_size = 3
    number_of_filters = 10  # number of convolutions' filters for each LSTM cells
    stride_conv = 1
    
    # define lstm elements
    number_of_lstm_units = 50  # number of hidden units in each lstm
    
    
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
                               padding='VALID') for i in range(batch_size)]
    
    # squeeze and stack the input of the lstm
    layer_conv = tf.squeeze(tf.stack([l for l in layer_conv], axis=-2), axis=-1)
    layer_conv = tf.add(layer_conv, bias_conv)
              
    # non-linear activation before lstm feeding                
#    layer_conv = tf.nn.leaky_relu(layer_conv)    

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
    
    # loss evaluation
    # calculate loss (L2, MSE, huber, hinge or sMAPE, leave uncommented one of them) and optimization algorithm
#    loss = tf.nn.l2_loss(target-prediction)
    loss = tf.losses.mean_squared_error(target, prediction)
#    loss = tf.losses.huber_loss(target, prediction, delta=.25)
#    loss = tf.losses.hinge_loss(target, prediction)
#    loss = (200/batch_size)*tf.reduce_mean(tf.abs(target-prediction))/tf.reduce_mean(target+prediction)
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
        n_mixtures = 2
        errors_valid = errors_valid[:iter_]
        gaussian_mixture = mixture.GaussianMixture(n_components=n_mixtures)
        gm = gaussian_mixture.fit(errors_valid.reshape(-1, 1))
        means_valid = gm.means_[:,0]
        stds_valid = gm.covariances_[:,0,0]**.5  # square it since it is the cov matrix
        weights_valid = gm.weights_
                
        # test
        predictions = np.zeros(shape=(len(y_test), batch_size))
        y_test = y_test[:x_test.shape[0]]

        # anomalies' statistics
        errors_test = np.zeros(shape=(len(y_test), batch_size))
        threshold = [scistats.norm.pdf(mean-2.*std, mean, std) for (mean, std) in zip(means_valid, stds_valid)]
        anomalies = np.array([np.array([False for _ in range(batch_size)]) for _ in range(len(y_test))])
        
        iter_ = 0
        
        while iter_ < int(np.floor(x_test.shape[0] / batch_size)):
    
            batch_x = x_test[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
            batch_y = y_test[np.newaxis, iter_*batch_size: (iter_+1)*batch_size]
                
            predictions[iter_*batch_size:(iter_+1)*batch_size] = sess.run(prediction, feed_dict={input_: batch_x,
                                                                                                 target: batch_y}).flatten()
            
            for i in range(batch_size):
                
                # evaluate Pr(Z=1|X) for each gaussian distribution
                num = np.array([w*scistats.norm.pdf(predictions[(iter_*batch_size), i]-batch_y[:,i], mean, std) for (mean, std, w) in zip(means_valid, stds_valid, weights_valid)])
                den = np.sum(num)
                
                index = np.argmax(num/den)
                errors_test[(iter_*batch_size), i] = scistats.norm.pdf(predictions[(iter_*batch_size), i]-batch_y[:,i], means_valid[index], stds_valid[index])
                anomalies[(iter_*batch_size), i] = (True if (errors_test[(iter_*batch_size), i] < threshold[index]) else False)
            
            iter_ +=  1
        
        anomalies = np.argwhere(anomalies.flatten() == True)
            
         
    predictions = predictions[:,-1]
    
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
