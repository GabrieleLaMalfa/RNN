# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:26:34 2019

@author: Emanuele
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scistats
import sys
import tensorflow as tf

sys.path.append('../utils')
import clustering as clst
import utils_dataset as utils


if __name__ == '__main__':
    
    # reset computational graph
    tf.reset_default_graph()
        
    batch_size = 15
    sequence_len = 8
    stride = 3
    learning_rate = 5e-3
    epochs = 25
    sigma_threshold = 5.  # /tau
    n_clusters = 4  # number of clusters for the k-means
    n_clusters_td = 2  # number of clusters for the k-means (td data)
    
    # define first convolutional layer(s)
    kernel_size_first = 3
    number_of_filters_first = 10  # number of convolutions' filters for each LSTM cells
    stride_conv_first = 1

    # define second convolutional layer(s)
    kernel_size_second = 2
    number_of_filters_second = 15  # number of convolutions' filters for each LSTM cells
    stride_conv_second = 2
    
    # define lstm elements
    number_of_lstm_units = 50  # number of hidden units in each lstm  
    
    # define input/output pairs
    input_ = tf.placeholder(tf.float32, [None, sequence_len, batch_size])  # (batch, input, time)
    mem_cluster = tf.placeholder(tf.float32, [None, batch_size, 2])  # for each point, its cluster info
    target = tf.placeholder(tf.float32, [None, batch_size])  # (batch, output)
    
    # first cnn layer
    weights_conv_first = [tf.Variable(tf.truncated_normal(shape=[kernel_size_first,
                                                           number_of_filters_first,
                                                           1])) for _ in range(batch_size)]
    
    bias_conv_first = tf.Variable(tf.zeros(shape=[batch_size]))
    
    # stack one input for each battery of filters
    input_stacked = tf.stack([input_]*number_of_filters_first, axis=3)
       
    layer_conv_first = [tf.nn.conv1d(input_stacked[:,:,i,:],
                               filters=weights_conv_first[i], 
                               stride=stride_conv_first, 
                               padding='SAME') for i in range(batch_size)]
    
    # squeeze and stack the input of the lstm
    layer_conv_first = tf.squeeze(tf.stack([l for l in layer_conv_first], axis=-2), axis=-1)
    layer_conv_first = tf.add(layer_conv_first, bias_conv_first)
              
    # non-linear activation before lstm feeding                
    layer_conv_first = tf.nn.tanh(layer_conv_first)
    
    #
    # second cnn layer
    weights_conv_second = [tf.Variable(tf.truncated_normal(shape=[kernel_size_second,
                                                           number_of_filters_second,
                                                           1])) for _ in range(batch_size)]
    
    bias_conv_second = tf.Variable(tf.zeros(shape=[batch_size]))
    
    # stack one input for each battery of filters
    input_stacked = tf.stack([layer_conv_first]*number_of_filters_second, axis=3)
       
    layer_conv_second = [tf.nn.conv1d(input_stacked[:,:,i,:],
                               filters=weights_conv_second[i], 
                               stride=stride_conv_second, 
                               padding='SAME') for i in range(batch_size)]
    
    # squeeze and stack the input of the lstm
    layer_conv_second = tf.squeeze(tf.stack([l for l in layer_conv_second], axis=-2), axis=-1)
    layer_conv_second = tf.add(layer_conv_second, bias_conv_second)
              
    # non-linear activation before lstm feeding                
    layer_conv_second = tf.nn.tanh(layer_conv_second)    
    
    # reshape the output so it can be feeded to the lstm (batch, time, input)
    number_of_lstm_inputs = layer_conv_second.get_shape().as_list()[1]
    layer_conv_flatten = tf.reshape(layer_conv_second, (-1, batch_size, number_of_lstm_inputs))

    # add the cluster's info to the lstm
    layer_conv_flatten = tf.concat([mem_cluster, layer_conv_flatten], 2)
        
    # define the LSTM cells
    cells = [tf.contrib.rnn.LSTMCell(number_of_lstm_units) for _ in range(1)]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)    
    outputs, _ = tf.nn.dynamic_rnn(multi_rnn_cell, 
                                   layer_conv_flatten,
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
#    prediction = tf.nn.relu(prediction)
    
#    # exponential decay of the predictions
#    decay = tf.constant(np.array([2**(-i) for i in range(batch_size+2)], dtype='float32'))
#    prediction = prediction*decay

    # loss evaluation
    # calculate loss (L2, MSE, huber, hinge, sMAPE: leave uncommented one of them)
    loss = tf.nn.l2_loss(target-prediction)
#    loss = tf.losses.mean_squared_error(target, prediction)
#    loss = tf.losses.huber_loss(target, prediction, delta=.25)
#    loss = tf.losses.hinge_loss(target, prediction)
#    loss = (200/batch_size)*tf.reduce_mean(tf.abs(target-prediction))/tf.reduce_mean(target+prediction)
    
    # optimization algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # extract clusters information from clean data
    x_train_tmp, y_train_tmp, x_valid_tmp, y_valid_tmp, x_test_tmp, y_test_tmp = utils.generate_batches(
                                                                                     filename='data/power_consumption.csv', 
                                                                                     window=sequence_len,
                                                                                     stride=stride,
                                                                                     mode='validation', 
                                                                                     non_train_percentage=.3,
                                                                                     val_rel_percentage=.8,                                                                                     
                                                                                     normalize=True,
                                                                                     time_difference=False,
                                                                                     td_method=None)
    
    # cluster info relative to signal's value (cluster's means)
    clusters_info = clst.k_means(x_train_tmp, n_clusters)    
    
    # extract train and test
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches(
                                                             filename='data/power_consumption.csv', 
                                                             window=sequence_len,
                                                             stride=stride,
                                                             mode='validation', 
                                                             non_train_percentage=.3,
                                                             val_rel_percentage=.8,
                                                             normalize=True,
                                                             time_difference=True,
                                                             td_method=np.log2)
    
    # cluster info relative to time difference (cluster's means)
    clusters_info_td = clst.k_means(x_train, n_clusters_td)
    
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
                
                # predict clusters memberships              
                cluster_batch_x =  x_train_tmp[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)           
                cluster_batch_x_td = x_train[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                mem = clusters_info.predict(cluster_batch_x[0,:,:].T).reshape(-1, batch_size, 1)
                mem_td = clusters_info_td.predict(cluster_batch_x_td[0,:,:].T).reshape(-1, batch_size, 1)
                mem = np.concatenate((clusters_info.cluster_centers_.sum(axis=1)[mem], 
                                      clusters_info_td.cluster_centers_.sum(axis=1)[mem_td]), axis=2)
                                
                sess.run(optimizer, feed_dict={input_: batch_x,
                                               mem_cluster: mem,
                                               target: batch_y})  
    
                iter_ +=  1

        # validation: we use the td clusters
        errors_valid = list(list() for _ in range(n_clusters_td))
        iter_ = 0
        
        while iter_ < int(np.floor(x_valid.shape[0] / batch_size)):
    
            batch_x = x_valid[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
            batch_y = y_valid[np.newaxis, iter_*batch_size: (iter_+1)*batch_size]
            
            # predict clusters memberships              
            cluster_batch_x =  x_valid_tmp[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)           
            cluster_batch_x_td = x_valid[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
            mem_value = clusters_info.predict(cluster_batch_x[0,:,:].T).reshape(-1, batch_size, 1)
            mem_value_td = clusters_info_td.predict(cluster_batch_x_td[0,:,:].T).reshape(-1, batch_size, 1)
            mem = np.concatenate((clusters_info.cluster_centers_.sum(axis=1)[mem_value], 
                                  clusters_info_td.cluster_centers_.sum(axis=1)[mem_value_td]), axis=2)             
            mem_indices = np.concatenate((mem_value, mem_value_td), axis=2)
            
            error = sess.run(prediction-batch_y, feed_dict={input_: batch_x,
                                                            mem_cluster: mem,
                                                            target: batch_y})
    
            for l in range(batch_size):
                
                errors_valid[mem_value_td[0,l,0]].append(error[0,l])

            iter_ +=  1
        
        # estimate mean and deviation of each errors' vector
        errors_valid = np.array([np.asarray(e) for e in errors_valid], dtype=object)
        means_valid = np.array([np.mean(e) for e in errors_valid])
        stds_valid = np.array([np.std(e) for e in errors_valid])
                
        # test
        predictions = np.zeros(shape=(int(np.floor(x_test.shape[0] / batch_size)), batch_size))
        y_test = y_test[:x_test.shape[0]]

        # anomalies' statistics
        gaussian_error_statistics = np.zeros(shape=(len(predictions), batch_size))
        errors_test = np.zeros(shape=(len(predictions), batch_size))
        threshold = [scistats.norm.pdf(mean-sigma_threshold*std, mean, std) for (mean, std) in zip(means_valid, stds_valid)]
        anomalies = np.array([np.array([False for _ in range(batch_size)]) for _ in range(len(y_test))])
        
        iter_ = 0
        
        while iter_ < int(np.floor(x_test.shape[0] / batch_size)):
    
            batch_x = x_test[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
            batch_y = y_test[np.newaxis, iter_*batch_size: (iter_+1)*batch_size]
            
            # predict clusters memberships              
            cluster_batch_x =  x_test_tmp[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)           
            cluster_batch_x_td = x_test[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
            mem_value = clusters_info.predict(cluster_batch_x[0,:,:].T).reshape(-1, batch_size, 1)
            mem_value_td = clusters_info_td.predict(cluster_batch_x_td[0,:,:].T).reshape(-1, batch_size, 1)
            mem = np.concatenate((clusters_info.cluster_centers_.sum(axis=1)[mem_value], 
                                  clusters_info_td.cluster_centers_.sum(axis=1)[mem_value_td]), axis=2)
            mem_indices = np.concatenate((mem_value, mem_value_td), axis=2)
            
            predictions[iter_] = sess.run(prediction, feed_dict={input_: batch_x,
                                                                 mem_cluster: mem,
                                                                 target: batch_y}).flatten()
                
            errors_test[iter_] = batch_y-predictions[iter_]
             
            for i in range(batch_size):
                
                # evaluate Pr(Z=1|X) for each gaussian distribution              
                gaussian_error_statistics[iter_, i] = scistats.norm.pdf(predictions[iter_, i]-batch_y[:,i], means_valid[mem_value_td[0,l,0]], stds_valid[mem_value_td[0,l,0]])
                anomalies[iter_, i] = (True if (gaussian_error_statistics[iter_, i] < threshold[mem_value_td[0,l,0]]) else False)
            
            iter_ +=  1
               
        # highlights anomalies
        anomalies = np.argwhere(anomalies.flatten() == True)
        
        # ignore the anomaly evalution if we are considering the last samples, since it may
        #  rise some difficulties the logarithm as time_difference in the last y_test's points
        anomalies = anomalies[anomalies < len(y_test)-15]
    
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
    
    # highlights anomalies
    for i in anomalies:
        
        if i <= len(y_test):
            
            plt.axvspan(i, i+1, color='yellow', alpha=0.5, lw=0)
        
    fig.tight_layout()

    fig.tight_layout()
    plt.show()
    
    # errors on test
    print("\nTest errors:")
    plt.hist(np.array(errors_test).ravel(), bins=30) 

    # performances
    target_anomalies = np.zeros(shape=int(np.floor(x_test.shape[0] / batch_size))*batch_size)
    
    # caveat: define the anomalies based on absolute position in test set (i.e. size matters!)
    # train 70%, validation_relative 80%
    target_anomalies[1400:1600] = 1
    
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
