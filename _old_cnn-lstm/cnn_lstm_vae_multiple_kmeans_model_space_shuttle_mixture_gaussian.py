# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 19:20:04 2019

@author: Emanuele
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scistats
import tensorflow as tf
import sys as sys

sys.path.append('../utils')
import clustering as clst
import utils_dataset as utils
import best_fit_distribution as bfd


if __name__ == '__main__':
    
    # training settings
    stop_on_growing_error = True  # early-stopping enabler
    stop_valid_percentage = 1.  # percentage of validation used for early-stopping
    
    # reset computational graph
    tf.reset_default_graph()
        
    batch_size = 5
    sequence_len = 6
    stride = 2
    learning_rate = 1e-4
    epochs = 250
    sigma_threshold = 1.  # /tau
    n_clusters = 5  # number of clusters for the k-means
    n_clusters_td = 5  # number of clusters for the k-means (td data)
    
    # define first convolutional layer(s)
    kernel_size_first = 3
    number_of_filters_first = 35  # number of convolutions' filters for each LSTM cells
    stride_conv_first = 2
    
    # define lstm parameters
    number_of_lstm_units = 50  # number of hidden units in each lstm  
    
    # define VAE parameters
    learning_rate_elbo = 1e-5
#    sigma_threshold_elbo = 2.5  # threshold for the VAE gaussian
    vae_hidden_size = 2
    sigma_threshold_elbo = 2.25
    
    vae_encoder_shape_weights = [batch_size*sequence_len, 35, vae_hidden_size*2]
    vae_decoder_shape_weights = [vae_hidden_size, 25, batch_size*sequence_len]

    zip_weights_encoder = zip(vae_encoder_shape_weights[:-1], vae_encoder_shape_weights[1:])
    weights_vae_encoder = [tf.Variable(tf.truncated_normal(shape=[shape,
                                                                  next_shape])) for (shape, next_shape) in zip_weights_encoder]
    bias_vae_encoder = [tf.Variable(tf.truncated_normal(shape=[shape])) for shape in vae_encoder_shape_weights[1:]]
    
    zip_weights_decoder = zip(vae_decoder_shape_weights[:-1], vae_decoder_shape_weights[1:])
    weights_vae_decoder = [tf.Variable(tf.truncated_normal(shape=[shape,
                                                                  next_shape])) for (shape, next_shape) in zip_weights_decoder]
    bias_vae_decoder = [tf.Variable(tf.truncated_normal(shape=[shape])) for shape in vae_decoder_shape_weights[1:]]
    
    
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
    layer_conv_first = tf.nn.leaky_relu(layer_conv_first)   
    
    # reshape the output so it can be feeded to the lstm (batch, time, input)
    number_of_lstm_inputs = layer_conv_first.get_shape().as_list()[1]
    layer_conv_flatten = tf.reshape(layer_conv_first, (-1, batch_size, number_of_lstm_inputs))

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

    # loss evaluation
    loss = tf.nn.l2_loss(target-prediction)
    
    # optimization algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # VAE parameters' initialization
    vae_encoder_shape_weights = [batch_size*sequence_len, vae_hidden_size*2]
    vae_decoder_shape_weights = [vae_hidden_size, batch_size*sequence_len]
    
    zip_weights_encoder = zip(vae_encoder_shape_weights[:-1], vae_encoder_shape_weights[1:])
    
    weights_vae_encoder = [tf.Variable(tf.truncated_normal(shape=[shape,
                                                                  next_shape])) for (shape, next_shape) in zip_weights_encoder]
    bias_vae_encoder = [tf.Variable(tf.truncated_normal(shape=[shape])) for shape in vae_encoder_shape_weights[1:]]
    
    zip_weights_decoder = zip(vae_decoder_shape_weights[:-1], vae_decoder_shape_weights[1:])
    weights_vae_decoder = [tf.Variable(tf.truncated_normal(shape=[shape,
                                                                  next_shape])) for (shape, next_shape) in zip_weights_decoder]
    bias_vae_decoder = [tf.Variable(tf.truncated_normal(shape=[shape])) for shape in vae_decoder_shape_weights[1:]]
    
    #
    # VAE graph's definition
    flattened_input = tf.layers.flatten(input_)
    
    vae_encoder = tf.matmul(flattened_input, weights_vae_encoder[0]) + bias_vae_encoder[0]
    
    for (w_vae, b_vae) in zip(weights_vae_encoder[1:], bias_vae_encoder[1:]):
        
        vae_encoder = tf.nn.relu(vae_encoder)
        vae_encoder = tf.matmul(vae_encoder, w_vae) + b_vae
    
    # means and variances' vectors of the learnt hidden distribution
    #  we assume the hidden gaussian's variances matrix is diagonal
    loc = tf.slice(tf.nn.relu(vae_encoder), [0, 0], [-1, vae_hidden_size])
    loc = tf.squeeze(loc, axis=0)
    scale = tf.slice(tf.nn.softplus(vae_encoder), [0, vae_hidden_size], [-1, vae_hidden_size])
    scale = tf.squeeze(scale, 0)
    
    vae_hidden_distr = tf.contrib.distributions.MultivariateNormalDiag(loc, scale)    
    vae_hidden_state = vae_hidden_distr.sample()
    
    feed_decoder = tf.reshape(vae_hidden_state, shape=(-1, vae_hidden_size))
    vae_decoder = tf.matmul(feed_decoder, weights_vae_decoder[0]) + bias_vae_decoder[0]
    vae_decoder = tf.nn.relu(vae_decoder)    
    
    for (w_vae, b_vae) in zip(weights_vae_decoder[1:], bias_vae_decoder[1:]):
        
        vae_decoder = tf.matmul(vae_decoder, w_vae) + b_vae
        vae_decoder = tf.nn.relu(vae_decoder)
    
    # time-series reconstruction and ELBO loss
    vae_reconstruction = tf.contrib.distributions.MultivariateNormalDiag(tf.constant(np.zeros(batch_size*sequence_len, dtype='float32')),
                                                                         tf.constant(np.ones(batch_size*sequence_len, dtype='float32')))
    likelihood = vae_reconstruction.log_prob(flattened_input)
    
    prior = tf.contrib.distributions.MultivariateNormalDiag(tf.constant(np.zeros(vae_hidden_size, dtype='float32')),
                                                            tf.constant(np.ones(vae_hidden_size, dtype='float32')))
    
    divergence = tf.contrib.distributions.kl_divergence(vae_hidden_distr, prior)
    elbo = tf.reduce_mean(likelihood - divergence)
    
    optimizer_elbo = tf.train.AdamOptimizer(learning_rate_elbo).minimize(elbo)       
    
    #
    # extract clusters information from clean data
    x_train_tmp, y_train_tmp, x_valid_tmp, y_valid_tmp, x_test_tmp, y_test_tmp = utils.generate_batches(
                                                                                     filename='../data/space_shuttle_marotta_valve.csv', 
                                                                                     window=sequence_len,
                                                                                     stride=stride,
                                                                                     mode='validation', 
                                                                                     non_train_percentage=.5,
                                                                                     val_rel_percentage=.5,                                                                                     
                                                                                     normalize='maxmin01',
                                                                                     time_difference=False,
                                                                                     td_method=None)
    
    # cluster info relative to signal's value (cluster's means)
    clusters_info = clst.k_means(x_train_tmp, n_clusters)    
    
    # extract train and test
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches(
                                                             filename='../data/space_shuttle_marotta_valve.csv', 
                                                             window=sequence_len,
                                                             stride=stride,
                                                             mode='validation', 
                                                             non_train_percentage=.5,
                                                             val_rel_percentage=.5,
                                                             normalize='maxmin01',
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
        last_error_on_valid = np.inf
        current_error_on_valid = .0
        e = 0
        
        while e < epochs:
            
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
        
                # run VAE encoding-decoding
                sess.run(optimizer_elbo, feed_dict={input_: cluster_batch_x})

                iter_ +=  1
            
            if stop_on_growing_error:

                current_error_on_valid = .0
                
                # verificate stop condition
                iter_val_ = 0
                while iter_val_ < int(stop_valid_percentage * np.floor(x_valid.shape[0] / batch_size)):
                    
                    batch_x_val = x_valid[iter_val_*batch_size: (iter_val_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                    batch_y_val = y_valid[np.newaxis, iter_val_*batch_size: (iter_val_+1)*batch_size]

                    # predict clusters memberships              
                    cluster_batch_x =  x_valid_tmp[iter_val_*batch_size: (iter_val_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)           
                    cluster_batch_x_td = x_valid[iter_val_*batch_size: (iter_val_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                    mem_value = clusters_info.predict(cluster_batch_x[0,:,:].T).reshape(-1, batch_size, 1)
                    mem_value_td = clusters_info_td.predict(cluster_batch_x_td[0,:,:].T).reshape(-1, batch_size, 1)
                    mem = np.concatenate((clusters_info.cluster_centers_.sum(axis=1)[mem_value], 
                                          clusters_info_td.cluster_centers_.sum(axis=1)[mem_value_td]), axis=2)             
                    mem_indices = np.concatenate((mem_value, mem_value_td), axis=2)
                    
                    current_error_on_valid +=  np.abs(np.sum(sess.run(prediction-batch_y, feed_dict={input_: batch_x_val,
                                                                                                     mem_cluster: mem,
                                                                                                     target: batch_y_val})))
                    
                    iter_val_ += 1
                    
                print("Past error on valid: ", last_error_on_valid)
                print("Current total error on valid: ", current_error_on_valid)
                
                if current_error_on_valid > last_error_on_valid:
            
                    print("Stop learning at epoch ", e, " out of ", epochs)
                    e = epochs
                        
                last_error_on_valid = current_error_on_valid  
    
            e += 1

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
        
        # elbo threshold
        mean_elbo = np.zeros(shape=vae_hidden_size)
        std_elbo = np.eye(vae_hidden_size)
        threshold_elbo = scistats.multivariate_normal.pdf(mean_elbo-sigma_threshold_elbo, mean_elbo, std_elbo)
        vae_anomalies = np.zeros(shape=(len(predictions)))
        
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
            
            # test if the VAE encoding is anomalous for the sample
            test_hidden_state = sess.run(vae_hidden_state, feed_dict={input_: cluster_batch_x})
            vae_anomalies[iter_] =  scistats.multivariate_normal.pdf(test_hidden_state, 
                                                                     np.zeros(shape=vae_hidden_size), 
                                                                     np.eye(vae_hidden_size))
                        
            if vae_anomalies[iter_] < threshold_elbo:
                   
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
    plt.show()
    
    # plot vae likelihood values
    fig, ax1 = plt.subplots()
    ax1.plot(vae_anomalies, 'b', label='time')
    ax1.set_ylabel('VAE: Anomalies likelihood')
    plt.legend(loc='best')
    
    # highlights elbo's boundary
    ax1.plot(np.array([threshold_elbo for _ in range(len(vae_anomalies))]), 'r', label='prediction')
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
