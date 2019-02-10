# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 09:36:16 2019

@author: Emanuele
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scistats
import sys
import tensorflow as tf

sys.path.append('../utils')
import utils_dataset as utils


if __name__ == '__main__':
    
    # reset computational graph
    tf.reset_default_graph()
        
    # data parameters
    batch_size = 5
    sequence_len = 15
    stride = 5
    
    # training epochs
    epochs = 25
    
    # define VAE parameters
    learning_rate_elbo = 1e-5
    vae_hidden_size = 5
    threshold_elbo = 5e-3
    
    # define input/output pairs
    input_ = tf.placeholder(tf.float32, [None, sequence_len, batch_size])  # (batch, input, time)
    target = tf.placeholder(tf.float32, [None, batch_size])  # (batch, output)
    
    # parameters' initialization
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

    # extract train and test
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches(
                                                             filename='data/power_consumption.csv', 
                                                             window=sequence_len,
                                                             stride=stride,
                                                             mode='validation', 
                                                             non_train_percentage=.3,
                                                             val_rel_percentage=.8,
                                                             normalize='maxmin01',
                                                             time_difference=True,
                                                             td_method=np.log2)
       
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
        
                # run VAE encoding-decoding
                sess.run(optimizer_elbo, feed_dict={input_: batch_x})
                
                iter_ +=  1
                
        # test
        y_test = y_test[:x_test.shape[0]]
        mean_elbo = np.zeros(shape=vae_hidden_size)
        std_elbo = np.eye(vae_hidden_size)
        vae_anomalies = np.zeros(shape=(int(np.floor(x_test.shape[0] / batch_size))))
        
#        threshold_elbo = scistats.multivariate_normal.pdf(mean_elbo-sigma_threshold_elbo, mean_elbo, std_elbo)

        
        iter_ = 0
        
        while iter_ < int(np.floor(x_test.shape[0] / batch_size)):
    
            batch_x = x_test[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
            batch_y = y_test[np.newaxis, iter_*batch_size: (iter_+1)*batch_size]
                        
            # test if the VAE encoding is anomalous for the sample
            test_hidden_state = sess.run(vae_hidden_state, feed_dict={input_: batch_x})
            vae_anomalies[iter_] =  scistats.multivariate_normal.pdf(test_hidden_state, 
                                                                     np.zeros(shape=vae_hidden_size), 
                                                                     np.eye(vae_hidden_size))
                                       
            iter_ +=  1
            
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

    
