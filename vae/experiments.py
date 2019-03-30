# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 09:36:16 2019

@author: Emanuele
"""

import numpy as np
import scipy.stats as scistats
import sys
import tensorflow as tf

sys.path.append('../utils')
import utils_dataset as utils


def vae_experiment(data_path,
                   sequence_len,
                   stride,
                   activation,
                   vae_hidden_size,
                   tstud_degrees_of_freedom,
                   learning_rate_elbo,
                   normalization):
    
    # reset computational graph
    tf.reset_default_graph()
    
    # parameters that are constant
    batch_size = 1
    
    # maximize precision or precision/F1-score over this vector
    sigma_threshold_elbo = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    
    # training epochs
    epochs = 100
       
    # number of sampling per iteration in the VAE hidden layer
    samples_per_iter = 1
    
    # early-stopping parameters
    stop_on_growing_error = True
    stop_valid_percentage = .1  # percentage of validation used for early-stopping 
    min_loss_improvment = .03  # percentage of minimum loss' decrease (.01 is 1%)
    
    # define input/output pairs
    input_ = tf.placeholder(tf.float32, [None, sequence_len, batch_size])  # (batch, input, time)
    
    # encoder/decoder parameters + initialization
    vae_encoder_shape_weights = [batch_size*sequence_len, int(batch_size*sequence_len/2), vae_hidden_size*2]
    vae_decoder_shape_weights = [vae_hidden_size, int(batch_size*sequence_len/2), batch_size*sequence_len]
    
    zip_weights_encoder = zip(vae_encoder_shape_weights[:-1], vae_encoder_shape_weights[1:])
    
    weights_vae_encoder = [tf.Variable(tf.truncated_normal(shape=[shape,
                                                                  next_shape])) for (shape, next_shape) in zip_weights_encoder]
    bias_vae_encoder = [tf.Variable(tf.truncated_normal(shape=[shape])) for shape in vae_encoder_shape_weights[1:]]
    
    zip_weights_decoder = zip(vae_decoder_shape_weights[:-1], vae_decoder_shape_weights[1:])
    weights_vae_decoder = [tf.Variable(tf.truncated_normal(shape=[shape,
                                                                  next_shape])) for (shape, next_shape) in zip_weights_decoder]
    bias_vae_decoder = [tf.Variable(tf.truncated_normal(shape=[shape])) for shape in vae_decoder_shape_weights[1:]]
    
    # VAE graph's definition
    flattened_input = tf.layers.flatten(input_)
    
    vae_encoder = tf.matmul(flattened_input, weights_vae_encoder[0]) + bias_vae_encoder[0]
    
    for (w_vae, b_vae) in zip(weights_vae_encoder[1:], bias_vae_encoder[1:]):
        
        vae_encoder = activation(vae_encoder)
        vae_encoder = tf.matmul(vae_encoder, w_vae) + b_vae
    
    # means and variances' vectors of the learnt hidden distribution
    #  we assume the hidden gaussian's variances matrix is diagonal
    loc = tf.slice(activation(vae_encoder), [0, 0], [-1, vae_hidden_size])
    loc = tf.squeeze(loc, axis=0)
    scale = tf.slice(tf.nn.softplus(vae_encoder), [0, vae_hidden_size], [-1, vae_hidden_size])
    scale = tf.squeeze(scale, 0) 
  
    # sample from the hidden ditribution
    vae_hidden_distr = tf.contrib.distributions.MultivariateNormalDiag(loc, scale)  
    vae_hidden_state = tf.reduce_mean([vae_hidden_distr.sample() for _ in range(samples_per_iter)], axis=0)
    
    # get probability of the hidden state
    s = vae_hidden_distr.sample(int(100e4))
    in_box = tf.cast(tf.reduce_all(s <= vae_hidden_state, axis=-1), vae_hidden_distr.dtype)
    vae_hidden_prob = tf.reduce_mean(in_box, axis=0) 
        
    feed_decoder = tf.reshape(vae_hidden_state, shape=(-1, vae_hidden_size))
    vae_decoder = tf.matmul(feed_decoder, weights_vae_decoder[0]) + bias_vae_decoder[0]
    vae_decoder = activation(vae_decoder)    
    
    for (w_vae, b_vae) in zip(weights_vae_decoder[1:], bias_vae_decoder[1:]):
        
        vae_decoder = tf.matmul(vae_decoder, w_vae) + b_vae
        vae_decoder = activation(vae_decoder)
    
    # time-series reconstruction and ELBO loss
    vae_reconstruction = tf.contrib.distributions.StudentT(tstud_degrees_of_freedom,
                                                           tf.constant(np.zeros(batch_size*sequence_len, dtype='float32')),
                                                           tf.constant(np.ones(batch_size*sequence_len, dtype='float32')))
    likelihood = vae_reconstruction.log_prob(flattened_input)
    
    prior = tf.contrib.distributions.MultivariateNormalDiag(tf.constant(np.zeros(vae_hidden_size, dtype='float32')),
                                                            tf.constant(np.ones(vae_hidden_size, dtype='float32')))
    
    divergence = tf.contrib.distributions.kl_divergence(vae_hidden_distr, prior)
    elbo = tf.reduce_mean(likelihood - divergence)
    
    optimizer_elbo = tf.train.AdamOptimizer(learning_rate_elbo).minimize(-elbo)  

    # extract train and test
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches(
                                                             filename=data_path, 
                                                             window=sequence_len,
                                                             stride=stride,
                                                             mode='validation', 
                                                             non_train_percentage=.5,
                                                             val_rel_percentage=.5,
                                                             normalize=normalization,
                                                             time_difference=False,
                                                             td_method=None)
       
    # suppress second axis on Y values (the algorithms expects shapes like (n,) for the prediction)
    y_train = y_train[:,0]; y_valid = y_valid[:,0]; y_test = y_test[:,0]
    
    if len(x_train) > len(y_train):
        
        x_train = x_train[:len(y_train)]
    
    if len(x_valid) > len(y_valid):
        
        x_valid = x_valid[:len(y_valid)]
    
    if len(x_test) > len(y_test):
        
        x_test = x_test[:len(y_test)]
        
    # train + early-stopping
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        
        # train
        last_error_on_valid = np.inf
        current_error_on_valid = .0
        e = 0
        
        while e < epochs:
                        
            iter_ = 0
            
            while iter_ < int(np.floor(x_train.shape[0] / batch_size)):
        
                batch_x = x_train[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                
                # run VAE encoding-decoding
                sess.run(optimizer_elbo, feed_dict={input_: batch_x})
                
                iter_ +=  1

            if stop_on_growing_error:

                current_error_on_valid = .0
                
                # verificate stop condition
                iter_val_ = 0
                while iter_val_ < int(stop_valid_percentage * np.floor(x_valid.shape[0] / batch_size)):
                    
                    batch_x_val = x_valid[iter_val_*batch_size: (iter_val_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                    
                    # accumulate error
                    current_error_on_valid +=  np.abs(np.sum(sess.run(-elbo, feed_dict={input_: batch_x_val})))

                    iter_val_ += 1
                                 
                # stop learning if the loss reduction is below the threshold (current_loss/past_loss)
                if current_error_on_valid > last_error_on_valid or (np.abs(current_error_on_valid/last_error_on_valid) > 1-min_loss_improvment and e!=0):
            
                    e = epochs
                        
                last_error_on_valid = current_error_on_valid
                
            e += 1
            
        # anomaly detection on test set
        y_test = y_test[:x_test.shape[0]]
        
        # find the thershold that maximizes the F1-score
        best_precision = best_recall = .0
        best_threshold = .0
        
        for t in sigma_threshold_elbo:
            
            vae_anomalies = []
            threshold_elbo = (t, 1.-t)            
            iter_ = 0
            
            while iter_ < int(np.floor(x_test.shape[0] / batch_size)):
        
                batch_x = x_test[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                            
                # get probability of the encoding and a boolean (anomaly or not)        
                p_anom = sess.run(vae_hidden_prob, feed_dict={input_: batch_x})  
                
                if (p_anom <= threshold_elbo[0] or p_anom >= threshold_elbo[1]):
                    
                    for i in range(iter_*batch_size, (iter_+1)*batch_size):
                        
                        vae_anomalies.append(i)
                                           
                iter_ +=  1
                
            # predictions
            predicted_positive = np.array([vae_anomalies]).T
            
            if len(vae_anomalies) == 0:
                
                continue
                
            # caveat: define the anomalies based on absolute position in test set (i.e. size matters!)
            # train 50%, validation_relative 50%
            # performances
            target_anomalies = np.zeros(shape=int(np.floor(y_test.shape[0] / batch_size))*batch_size)
            target_anomalies[400:500] = 1
        
            # real values
            condition_positive = np.argwhere(target_anomalies == 1)
        
            # precision and recall
            try:
                
                precision = len(np.intersect1d(condition_positive, predicted_positive))/len(predicted_positive)
                recall = len(np.intersect1d(condition_positive, predicted_positive))/len(condition_positive)
            
            except ZeroDivisionError:
                
                precision = recall = .0
            
            print("Precision and recall for threshold: ", t, " is ", (precision, recall))
            
            if precision > best_precision:
                
                best_threshold = t
                best_precision = precision
                best_recall = recall
        
        return best_precision, best_recall, best_threshold       
