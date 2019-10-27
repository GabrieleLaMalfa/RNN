# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 09:36:16 2019

@author: Emanuele
"""

import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scistats
import sys
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.append('../../utils')
import utils_dataset as utils


if __name__ == '__main__':
    
    # parameters of the model
    data_path = '../../data/space_shuttle_marotta_valve.csv'
    sequence_len = 45
    batch_size = 1
    stride = 10
    random_stride = False  # for each training epoch, use a random value of stride between 1 and stride
    vae_hidden_size = 1
    subsampling = 1
    elbo_importance = (.2, 1.)  # relative importance to reconstruction and divergence
    lambda_reg = (5e-3, 5e-3)  # elastic net 'lambdas', L1-L2
    rounding = None
    
    # maximize precision or F1-score over this vector
    sigma_threshold_elbo = [1e-2] # [i*1e-3 for i in range(1, 100, 10)]
    
    learning_rate_elbo = 1e-3
    vae_activation = tf.nn.relu
    normalization = 'maxmin-11'
    
    # training epochs
    epochs = 100
       
    # number of sampling per iteration in the VAE hidden layer
    samples_per_iter = 1
    
    # early-stopping parameters
    stop_on_growing_error = True
    stop_valid_percentage = .5  # percentage of validation used for early-stopping 
    min_loss_improvment = .005  # percentage of minimum loss' decrease (.01 is 1%)
    
    # reset computational graph
    tf.reset_default_graph()
    
    # create the computational graph
    with tf.device('/device:CPU:0'):
                
        # define input/output pairs
        input_ = tf.placeholder(tf.float32, [None, sequence_len, batch_size])  # (batch, input, time)
        
        # encoder/decoder parameters + initialization
        vae_encoder_shape_weights = [batch_size*sequence_len, 
                                     int(batch_size*sequence_len*.5), 
                                     vae_hidden_size*2]
        vae_decoder_shape_weights = [vae_hidden_size, 
                                     int(batch_size*sequence_len*.5), 
                                     batch_size*sequence_len]
        
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
        vae_encoder = vae_activation(vae_encoder)
        
        for (w_vae, b_vae) in zip(weights_vae_encoder[1:], bias_vae_encoder[1:]):
            
            vae_encoder = tf.matmul(vae_encoder, w_vae) + b_vae
            vae_encoder = vae_activation(vae_encoder)
        
        # means and variances' vectors of the learnt hidden distribution
        #  we assume the hidden gaussian's variances matrix is diagonal
        loc = tf.slice(vae_encoder, [0, 0], [-1, vae_hidden_size])
        loc = tf.squeeze(loc, axis=0)
        scale = tf.slice(tf.nn.softplus(vae_encoder), [0, vae_hidden_size], [-1, vae_hidden_size])
        scale = tf.squeeze(scale, 0) 
        
        """
        # the distribution is in charge of generating the hidden sample
        hidden_sample = tf.slice(vae_encoder, [0, 2*vae_hidden_size], [-1, vae_hidden_size])
        """
        
        # sample from the hidden ditribution
        vae_hidden_distr = tfp.distributions.MultivariateNormalDiag(loc, scale)
        
        # re-parametrization trick: sample from standard multivariate gaussian,
        #  multiply by std and add mean (from the input sample)
        prior = tfp.distributions.MultivariateNormalDiag(tf.zeros(vae_hidden_size),
                                                         tf.ones(vae_hidden_size))
        
        hidden_sample = prior.sample()*scale + loc
        
        # get probability of the hidden state
        vae_hidden_prob = prior.prob(hidden_sample)
        
        """
        # get probability of the hidden state
        s_ = vae_hidden_distr.sample(int(100e6))
        in_box = tf.cast(tf.reduce_all(s_ <= hidden_sample, axis=-1), prior.dtype)
        vae_hidden_prob = tf.reduce_mean(in_box, axis=0)
        """
            
        feed_decoder = tf.reshape(hidden_sample, shape=(-1, vae_hidden_size))
        vae_decoder = tf.matmul(feed_decoder, weights_vae_decoder[0]) + bias_vae_decoder[0]
        vae_decoder = vae_activation(vae_decoder)    
        
        for (w_vae, b_vae) in zip(weights_vae_decoder[1:], bias_vae_decoder[1:]):
            
            vae_decoder = tf.matmul(vae_decoder, w_vae) + b_vae
            vae_decoder = vae_activation(vae_decoder)
        
        # time-series reconstruction and ELBO loss
        vae_reconstruction = tfp.distributions.MultivariateNormalDiag(tf.constant(np.zeros(batch_size*sequence_len, dtype='float32')),
                                                                      tf.constant(np.ones(batch_size*sequence_len, dtype='float32')))
            
        likelihood = elbo_importance[0]*vae_reconstruction.log_prob(vae_decoder)            
        divergence = elbo_importance[1]*tfp.distributions.kl_divergence(prior, vae_hidden_distr)        
        elbo = tf.reduce_mean(likelihood - divergence)
        
        # apply elastic net regularization (lambda_reg is the couple parameter that controls L1-L2 combination)
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=lambda_reg[0], scope=None)
        nn_params = tf.trainable_variables() # all vars of your graph
        l1_regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, nn_params)
        
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=lambda_reg[1], scope=None)
        l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, nn_params)
        
        regularized_elbo = -elbo + l1_regularization_penalty + l2_regularization_penalty
            
        optimizer_elbo = tf.train.AdamOptimizer(learning_rate_elbo).minimize(regularized_elbo)
        
    if random_stride == False:
        
        # extract train and test
        x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches(
                                                                 filename=data_path, 
                                                                 window=sequence_len,
                                                                 stride=stride,
                                                                 mode='strided-validation', 
                                                                 non_train_percentage=.5,
                                                                 val_rel_percentage=.5,
                                                                 normalize=normalization,
                                                                 time_difference=False,
                                                                 td_method=None,
                                                                 subsampling=subsampling,
                                                                 rounding=rounding)
       
        # suppress second axis on Y values (the algorithms expects shapes like (n,) for the prediction)
        y_train = y_train[:,0]; y_valid = y_valid[:,0]; y_test = y_test[:,0]
        
        if len(x_train) > len(y_train):
            
            x_train = x_train[:len(y_train)]
        
        if len(x_valid) > len(y_valid):
            
            x_valid = x_valid[:len(y_valid)]
        
        if len(x_test) > len(y_test):
            
            x_test = x_test[:len(y_test)]
        
        

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                          log_device_placement=True)) as sess:
        
        # create dataset with random stride
        # extract train and test
        if random_stride == True: 
            
            x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches(
                                                                     filename=data_path, 
                                                                     window=sequence_len,
                                                                     stride=np.random.randint(1, stride),
                                                                     mode='strided-validation', 
                                                                     non_train_percentage=.5,
                                                                     val_rel_percentage=.5,
                                                                     normalize=normalization,
                                                                     time_difference=False,
                                                                     td_method=None,
                                                                     subsampling=subsampling,
                                                                     rounding=rounding)
           
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
        
        sess.run(init)
        tf.random.set_random_seed(42)
        
        # train
        last_error_on_valid = np.inf
        current_error_on_valid = .0
        e = 0
        
        while e < epochs:
                
            print("epoch ", e)
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
                    
                print("Previous error on valid ", last_error_on_valid)
                print("Current error on valid ", current_error_on_valid)
                                 
                # stop learning if the loss reduction is below the threshold (current_loss/past_loss)
                if current_error_on_valid > last_error_on_valid or (np.abs(current_error_on_valid/last_error_on_valid) > 1-min_loss_improvment and e!=0):
                    
                    print("Early stopping: validation error has increased since last epoch.")
                    e = epochs
                        
                last_error_on_valid = current_error_on_valid
                
            e += 1
            
        # anomaly detection on test set
        y_test = y_test[:x_test.shape[0]]
        
        # find the thershold that maximizes the F1-score
        best_precision = best_recall = best_threshold = .0
        best_predicted_positive = np.array([])
        condition_positive = np.array([])
        
        for t in sigma_threshold_elbo:
            
            print("Optimizing with threshold's value: ", t)
            
            vae_anomalies = []
            p_anom = np.zeros(shape=(int(np.floor(x_test.shape[0] / batch_size)),))
            threshold_elbo = (t, 1.-t)            
            iter_ = 0
            
            while iter_ < int(np.floor(x_test.shape[0] / batch_size)):
                        
                batch_x = x_test[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                            
                # get probability of the encoding and a boolean (anomaly or not)        
                p_anom[iter_] = sess.run(vae_hidden_prob, feed_dict={input_: batch_x})  
                
                """
                # highlight anomalies   (the whole window is considered)                 
                if (p_anom[iter_] <= threshold_elbo[0] and iter_<int(np.floor(x_test.shape[0] / batch_size))-sequence_len):
                    
                    for i in range(iter_, iter_+sequence_len):
                        
                        vae_anomalies.append(i)
                """
                                           
                iter_ +=  1
                
            # predictions
            predicted_positive = np.array([vae_anomalies]).T
            
            if len(vae_anomalies) == 0:
                
                continue
                
            # caveat: define the anomalies based on absolute position in test set (i.e. size matters!)
            # train 50%, validation_relative 50%
            # performances
            target_anomalies = np.zeros(shape=int(np.floor(y_test.shape[0] / batch_size))*batch_size)
            target_anomalies[512:535] = 1
        
            # real values
            condition_positive = np.argwhere(target_anomalies == 1)
        
            # precision and recall
            try:
                
                precision = len(np.intersect1d(condition_positive, predicted_positive))/len(predicted_positive)
                recall = len(np.intersect1d(condition_positive, predicted_positive))/len(condition_positive)
            
            except ZeroDivisionError:
                
                precision = recall = .0
                
            print("Precision and recall for threshold: ", t, " is ", (precision, recall))
            
            if precision >= best_precision:
                
                best_threshold = t
                best_precision = precision
                best_recall = recall
                best_predicted_positive = cp.copy(predicted_positive)
                
        # plot data series    
        fig = plt.figure()
        
        ax1 = plt.subplot(211)
        print("\nTime series:")
        ax1.plot(y_test, 'b', label='index')
        bbox_props = dict(boxstyle="square", fc="white", ec="black", lw=1)
        ax1.annotate('Space Shuttle Marotta Anomaly', xy=(1700, 13.5), xytext=(1700, 13.5),
            arrowprops=dict(facecolor='black', width = 0.02, headwidth = 2, headlength = 3, shrink=0.1), bbox=bbox_props, size = 6)

        #ax1.set_xlabel('Time')
        ax1.set_ylabel('Space Shuttle Marotta')
        
        # plot predictions
        for i in vae_anomalies:
    
            plt.axvspan(i, i+1, color='green', alpha=0.7, lw=0)
            
        for j in target_anomalies:
            plt.axvspan(j, color='orange', alpha=0.3, lw=0)
        fig.tight_layout()
        plt.show()
        
        
        ax2 = plt.subplot(212)
        ax2.scatter([i for i in range(len(p_anom))],p_anom)
        #ax2.set_xlabel('Time')
        ax2.set_ylabel('Likelihood')
                
        fig.tight_layout()
        plt.show()
                
        print("Anomalies in the series:", condition_positive.T)
        print("Anomalies detected with threshold: ", best_threshold)
        print(best_predicted_positive.T)
        print("Precision and recall ", best_precision, best_recall)