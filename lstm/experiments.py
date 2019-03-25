# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:27:05 2018

@author: Emanuele
"""

import numpy as np
import scipy.stats as scistats  # ignore eventual warning, it is used (badly)
import sys as sys
import tensorflow as tf

sys.path.append('../utils')
import utils_dataset as LSTM_exp
import best_fit_distribution as bfd


def lstm_experiment(data_path,
                    window,
                    stride,
                    batch_size,
                    lstm_params,
                    sigma_threshold,
                    l_rate,
                    activation,
                    normalization):
    
    lstm_params = [lstm_params]
    lstm_activation = [activation]
    non_train_percentage = 0.5
    training_epochs = 250
    val_rel_percentage = .5
    normalize = normalization
    time_difference = False
    td_method = None
    stop_on_growing_error = True
    stop_valid_percentage = 1.  # percentage of validation set used to stop learning

    results = LSTM_exp.lstm_exp(filename=data_path, 
                                window=window,
                                stride=stride,
                                batch_size=batch_size,
                                lstm_params=lstm_params,
                                lstm_activation=lstm_activation,
                                l_rate=l_rate, 
                                non_train_percentage=non_train_percentage, 
                                training_epochs=training_epochs,
                                l_rate_test=.05, 
                                val_rel_percentage=val_rel_percentage,
                                normalize=normalize,
                                time_difference=time_difference,
                                td_method=td_method,
                                stop_on_growing_error=stop_on_growing_error,
                                stop_valid_percentage=stop_valid_percentage,
                                verbose=False)

    # MLE on validation: estimate mean and variance
    val_errors = np.concatenate(results['Validation_Errors']).ravel()
    best_fitting, fitting_params, _ = bfd.best_fit_distribution(val_errors, top_n=1)
    best_fitting = 'scistats.' + best_fitting[0]
    fitting_params = fitting_params[0]
    
    best_fitting_distr = eval(best_fitting)(*fitting_params)   # 'non ne EVALe la pena' (italians only)
    
    # anomaly detection
    anomaly_threshold = (best_fitting_distr.ppf(sigma_threshold),
                         best_fitting_distr.ppf(1.-sigma_threshold))

    # turn test errors into a numpy array
    test_errors = np.concatenate(results['Test_Errors']).ravel()

    list_anomalies = list()
    
    for i in range(len(test_errors)):

        tmp = best_fitting_distr.cdf(test_errors[i])
        tmp = best_fitting_distr.ppf(tmp)

        # don't consider the last samples as anomalies since the logarithm as 
        #  time_difference method may 'corrupt' them (and there are NO anomalies there)
        if (tmp <= anomaly_threshold[0] or tmp >= anomaly_threshold[1]) and i<len(test_errors)-15:

            list_anomalies.append(i)

    
    # plot results
    plot_y = np.concatenate(results['Y']).ravel()

    # performances
    target_anomalies = np.zeros(shape=int(np.floor(plot_y.shape[0] / batch_size))*batch_size)
    
    # caveat: define the anomalies based on absolute position in test set (i.e. size matters!)
    # train 50%, validation_relative 50%
    target_anomalies[500:600] = 1
    
    # real values
    condition_positive = np.argwhere(target_anomalies == 1)
    
    # predictions
    predicted_positive = np.array([list_anomalies]).T
    
    # precision and recall
    precision = len(np.intersect1d(condition_positive, predicted_positive))/len(predicted_positive)
    recall = len(np.intersect1d(condition_positive, predicted_positive))/len(condition_positive)
    
    # save sMAPE of each model
    sMAPE_error_len = len(np.array(results['Test_Errors']).ravel())
    sMAPE_den = np.abs(np.array(results['Y_HAT']).ravel()[:sMAPE_error_len])+np.abs(np.array(results['Y_test']).ravel()[:sMAPE_error_len])
    perc_error = np.mean(200*(np.abs(np.array(results['Test_Errors']).ravel()[:sMAPE_error_len]))/sMAPE_den)
    
    return precision, recall, perc_error
