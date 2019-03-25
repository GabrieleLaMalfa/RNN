# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:27:05 2018

@author: Emanuele
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scistats  # ignore eventual warning, it is used (badly)
import sys as sys
import tensorflow as tf

sys.path.append('../../utils')
import utils_dataset as LSTM_exp
import best_fit_distribution as bfd


if __name__ == '__main__':

    DATA_PATH = '../../data/space_shuttle_marotta_valve.csv'
    
    window = 15
    stride = 3
    batch_size = 10
    sigma_threshold = 0.002  # n-th percentile, used for double tail test
    l_rate = 2e-4
    
    lstm_params = [100]
    lstm_activation = [tf.nn.tanh]
    non_train_percentage = 0.5
    training_epochs = 250
    val_rel_percentage = .5
    normalize = 'maxmin01'
    time_difference = False
    td_method = None
    stop_on_growing_error = True
    stop_valid_percentage = 1.  # percentage of validation set used to stop learning

    results = LSTM_exp.lstm_exp(filename=DATA_PATH, 
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
                                stop_valid_percentage=stop_valid_percentage)

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

    print("Anomalies detected with threshold: ", anomaly_threshold)
    list_anomalies = list()
    
    for i in range(len(test_errors)):

        tmp = best_fitting_distr.cdf(test_errors[i])
        tmp = best_fitting_distr.ppf(tmp)

        # don't consider the last samples as anomalies since the logarithm as 
        #  time_difference method may 'corrupt' them (and there are NO anomalies there)
        if (tmp <= anomaly_threshold[0] or tmp >= anomaly_threshold[1]) and i<len(test_errors)-15:

            print("\tPoint number ", i, " is an anomaly: P(x) is ", best_fitting_distr.cdf(tmp))
            list_anomalies.append(i)

    
    # plot results
    plot_y = np.concatenate(results['Y']).ravel()
    plot_y_hat = np.concatenate(results['Y_HAT']).ravel()
    
    fig, ax1 = plt.subplots()

    # plot data series
    print("\nPrediction:")
    ax1.plot(plot_y, 'b', label='index')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Space Shuttle')

    # plot predictions
    ax1.plot(plot_y_hat, 'r', label='prediction')
    ax1.set_ylabel('Prediction')
    plt.legend(loc='best')

    for i in list_anomalies:

        plt.axvspan(i, i+1, color='yellow', alpha=0.5, lw=0)

    fig.tight_layout()
    plt.show()
    
    print("Total test error:", np.sum(np.abs(results['Test_Errors'])))
    
    # plot reconstructed signal
    tot_y = 0.
    tot_y_hat = 0.
    recovered_plot_y = np.zeros(shape=len(plot_y_hat)+1)
    recovered_plot_y_hat = np.zeros(shape=len(plot_y_hat)+1)
    for i in range(1, len(recovered_plot_y)):
        
        recovered_plot_y[i] = tot_y
        recovered_plot_y_hat[i] = tot_y_hat
        
        tot_y += plot_y[i-1]
        tot_y_hat += plot_y_hat[i-1] 
                
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
    plt.hist(np.array(results['Test_Errors']).ravel(), bins=30) 

    # performances
    target_anomalies = np.zeros(shape=int(np.floor(plot_y.shape[0] / batch_size))*batch_size)
    
    # caveat: define the anomalies based on absolute position in test set (i.e. size matters!)
    # train 50%, validation_relative 50%
    target_anomalies[500:600] = 1
    
    # real values
    condition_positive = np.argwhere(target_anomalies == 1)
    condition_negative = np.argwhere(target_anomalies == 0)
    
    # predictions
    predicted_positive = np.array([list_anomalies]).T
    predicted_negative = np.setdiff1d(np.array([i for i in range(len(target_anomalies))]), 
                                      predicted_positive,
                                      assume_unique=True)
    
    # precision and recall
    precision = len(np.intersect1d(condition_positive, predicted_positive))/len(predicted_positive)
    recall = len(np.intersect1d(condition_positive, predicted_positive))/len(condition_positive)
    
    
    print("Anomalies: ", condition_positive.T)
    print("Anomalies Detected: ", predicted_positive.T)
    print("Precision: ", precision)
    print("Recall: ", recall)   
    
    # top-n distributions that fit the test errors.
    top_n = 10
    cols = [col for col in bfd.best_fit_distribution(np.array(results['Test_Errors']).ravel(), top_n=top_n)]
    top_n_distr = pd.DataFrame(cols, index=['NAME', 'PARAMS', 'ERRORS'])
    print("\n\nTop distributions: NAME ERRORS PARAM ", top_n_distr)
    
    file_ptr = np.loadtxt('../../__tmp/__tmp_res.csv', dtype=object)
    for i in range(top_n):
        
        file_ptr = np.append(file_ptr, top_n_distr[i]['NAME'])
    
    np.savetxt('../../__tmp/__tmp_res.csv', file_ptr, fmt='%s')
    
    
    # save sMAPE of each model
    sMAPE_error_len = len(np.array(results['Test_Errors']).ravel())
    sMAPE_den = np.abs(np.array(results['Y_HAT']).ravel()[:sMAPE_error_len])+np.abs(np.array(results['Y_test']).ravel()[:sMAPE_error_len])
    perc_error = np.mean(200*(np.abs(np.array(results['Test_Errors']).ravel()[:sMAPE_error_len]))/sMAPE_den)
    
    
    print("Percentage error: ", perc_error)
    
    file_ptr = np.loadtxt('../../__tmp/__tmp_err.csv', dtype=object)
    file_ptr = np.append(file_ptr, str(perc_error))
    np.savetxt('../../__tmp/__tmp_err.csv', file_ptr, fmt='%s')
    
