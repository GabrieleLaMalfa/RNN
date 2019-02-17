# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:27:05 2018

@author: Emanuele
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scistats
import sys as sys

sys.path.append('../../utils')
import utils_dataset as LSTM_exp
import best_fit_distribution as bfd


if __name__ == '__main__':

    DATA_PATH = '../../data/sin.csv'
    num_units = 35
    window = 5
    stride = 3
    batch_size = 3
    l_rate = 1e-2
    non_train_percentage = .5
    training_epochs = 10
    val_rel_percentage = .5
    normalize = 'maxmin-11'
    time_difference = False
    td_method = None

    results = LSTM_exp.lstm_exp(filename=DATA_PATH, 
                                num_units=num_units, 
                                window=window,
                                stride=stride,
                                batch_size=batch_size,
                                l_rate=l_rate, 
                                non_train_percentage=non_train_percentage, 
                                training_epochs=training_epochs,
                                l_rate_test=.05, 
                                val_rel_percentage=val_rel_percentage,
                                normalize=normalize,
                                time_difference=time_difference,
                                td_method=td_method)

    # MLE on validation: estimate mean and variance
    val_errors = np.concatenate(results['Validation_Errors']).ravel()
    mean = np.mean(val_errors)
    std = np.std(val_errors)
    
    # Anomaly detection
    sigma_threshold = 5.  # /tau
    anomaly_threshold = scistats.norm.pdf(mean-sigma_threshold*std, mean, std)

    # turn test errors into a numpy array
    test_errors = np.concatenate(results['Test_Errors']).ravel()

    print("Anomalies detected with threshold: ", anomaly_threshold)
    list_anomalies = list()
    
    for i in range(len(test_errors)):

        tmp = scistats.norm.pdf(test_errors[i], mean, std)

        # don't consider the last samples as anomalies since the logarithm as 
        #  time_difference method may 'corrupt' them (and there are NO anomalies there)
        if tmp <= anomaly_threshold and i<len(test_errors)-15:

            print("\tPoint number ", i, " is an anomaly: P(x) is ", tmp)
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
    target_anomalies[860:909] = target_anomalies[1111:1140] = 1
    target_anomalies[1561:1599] = target_anomalies[1906:1945] = 1
    
    # real values
    condition_positive = np.argwhere(target_anomalies == 1)
    condition_negative = np.argwhere(target_anomalies == 0)
    
    # predictions
    predicted_positive = np.array([list_anomalies]).T
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
