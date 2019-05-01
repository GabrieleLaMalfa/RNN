# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:27:05 2018

@author: Emanuele
"""

import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scistats  # ignore eventual warning, it is used (badly)
import sys as sys
import tensorflow as tf

sys.path.append('../../utils')
import utils_dataset as LSTM_exp
import best_fit_distribution as bfd


if __name__ == '__main__':

    DATA_PATH = '../../data/space_shuttle_marotta_valve.csv'
    
    window = 10
    stride = 1
    batch_size = 20
    l_rate = 3e-4    
    lstm_params = [150]
    lstm_activation = [tf.nn.tanh]
    
    # optimize over this vector the precision or F1-score
    sigma_threshold = [round(i*1e-4, 5) for i in range(1, 100, 5)]

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
    
    # define data for anomaly detection and plot
    plot_y = np.concatenate(results['Y']).ravel()
    plot_y_hat = np.concatenate(results['Y_HAT']).ravel()
    
    best_precision = best_recall = .0
    
    for t in sigma_threshold:
        
        # anomaly detection
        anomaly_threshold = (best_fitting_distr.ppf(t),
                             best_fitting_distr.ppf(1.-t))
    
        # turn test errors into a numpy array
        test_errors = np.concatenate(results['Test_Errors']).ravel()
    
        print("\nAnomalies detected with threshold: ", t)
        list_anomalies = list()
        
        for i in range(len(test_errors)):
    
            tmp = best_fitting_distr.cdf(test_errors[i])
            tmp = best_fitting_distr.ppf(tmp)
    
            # don't consider the last samples as anomalies since the logarithm as 
            #  time_difference method may 'corrupt' them (and there are NO anomalies there)
            if (tmp <= anomaly_threshold[0] or tmp >= anomaly_threshold[1]) and i<len(test_errors)-15:
    
                print("\tPoint number ", i, " is an anomaly: P(x) is ", best_fitting_distr.cdf(tmp))
                list_anomalies.append(i)
                
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
        
        # optimize precision wrt the thresholds
        try:

            precision = len(np.intersect1d(condition_positive, predicted_positive))/len(predicted_positive)
            recall = len(np.intersect1d(condition_positive, predicted_positive))/len(condition_positive)
        
        except ZeroDivisionError:
            
            precision = recall = 1e-5
        
        print("Precision and recall for threshold ", t, " are: ", (precision, recall), "\n")
        
        if precision >= best_precision:
            
            best_precision = precision
            best_recall = recall
            best_predicted_positive = cp.copy(predicted_positive)
    
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
    
    print("Anomalies: ", condition_positive.T)
    print("Anomalies Detected: ", best_predicted_positive.T)
    print("Precision: ", best_precision)
    print("Recall: ", best_recall)      
    
    # save sMAPE of each model
    sMAPE_error_len = len(np.array(results['Test_Errors']).ravel())
    sMAPE_den = np.abs(np.array(results['Y_HAT']).ravel()[:sMAPE_error_len])+np.abs(np.array(results['Y_test']).ravel()[:sMAPE_error_len])
    perc_error = np.mean(200*(np.abs(np.array(results['Test_Errors']).ravel()[:sMAPE_error_len]))/sMAPE_den)
    
    print("Percentage error: ", perc_error)
