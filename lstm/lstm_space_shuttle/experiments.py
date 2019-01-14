# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:27:05 2018

@author: Gabriele
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scistats

import anomaly_detection as LSTM_exp


if __name__ == '__main__':

    DATA_PATH = 'space_shuttle_marotta_valve.csv'

    results = LSTM_exp.lstm_exp(filename=DATA_PATH, 
                                num_units=32, 
                                window=3,
                                stride=3,
                                batch_size=1,
                                l_rate=2e-3, 
                                non_train_percentage=0.5, 
                                training_epochs=25,
                                l_rate_test=.05, 
                                val_rel_percentage=.5,
                                normalize=True,
                                time_difference=True,
                                td_method=None)

    # MLE on validation: estimate mean and variance
    val_errors = np.concatenate(results['Validation_Errors']).ravel()
    mean = np.mean(val_errors)
    std = np.std(val_errors)
    
    # Anomaly detection
    sigma_threshold = 2.  # /tau
    anomaly_threshold = scistats.norm.pdf(mean-sigma_threshold*std, mean, std)

    # turn test errors into a numpy array
    test_errors = np.concatenate(results['Test_Errors']).ravel()

    print("Anomalies detected with threshold: ", anomaly_threshold)
    list_anomalies = list()
    
    for i in range(len(test_errors)):

        tmp = scistats.norm.pdf(test_errors[i], mean, std)

        if tmp <= anomaly_threshold:

            print("\tPoint number ", i, " is an anomaly: P(x) is ", tmp)
            list_anomalies.append(i)

    # plot results
    plot_y = np.concatenate(results['Y']).ravel()
    plot_y_hat = np.concatenate(results['Y_HAT']).ravel()
    fig, ax1 = plt.subplots()

    # plot data series
    ax1.plot(plot_y, 'b', label='index')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Space Shuttle')

    # plot predictions
    ax1.plot(plot_y_hat, 'r', label='prediction')
    ax1.set_ylabel('Prediction')
    plt.legend(loc='best')

#    # plot anomaly's likelihood
#    ax1.stem(range(len(test_errors)), test_errors, markerfmt=' ')
#    ax1.set_ylabel("Anomaly's Likelihood")
#    plt.legend(loc='best')

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
    ax1.plot(recovered_plot_y, 'b', label='index')
    ax1.set_xlabel('RECONSTRUCTION: Date')
    ax1.set_ylabel('Space Shuttle')

    # plot predictions
    ax1.plot(recovered_plot_y_hat, 'r', label='prediction')
    ax1.set_ylabel('RECONSTRUCTION: Prediction')
    plt.legend(loc='best')

    fig.tight_layout()
    plt.show()
    
