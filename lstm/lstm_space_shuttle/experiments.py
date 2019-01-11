# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:27:05 2018

@author: Gabriele
"""

import numpy as np
import matplotlib.pyplot as plt

import anomaly_detection as LSTM_exp


if __name__ == '__main__':

    DATA_PATH = 'space_shuttle_marotta_valve.csv'

    results = LSTM_exp.lstm_exp(filename=DATA_PATH, num_units=128, window=3, batch_size=5,
                                l_rate=1e-5, non_train_percentage=0.3, training_epochs=10,
                                l_rate_test=.05, val_rel_percentage=.8)

    # Anomaly detection
    anomaly_threshold = 5e-3  # /tau

    # MLE on validation: estimate mean and variance
    val_errors = np.concatenate(results['Validation_Errors']).ravel()
    mean = np.mean(val_errors)
    variance = np.var(val_errors)

    # turn test errors into a numpy array
    test_errors = np.concatenate(results['Test_Errors']).ravel()

    print("Anomalies detected with threshold: ", anomaly_threshold)
    list_anomalies = list()
    for i in range(len(test_errors)):

        tmp = LSTM_exp.gaussian_anomaly_detection(test_errors[i], mean, variance, anomaly_threshold)

        if tmp[0] is True:

            print("\tPoint number ", i, " is an anomaly: P(x) is ", tmp[1])
            list_anomalies.append(i)

    # plot results
    plot_y = np.concatenate(results['Y']).ravel()
    plot_y_hat = np.concatenate(results['Y_HAT']).ravel()
    fig, ax1 = plt.subplots()

    # plot data series
    ax1.plot(plot_y, 'b', label='index')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('TOPIX')

    # plot predictions
    ax1.plot(plot_y_hat, 'r', label='prediction')
    ax1.set_ylabel('Change Point')
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
