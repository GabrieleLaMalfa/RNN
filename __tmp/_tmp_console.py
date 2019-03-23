# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:41:35 2019

@author: Emanuele
"""

total_precision = 0.
total_recall = 0.
total_sMAPE = 0.
n_exp = 0

for _ in range(10):
    
    try:
        runfile('C:/Users/Emanuele/Desktop/Github/RNN/lstm/lstm_space_shuttle/experiments.py', wdir='C:/Users/Emanuele/Desktop/Github/RNN/lstm/lstm_space_shuttle')
        total_precision += precision
        total_recall += recall
        total_sMAPE += perc_error
        n_exp += 1
    
    except ZeroDivisionError:
        
        pass
    
total_precision /= n_exp
total_recall /= n_exp
total_sMAPE /= n_exp

## Print out results
print(n_exp, " experiments successfully executed.\nParameters of the model are:\n")
print("- window: ", window)
print("- stride: ", stride)
print("- batch: ", batch_size)
print("- lstm hidden units: ", lstm_params)
print("- threshold: ", anomaly_threshold)
print("- learning rate: ", l_rate)
print("Error rate (sMAPE) is: ", total_sMAPE)
print("Precision and recall of the model are: ", (precision, recall))