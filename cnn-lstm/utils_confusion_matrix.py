# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 08:53:54 2019

@author: Emanuele
"""

import numpy as np


def precision(filename, prediction, target):
    
    pass
    
    
def recall(filename, prediction, target):
    
    pass


def f1_score(filename, prediction, target):
    
    pass


def fall_out(filename, prediction, target):
    
    pass

condition_positive = np.argwhere(target_anomalies == 1)
condition_negative = np.argwhere(target_anomalies == 0)
predicted_positive = anomalies
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
