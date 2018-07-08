# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:04:45 2018

@author: Gabriele
"""

import numpy as np

####### Activation functions

def tanh(a):
    
    res = (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))

    return res 

def softmax(a):
    
    res = (np.exp(a)) / np.sum(np.exp(a) , axis = 0)

    return res      
    