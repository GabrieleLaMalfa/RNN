# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:22:07 2018

@author: Gabriele
"""

import activations as act

import numpy as np

##### Derivatives of functions

def dtanh(a):
    
    res = np.multiply((1-a), (1+a))
    
    return res
    
def dsoftmax(a):
    
    res = act.softmax(a) * (1 - act.softmax(a))

    return res    