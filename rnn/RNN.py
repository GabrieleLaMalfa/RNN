# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 09:06:22 2018

@author: Gabriele
"""
import derivatives as derv
import activations as actv


import numpy as np

################# RNN Forward and Backward Propagation ######################

activations_dictionary = {"tanh": actv.tanh , "softmax": actv.softmax}
derivatives_dictionary = {"dtanh": derv.dtanh , "dsoftmax": derv.dsoftmax}

class RNN(object):
    
    def __init__(self , )

