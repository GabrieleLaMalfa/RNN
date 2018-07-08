# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 09:06:22 2018

@author: Gabriele
"""
import derivatives as derv
import activations as actv


import numpy as np

################# RNN class ######################

activations_dictionary = {"tanh": actv.tanh , "softmax": actv.softmax}
derivatives_dictionary = {"dtanh": derv.dtanh , "dsoftmax": derv.dsoftmax}

class RNN(object):
    
    def __init__(self , seq_length , input_size , output_size , a_size , 
                 activations , out_activations , loss , k = None):
        
        self.seq_length = seq_length
        self.input_size = input_size
        self.output_size = output_size
        self.a_size = a_size
        
        if type(activations) is str:
            self.activations = [activations for _ in range (self.seq_length)]
        if type(out_activations) is str:
            self.out_activations = [out_activations for _ in range(self.seq_length)]
        
        
        self.weights_x = np.zeros((a_size , input_size))
        self.weights_a = np.zeros((a_size , (a_size if k is None else k)))
        self.weights_y = np.zeros((output_size , a_size))
        self.bias_x = np.zeros((a_size , 1))
        self.bias_a = np.zeros((a_size , 1))
        self.bias_y = np.zeros((output_size , 1))
        self.current_activation = np.zeros((a_size, 1))
    
        
    def forward(self , x):
        print("aaaaa")
        predict = list()
        for i in range(self.seq_length):
            
            z = np.dot(self.weights_x , x[:,i,np.newaxis]) + self.bias_x
            z += np.dot(self.weights_a , self.current_activation) + self.bias_a  
            act_i = activations_dictionary[self.activations[i]](z)
            print(act_i.shape)
            
            g = (np.dot(self.weights_y , act_i) + self.bias_y)            
            predict.append(activations_dictionary[self.out_activations[i]](g))

            self.current_activation = act_i
            
        return predict
    
"""test"""
x = np.random.rand(5,3)

net = RNN(3, 5, 30, 4, 'tanh', 'softmax', 'L2')

res = net.forward(x)
        
        
