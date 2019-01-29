# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:18:22 2019

@author: Emanuele
"""

import numpy as np
import matplotlib.pyplot as plt


"""
 Calculate the autocorrelation function between a signal y(t) and istelf shifted
  towards the future, i.e. y(t+\delta), where delta is a variable you can set.
  Takes as input:
      data:numpy.array, array of size (n,);
      max_lag:integer, the max timestep you want to calculate correlation between data and its shifted version
       e.g. max_lag=0 means correlating the input with itself, with no shift, max_lag=1 means correlating
        the input with itself shifted in the future of 1 timestep, i.e. max_lag=k means correlating
        y(t) with y(t+k);
      scale:boolean, True if you want to scale the autocorrelation by the length of the time series
       considered (a longer time series may have larger correlation since it has more data)
      plot:boolean, True if you want to plot the autocorrelation function. 
  Returns:
      res:numpy.array, the autocorrelation vector whose size is (max_lag,)
"""
def autocorrelation_function(data, max_lag, plot=False, scale=True):
    
    ts_length = len(data)    
    autocorrelation = np.zeros(shape=(max_lag,))
    
    for i in range(max_lag):
        
        autocorrelation[i] = np.correlate(data[:ts_length-i], data[i:])/(ts_length-i)
        
    autocorrelation = autocorrelation.flatten()
        
    if plot is True:
        
        plt.plot(autocorrelation)
        
    return autocorrelation
