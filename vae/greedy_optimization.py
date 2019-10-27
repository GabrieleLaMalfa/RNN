# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:41:35 2019

@author: Emanuele
"""

import numpy as np
import tensorflow as tf
import sys

sys.path.append('../')
import experiments as vae_experiments

if __name__ == '__main__':
    
    # define the target dataset
    data_path = '../data/space_shuttle_marotta_valve.csv'
    
    # experiments per optimization's round
    total_exp = 2
    
    # define the optimization parameters' space
    SEQUENCE_LEN = [75, 100, 150]
    STRIDE = [1]
    ACTIVATION = [tf.nn.tanh]
    VAE_HIDDEN_SIZE = [2, 3, 5]
    L_RATE_ELBO = [1e-5]
    NORMALIZAITON = ['maxmin-11', 'maxmin01']
    
    PARAMETERS = [SEQUENCE_LEN, STRIDE, ACTIVATION, VAE_HIDDEN_SIZE, L_RATE_ELBO, NORMALIZAITON]
    
    # collect initial random (and best so far..) parameters
    params_seed = [np.random.randint(0, len(p)) for p in PARAMETERS]
    initial_params = [PARAMETERS[i][params_seed[i]] for i in range(len(PARAMETERS))]
    
    if initial_params[1] == 'half':
        
        initial_params[1] = max(1, int(initial_params[0]/2))
    
    elif initial_params[1] == 'window':
        
        initial_params[1] = initial_params[0]
      
    # greedy search for the best parameters
    total_precision = 0.
    total_recall = 0.
    prev_best_f1 = 0.
    
    is_not_first_round = False  # optimize all the params in the row, if this is the very first attempt
    target_score = lambda p, r: (p > .5)
    objective_reached = target_score(total_precision, total_recall)
      
    while (objective_reached is False):
    
        print("###########################")
        print("Best model has those parameters: ", initial_params)
        print("Precision and recall are: ", (total_precision, total_recall))
        print("###########################")
    
        # choose a random dimension and optimize over it
        opt_dim = np.random.randint(0, len(PARAMETERS))
        
        # save previous best parameter over the chosen dimension
        best_param_over_dim = initial_params[opt_dim]
                
        for p in PARAMETERS[opt_dim]:
            
            # skip optimization on the same parameters (except for the first round)
            if p == best_param_over_dim:
                
                if is_not_first_round:
                
                    continue  
                
                else:
                    
                    is_not_first_round = True
            
            # assign the new paramater and check if this configuration optimizes the precision
            new_params = initial_params.copy()
            new_params[opt_dim] = p
            
            print("\n")
            print("Starting optimization over dim ", opt_dim)
            print("Starting parameters ", new_params)
            print("Vector of optmization parameters ", PARAMETERS[opt_dim])
            print("Parameter considered at this iteration: [", p, "]")
            print("Previous best results (precision, recall) ", total_precision, total_recall)
            print("\n")
            
            if new_params[1] == 'half':
        
                new_params[1] = max(1, int(initial_params[0]/2))
    
            elif new_params[1] == 'window':
        
                new_params[1] = new_params[0]
            
            prev_best_precision = .0
            total_precision = 0.
            total_recall = 0.            
            n_successful_exp = 0
            n_ignored_experiments = 0
            best_threshold = []
        
            for i in range(total_exp):
                
                try:
                    
                    # optimize with the new parameters
                    tp, tr, b_tr =  vae_experiments.vae_experiment(data_path, *new_params)
                    total_precision += tp
                    total_recall += tr
                    best_threshold.append(b_tr)
                    n_successful_exp += 1
                
                except ZeroDivisionError:
                    
                    print("\n Experiment n° ", i," will be ignored!")
                    n_ignored_experiments += 1
                    pass
         
            try: 
                
                total_precision /= n_successful_exp
                total_recall /= n_successful_exp
                
                if target_score(total_precision, total_recall):
                    
                    objective_reached = True
                    break
                
                # prevent numerical errors with precision and recall
                if total_precision == 0.:
                    
                    total_precision = 1e-5
                
                if total_recall == 0.:
                    
                    total_recall = 1e-5
                     
                actual_f1 = 2*(total_precision * total_recall)/(total_precision + total_recall)
                if total_precision > prev_best_precision:
                    
                    initial_params[opt_dim] = p
                    prev_best_precision = total_precision 
                    prev_best_f1 = actual_f1
            
            except:
                
                print("\n All the experiments have failed, proceed with the next parameters \n")
                    
            
    ## Print out results
    print(n_ignored_experiments, " out of ", total_exp, " experiments has been ignored due to division error!")
    print(n_successful_exp, " experiments successfully executed.\nParameters of the model are:\n")
    print("- window: ", new_params[0])
    print("- stride: ", new_params[1])
    print("- activation: ", new_params[2])
    print("- vae hidden size: ", new_params[3])
    print("- hidden distr. degrees of freedom: ", new_params[4])
    print("- threshold: ", best_threshold)
    print("- learning rate: ", new_params[5])
    print("- normalization: ", new_params[6])
    print("Precision and recall of the model are: ", (total_precision, total_recall))
