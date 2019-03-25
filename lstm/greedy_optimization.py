# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:41:35 2019

@author: Emanuele
"""

import numpy as np

import experiments as lstm_experiments

if __name__ == '__main__':
    
    # define the target dataset
    data_path = '../data/space_shuttle_marotta_valve.csv'
    
    # experiments per optimization's round
    total_exp = 20
    min_total_exp = 10  # number of experiments that cannot fail
    
    # define the optimization parameters' space
    WINDOW = [2, 3, 5, 10, 15]
    STRIDE = [1, 'half', 'window']
    BATCH = [3, 5, 7, 10, 15]
    LSTM_PARAMS = [50, 80]
    THRESHOLD = [5e-3, 1e-2, 5e-2]
    L_RATE = [5e-4, 1e-3]
    
    PARAMETERS = [WINDOW, STRIDE, BATCH, LSTM_PARAMS, THRESHOLD, L_RATE]
    
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
    total_sMAPE = 0.
    prev_best_f1 = 0.
    
    is_first_round = True  # optimize all the params in the row, if this is the very first attempt
    target_score = lambda p, r: (p >= .7 and r >= .1)
    objective_reached = target_score(total_precision, total_recall)
      
    while (objective_reached is False):
    
        # choose a random dimension and optimize over it
        opt_dim = np.random.randint(0, len(PARAMETERS))
        
        # save previous best parameter over the chosen dimension
        best_param_over_dim = initial_params[opt_dim]
                
        for p in PARAMETERS[opt_dim]:
            
            if p == best_param_over_dim and is_first_round:
                
                is_first_round = False
                continue
            
            # assign the new paramater and check if this configuration optimizes the F1-score
            new_params = initial_params.copy()
            new_params[opt_dim] = p
            
            print("\n")
            print("Starting optimization over dimension: ", opt_dim)
            print("Starting optimization with parameters: ", new_params)
            print("Vector of parameters that will be tried for optmization: ", PARAMETERS[opt_dim])
            print("Parameter, from previous vector, considered at this iteration: [", p, "]")
            print("Previous best results (precision, recall, sMAPE) ", total_precision, total_recall, total_sMAPE)
            print("\n")
            
            if new_params[1] == 'half':
        
                new_params[1] = max(1, int(initial_params[0]/2))
    
            elif new_params[1] == 'window':
        
                new_params[1] = new_params[0]
        
            total_precision = 0.
            total_recall = 0.            
            total_sMAPE = 0.
            n_successful_exp = 0
            n_ignored_experiments = 0
        
            for i in range(total_exp):
                
                if n_ignored_experiments > 10:
                    
                    break
                
                try:
                    
                    # optimize with the new parameters
                    tp, tr, ts = lstm_experiments.lstm_experiment(data_path, *new_params)
                    total_precision += tp
                    total_recall += tr
                    total_sMAPE += ts
                    n_successful_exp += 1
                
                except ZeroDivisionError:
                    
                    print("\n Experiment n°", i," will be ignored!")
                    n_ignored_experiments += 1
                    pass
         
            try: 
                
                total_precision /= n_successful_exp
                total_recall /= n_successful_exp
                total_sMAPE /= n_successful_exp
                
                if target_score(total_precision, total_recall):
                    
                    if (n_successful_exp >= min_total_exp):
                    
                        objective_reached = True
                        initial_params = new_params.copy()
                        break
                    
                    else:
                        
                        print("Precision and recall are fine but there's no enough experiments to confirm the hypotesis.")
                
                # prevent numerical errors with precision and recall
                if total_precision == 0.:
                    
                    total_precision = 1e-5
                
                if total_recall == 0.:
                    
                    total_recall = 1e-5
                    
                actual_f1 = 2*(total_precision * total_recall)/(total_precision + total_recall)
                
                if actual_f1 > prev_best_f1:
                    
                    initial_params[opt_dim] = p
                    prev_best_f1 = actual_f1  
            
            except:
                
                print("\n All the experiments have failed, proceed with the next parameters \n")
                    
            
    ## Print out results
    print(n_ignored_experiments, " out of ", total_exp, " experiments has been ignored due to division error!")
    print(n_successful_exp, " experiments successfully executed.\nParameters of the model are:\n")
    print("- window: ", initial_params[0])
    print("- stride: ", initial_params[1])
    print("- batch: ", initial_params[2])
    print("- lstm params: ", initial_params[3])
    print("- threshold: ", initial_params[4])
    print("- learning rate: ", initial_params[5])
    print("Error rate (sMAPE) is: ", total_sMAPE)
    print("Precision and recall of the model are: ", (total_precision, total_recall))
