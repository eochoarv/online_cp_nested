#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:45:12 2026

@author: eochoa
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from utils_online_cp import *
#from DtACI import *
#from sklearn.ensemble import RandomForestRegressor
from datasets import load_dataset
from matplotlib.cm import get_cmap
from collections import defaultdict
#import cvxpy as cp
from sklearn.isotonic import isotonic_regression
from utils_exps import *
from tqdm import tqdm




#data = load_dataset("AMZN")
#forecasts = np.load('test_log.npz')['forecasts']
#data['forecasts'] = forecasts
#data['scores'] = np.abs(forecasts - data['y'].to_numpy())
folder = 'results/inflation/'
#folder = 'results/synthetic/final/'
data = pd.read_csv(folder+'data.csv')

try:
    del data['Unnamed: 0']
except:
    print('data loaded')
    
scores = data['scores'].to_numpy()
alphas = np.linspace(0.01,.99, 99)
T = scores.shape[0]
vect_ones = np.ones(T)
etas = np.exp(np.linspace(-4, 3, 25))
window = 200


methods_dict = {
    "quant_tracking_multi_alpha": [
        quant_tracking_multi_alpha, 
        [
            {'lr': eta*vect_ones} for eta in etas
         
         ],
        ],
    "quant_tracking_simple_proj": [
        quant_tracking_simple_proj, 
        [
            {'lr': eta*vect_ones} for eta in etas
         
         ]
        ],
    #"quant_tracking_bounded": [
    #    quant_tracking_bounded, 
    #    [
    #        {'lr': eta} for eta in etas
    #     
    #     ]
    #    ],
    "quant_tracking_monotonic": [
        quant_tracking_monotonic, 
        [
            {'lr': eta*vect_ones} for eta in etas
         
         ]
        ],
    "quant_tracking_proj": [
        quant_tracking_proj, 
        [
            {'lr': eta*vect_ones} for eta in etas
         
         ]
        ],
    }
    
results = {}
df_1 = pd.DataFrame()
df_2 = pd.DataFrame()

for m in (methods_dict):
    
    method = methods_dict[m][0]
    method_params = methods_dict[m][1]
    df_cov = pd.DataFrame()
    df_set_size = pd.DataFrame()
    for mp in tqdm(method_params):
        result = method(
                scores,
                alphas,
                **mp
            )
        cov_df = table_cov(result, alphas, window).mean()
        df_cov = pd.concat(
            [df_cov, cov_df], axis=0
            )
        
        set_size_df = table_set_size(result, alphas, window).mean()
        
        df_set_size = pd.concat(
            [df_set_size, set_size_df], axis=0
            )
    df_cov['lr'] = etas
    df_set_size['lr'] = etas
    
    df_cov = df_cov.set_index('lr')
    df_set_size = df_set_size.set_index('lr')
    
    df_cov.columns = [m]
    df_set_size.columns = [m]
    
    df_1 = pd.concat(
        [df_1, df_cov], axis=1
        )
    
    df_2 = pd.concat(
        [df_2, df_set_size], axis=1
        )
    

#filename = 'results.json'

plt.plot(df_1)
plt.legend(df_1.columns)
plt.show()

plt.plot(df_2)
plt.legend(df_2.columns)
plt.show()

df_1.to_csv(folder+'table_coverage_optimal_learning_rate.csv')
df_2.to_csv(folder+'table_set_size_optimal_learning_rate.csv')

#with open(filename, 'w') as json_file:
#    json.dump(results, json_file, indent=4) # 'indent=4' makes the file human-readable

#print(f"Dictionary successfully saved to {filename}")