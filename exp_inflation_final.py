#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:46:16 2026

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
from utils_exps import *



folder = 'results/inflation/final/'
data = pd.read_csv('data/data_inflation.csv')
data = data.rename({'Unnamed: 0':'Date'}, axis=1).set_index('Date')

data.to_csv(folder+'data.csv')
scores = data['scores'].to_numpy()
K = 100
alphas = np.linspace(1/K,1-1/K,K-1)
#alphas = np.linspace(0.01,.99, 9)
pd.DataFrame({"alpha":alphas}).to_csv(folder+'alphas.csv')

eta = .5
rolling_window = 200
T = scores.shape[0]
vect_ones = np.ones(T)

methods_dict = {
    "quant_tracking_multi_alpha": [
        quant_tracking_multi_alpha, 
        {'lr':.8*vect_ones},
        ],
    "quant_tracking_simple_proj": [
        quant_tracking_simple_proj, 
        {'lr':.8*vect_ones}
        ],
    #"quant_tracking_bounded": [
    #    quant_tracking_bounded, 
    #    {'lr':.2}
    #    ],
    "quant_tracking_monotonic": [
        quant_tracking_monotonic, 
        {'lr':3.5*vect_ones}
        ],
    "quant_tracking_monotonic_fixed": [
        quant_tracking_monotonic_fixed, 
        {'lr':0.002*vect_ones}
        ],
    "quant_tracking_proj": [
        quant_tracking_proj, 
        {'lr':.8*vect_ones, 'eps':0}
        ],
    }

#methods_dict = {"quant_tracking_monotonic_fixed_"+str(i): [quant_tracking_monotonic_fixed, {'lr':eta*vect_ones}] 
#                for i, eta in enumerate([0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.05])}
    
results = {}
df = pd.DataFrame()
df_set_size = pd.DataFrame()


for m in methods_dict:
    
    method = methods_dict[m][0]
    method_params = methods_dict[m][1]
    result = method(
            scores,
            alphas,
            **method_params
        )
    cov_df = table_cov(result, alphas, rolling_window)
    df = pd.concat(
        [df, cov_df], axis=1
        )
    
    set_size_df = table_set_size(result, alphas, rolling_window)
    df_set_size = pd.concat(
        [df_set_size, set_size_df], axis=1
        )

    results[m] = result

filename = folder+'results.json'

df.to_csv(folder+'table_coverage.csv')
df_set_size.to_csv(folder+'table_set_size.csv')

with open(filename, 'w') as json_file:
    json.dump(results, json_file, indent=4) # 'indent=4' makes the file human-readable

print(f"Dictionary successfully saved to {filename}")