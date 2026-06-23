#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:57:27 2026

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



data = pd.read_csv('data/data_inflation.csv')
del data['Unnamed: 0']

data.to_csv('results/inflation/data.csv')


scores = data['scores'].to_numpy()
alphas = np.linspace(0.01,.99, 99)
pd.DataFrame({"alpha":alphas}).to_csv('results/inflation/alphas.csv')
eta = .5
rolling_window = 200

methods_dict = {
    "quant_tracking_multi_alpha": [
        quant_tracking_multi_alpha, 
        {'lr':.14},
        ],
    "quant_tracking_simple_proj": [
        quant_tracking_simple_proj, 
        {'lr':.14}
        ],
    #"quant_tracking_bounded": [
    #    quant_tracking_bounded, 
    #    {'lr':.2}
    #    ],
    "quant_tracking_monotonic": [
        quant_tracking_monotonic, 
        {'lr':3.4}
        ],
#    "quant_tracking_monotonic_2": [
#        quant_tracking_monotonic, 
#        {'lr':eta, 'delta':0.0005}
#        ],
    "quant_tracking_proj": [
        quant_tracking_proj, 
        {'lr':.14, 'eps':0.0001}
        ],
    }

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

filename = 'results/inflation/results.json'

df.to_csv('results/inflation/table_coverage.csv')
df_set_size.to_csv('results/inflation/table_set_size.csv')

with open(filename, 'w') as json_file:
    json.dump(results, json_file, indent=4) # 'indent=4' makes the file human-readable

print(f"Dictionary successfully saved to {filename}")