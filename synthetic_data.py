#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:32:05 2026

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



folder = 'results/synthetic/'
T = 10000
alphas = np.linspace(0.01,.99,99)
#scale = .1*((np.cos(np.linspace(0,3*np.pi, 10000)))**2 * np.linspace(0.1, 5, 10000))
scale = np.linspace(.001, 4, T)**2
#data, _ = gen_synthetic(T, folder, scale, alphas)#pd.read_csv('data/synthetic_data.csv')
data, _ = gen_synthetic_beta(T, folder, alphas)
#del data['Unnamed: 0']

scores = data['scores'].to_numpy()

pd.DataFrame({"alpha":alphas}).to_csv(folder+'alphas.csv')
lr = {
    "quant_tracking_multi_alpha": .1*np.ones(scores.shape[0]),#.1*np.ones(scores.shape[0]),
    "quant_tracking_simple_proj":.1*np.ones(scores.shape[0]),# np.linspace(.01, 0.5, scores.shape[0]),
    "quant_tracking_monotonic": .05*np.ones(scores.shape[0]),#np.linspace(1, 3, scores.shape[0]),
    "quant_tracking_proj": .1*np.ones(scores.shape[0])#np.linspace(.01, 0.5, scores.shape[0])
    }
T_burnin = 100
ahead = 1

methods = {
    "quant_tracking_multi_alpha":quant_tracking_multi_alpha,
    "quant_tracking_simple_proj":quant_tracking_simple_proj,
    "quant_tracking_monotonic":quant_tracking_monotonic,
    "quant_tracking_proj":quant_tracking_proj
    }

method_param = {'B_var':True, 'B':np.max(scores)}

results = {}
df = pd.DataFrame()
df_set_size = pd.DataFrame()
rolling_window = 200
for x in methods:
    
    method = methods[x]
    result = method(
            scores,
            alphas,
            lr[x],
            **method_param
            #ahead
        )
    cov_df = table_cov(result, alphas, rolling_window)
    df = pd.concat(
        [df, cov_df], axis=1
        )
    
    set_size_df = table_set_size(result, alphas, rolling_window)
    df_set_size = pd.concat(
        [df_set_size, set_size_df], axis=1
        )
    
    results[x] = result

filename = folder+'results.json'

df.to_csv(folder+'table_coverage.csv')
df_set_size.to_csv(folder+'table_set_size.csv')

with open(filename, 'w') as json_file:
    json.dump(results, json_file, indent=4) # 'indent=4' makes the file human-readable

print(f"Dictionary successfully saved to {filename}")