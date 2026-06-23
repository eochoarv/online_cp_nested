#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:25:22 2026

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

data = pd.read_csv('elec2.csv')
del data['Unnamed: 0']


scores = data['scores'].to_numpy()[10:]
alphas = np.linspace(0.01,.99,100)
lr = .1
T_burnin = 100
ahead = 1

methods = {
    "quant_tracking_multi_alpha":quant_tracking_multi_alpha,
    "quant_tracking_bounded":quant_tracking_bounded,
    "quant_tracking_monotonic":quant_tracking_monotonic,
    "quant_tracking_proj":quant_tracking_proj
    }

results = {}
df = pd.DataFrame()
for x in methods:
    
    method = methods[x]
    result = method(
            scores,
            alphas,
            lr,
            #ahead
        )
    cov_df = table_cov(result, alphas, 200)
    df = pd.concat(
        [df, cov_df], axis=1
        )
    
    results[x] = result

filename = 'results_elec2.json'

df.to_csv('table_coverage_elec2.csv')

with open(filename, 'w') as json_file:
    json.dump(results, json_file, indent=4) # 'indent=4' makes the file human-readable

print(f"Dictionary successfully saved to {filename}")