#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 15:31:24 2026

@author: eochoa
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from datasets import load_dataset
from utils_exps import *
    

#folder = 'results/synthetic/'
folder = 'results/inflation/final/'
forecasts = True

dict_methods = {
    'quant_tracking_multi_alpha': 'Quantile Tracker',
 'quant_tracking_simple_proj': 'Quantile Tracker Projected',
 'quant_tracking_monotonic_fixed': 'Exponentiated Gradient',
 #'quant_tracking_monotonic_fixed': 'Exponentiated Gradient Fixed',
 'quant_tracking_proj': 'Projected Gradient',
               }

with open(folder+'results.json', 'r') as file:
    results = json.load(file)
        
table_cov = pd.read_csv(folder+'table_coverage.csv')
del table_cov['Unnamed: 0']

table_set_size = pd.read_csv(folder+'table_set_size.csv')
del table_set_size['Unnamed: 0']

data = pd.read_csv(folder+'data.csv')

try:
    #del data['Unnamed: 0']
    data = data.rename({'Unnamed: 0':'Date'}, axis=1).set_index('Date')
    data.index = pd.to_datetime(data.index)
except:
    print('data loaded')

alphas = pd.read_csv(folder+'alphas.csv')
del alphas['Unnamed: 0']
alphas = alphas['alpha'].to_numpy()


for m in table_cov.columns:
    _ = plt.plot(alphas,
                 table_cov[m],
                 label=m)
plt.legend()
plt.show()


for m in table_set_size.columns:
    _ = plt.plot(alphas,
                 table_set_size[m],
                 label=m)
plt.legend()
plt.show()

for m in results:
    if m == "quant_tracking_monotonic":
        continue
    result = results[m]
    print(m)
    plot_dist_cov_rolling_mean(data, result, alphas, 120, label=dict_methods[m])

plt.legend()
filename = folder+'plot_rolling_cov_error.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()

for m in results:
    if m == "quant_tracking_monotonic":
        continue
    result = results[m]
    print(m)
    plot_dist_cov_mean(data, result, alphas, label=dict_methods[m])

plt.legend()
filename = folder+'plot_cov_error.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()

for m in results:
    filename = folder+'plot_pred_sets_'+m+'.png'
    result = results[m]
    print(m)
    #plot_dist_cov(result, alphas)
    if forecasts:
        plot_pred_sets(data, result, alphas, filename)