#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:01:50 2026

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
from tqdm import tqdm
from scipy.stats import gamma 
from scipy.stats import beta 
from brownian_motion import *


# dict_param = {
#    'T':50000,
#    'N_sim':10,
#    'K':10,
#    'rolling_window':10000,
#    'folder':'results/synthetic/final/',
#    'x0':5, 
#    'lower':1, 
#    'upper':10, 
#    'step_std':.025, 
#    'w':1,
# }

with open('dict_param_synthetic.json', 'r', encoding='utf-8') as file:
    dict_param = json.load(file)

folder = dict_param['folder']
T = dict_param['T']
N_sim = dict_param['N_sim']
K = dict_param['K']
rolling_window = dict_param['rolling_window']

x0 = dict_param['x0']
lower = dict_param['lower'] 
upper = dict_param['upper']
step_std = dict_param['step_std']
w = dict_param['w']

alphas = np.linspace(1/K,1-1/K,K-1)
rng = np.random.default_rng(0)

dict_gen_data = {
 'T':T, 
 'folder':folder, 
 'alphas':alphas, 
 'x0':x0, 
 'lower':lower, 
 'upper':upper, 
 'step_std':step_std, 
 'w':w, 
 'rng':rng
 }

filename = folder+'dict_param.json'
with open(filename, 'w') as json_file:
    json.dump(dict_param, json_file, indent=4)

pd.DataFrame({"alpha":alphas}).to_csv(folder+'alphas.csv')
vect_ones = np.ones(T)


methods = {
    "quant_tracking_multi_alpha":[quant_tracking_multi_alpha, 0.0545],
    "quant_tracking_simple_proj":[quant_tracking_simple_proj, 0.0545],
    "quant_tracking_monotonic":[quant_tracking_monotonic, 0.0545],
    "quant_tracking_monotonic_fixed":[quant_tracking_monotonic_fixed, 0.0006951927961775605],
    "quant_tracking_proj":[quant_tracking_proj, 0.0545]
    }


results = {}
df_diff_cov = pd.DataFrame()
df_diff_quant = pd.DataFrame()
df_set_size = pd.DataFrame()
q_dict = defaultdict(list)

for i in tqdm(range(N_sim)):
    
    data, Q = gen_synthetic_unif(T, folder, alphas, x0, lower, upper, step_std, w, rng, save=False, plot=False)
    scores = data['scores'].to_numpy()
    z = data['z'].to_numpy()

    #data, Q = gen_synthetic_beta_2(T, folder, alphas, a, b, save=False, plot=False)
    #data, Q = gen_synthetic_beta_2(T, folder, alphas, B=2, save=False, plot=False)
    #data, Q = gen_synthetic(T, folder, scale, alphas, save=False, plot=False)
    #
    
    df_diff_quant_i = pd.DataFrame()
    df_i = pd.DataFrame()
    df_ss = pd.DataFrame()
    
    for x in (methods):
        method = methods[x][0]
        lr = methods[x][1]
        result = method(
                    scores,
                    alphas,
                    lr*vect_ones
                )
            
        qs = np.array(result['q'])

        df_diff_quant_i = pd.concat([
            df_diff_quant_i, 
            pd.DataFrame({x:np.abs(qs - Q).sum(1)}).rolling(rolling_window).mean()
            ], axis=1)
            
        df_i = pd.concat([
                df_i, 
                pd.DataFrame({x:np.abs(np.array(result['coverage']).mean(0)+alphas-1)})
                ], axis=1)
        
        
        df_ss = pd.concat([
            df_ss,
            pd.DataFrame({x:np.abs(qs).mean(0)})
            ], axis=1)
    
        q_dict[x].append(result['q'])
        
    result = {}
    result['coverage'] = [(Q[t] >= scores[t])*1 for t in range(scores.shape[0])]
    df_i = pd.concat([
            df_i, 
            pd.DataFrame({'ground_truth':np.abs(np.array(result['coverage']).mean(0)+alphas-1)})
            ], axis=1)
    
    df_ss = pd.concat([
            df_ss, 
            pd.DataFrame({'ground_truth':np.abs(Q).mean(0)})
            #pd.DataFrame({'ground_truth':np.abs(Q).sum(1)}).rolling(rolling_window).mean()
            ], axis=1)
    
    
    df_set_size = pd.concat([
        df_set_size,
        df_ss.T
        ], axis=0)
    
    df_diff_quant = pd.concat([
        df_diff_quant,
        df_diff_quant_i.T
        ], axis=0)
    
    df_diff_cov = pd.concat([
        df_diff_cov,
        df_i.T
        ], axis=0)
    
    plt.plot(z, alpha=.2, color='blue')
    plt.ylabel(r'$z_t$', fontsize=15)
    plt.xlabel('time (t)', fontsize=15)

plt.show()
df_diff_quant = df_diff_quant.reset_index().rename({'index':'method'}, axis=1)

table_mean = df_diff_quant.groupby('method').mean()
table_mean.columns = np.int64(table_mean.columns)
table_mean = table_mean.T

table_std = df_diff_quant.groupby('method').std()
table_std.columns = np.int64(table_std.columns)
table_std = table_std.T

method_list = [
    'quant_tracking_multi_alpha',
    'quant_tracking_simple_proj',
    #'quant_tracking_monotonic',
    'quant_tracking_monotonic_fixed',
    'quant_tracking_proj'
] 

dict_methods = {
    'quant_tracking_multi_alpha': 'Quantile Tracker',
 'quant_tracking_simple_proj': 'Quantile Tracker Projected',
 'quant_tracking_monotonic_fixed': 'Exponentiated Gradient',
 #'quant_tracking_monotonic_fixed': 'Exponentiated Gradient Fixed',
 'quant_tracking_proj': 'Projected Gradient',
               }

for m in dict_methods.keys():
    plt.plot(
        #np.log(table_mean[m]),
        table_mean[m],
        label=dict_methods[m]
    )
    plt.fill_between(
        np.arange(table_mean.shape[0]),
        (table_mean[m] - table_std[m]),
        (table_mean[m] + table_std[m]),
        #np.log(table_mean[m] - table_std[m]),
        #np.log(table_mean[m] + table_std[m]),
        alpha=.3            
    )

plt.xlabel('time t')
plt.legend()
plt.show()

df_set_size.to_csv(folder+'table_set_size.csv')
df_diff_quant.to_csv(folder+'table_diff_quant.csv')
df_diff_cov.to_csv(folder+'table_diff_cov.csv')

filename = folder+'qs.json'

with open(filename, 'w') as json_file:
    json.dump(q_dict, json_file, indent=4) # 'indent=4' makes the file human-readable

print(f"Dictionary successfully saved to {filename}")

