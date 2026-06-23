#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:58:45 2026

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

lr = 10**(np.linspace(-6, 0, 20))

methods = {
    "quant_tracking_multi_alpha":quant_tracking_multi_alpha,
    "quant_tracking_simple_proj":quant_tracking_simple_proj,
    "quant_tracking_monotonic":quant_tracking_monotonic,
    "quant_tracking_monotonic_fixed":quant_tracking_monotonic_fixed,
    "quant_tracking_proj":quant_tracking_proj
    }

results = {}
df = pd.DataFrame()
df_set_size = pd.DataFrame()
df_diff_quant = pd.DataFrame()

data, Q = gen_synthetic_unif(T, folder, alphas, x0, lower, upper, step_std, w, rng, save=False, plot=True)
scores = data['scores'].to_numpy()

for x in tqdm(methods):
    method = methods[x]
    df_diff_quant_m = pd.DataFrame()
    df_m = pd.DataFrame()
    for eta in lr:
        result = method(
                scores,
                alphas,
                eta**np.ones(T),
                #ahead
            )
        
        qs = np.array(result['q'])
        df_diff_quant_m = pd.concat(
            [
                df_diff_quant_m, 
                pd.DataFrame({eta:np.abs(qs - Q).sum(1)}).rolling(rolling_window).mean()
            ], 
            axis=1
            )
        
        df_m = pd.concat(
            [
                df_m, 
                pd.DataFrame({eta:np.abs(np.array(result['coverage']).mean(0)+alphas-1)})
            ], 
            axis=1
            )
        
    df_diff_quant_m.to_csv(folder+'table_diff_quant_'+x+'.csv')
    df_m.to_csv(folder+'table_diff_cov_'+x+'.csv')

method_list = [
    'quant_tracking_multi_alpha',
    'quant_tracking_simple_proj',
    'quant_tracking_monotonic',
    'quant_tracking_monotonic_fixed',
    'quant_tracking_proj'
] 

for m in method_list:
    table_diff_quant = pd.read_csv(folder+'table_diff_quant_'+m+'.csv')
    table_diff_cov = pd.read_csv(folder+'table_diff_cov_'+m+'.csv')
    del table_diff_quant['Unnamed: 0']
    opt_id = np.argmin(table_diff_quant.mean())
    for eta in table_diff_quant.columns[opt_id:opt_id+1]:
        _ = plt.plot(table_diff_quant[eta], label=m)#.plot()
        

plt.legend()
plt.show()


for m in method_list:
    table_diff_quant = pd.read_csv(folder+'table_diff_quant_'+m+'.csv')
    del table_diff_quant['Unnamed: 0']
    print(
        m,
        np.argmin(table_diff_quant.mean()),
        np.min(table_diff_quant.mean()),
        table_diff_quant.columns[np.argmin(table_diff_quant.mean())]
         )