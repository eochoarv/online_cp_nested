#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:34:41 2026

@author: eochoa
"""

import pandas_datareader.data as web
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.ar_model import AutoReg

cpi = web.DataReader("CPIAUCSL", "fred", "1947-01-01", "2026-01-01")
cpi = cpi.asfreq("MS").interpolate(method="time")
inf = cpi["CPIAUCSL"].pct_change(12) * 100  # YoY %

#inf = np.array(inf)[12:]
inf = inf.dropna()
n = inf.shape[0]

h = 6
window = 60
predictions = []

for t in tqdm(range(window+8, n)):
    inf_train = inf[t-window:t]
    model = AutoReg(inf_train, lags=3, old_names=False).fit()
    pred = model.get_prediction(start=len(inf_train), end=len(inf_train)+h)
    predictions.append(pred.predicted_mean[-1:])
    
    
scores = np.abs((inf - pd.concat(predictions)).dropna())

data = pd.concat([
    inf,
    pd.concat(predictions),
    scores,
], axis=1).dropna()

data.columns = ['y', 'forecasts', 'scores']

data.to_csv('data_inflation.csv')