#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:39:35 2026

@author: eochoa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#scale = .1*np.random.rand(T)
#scale = .1*np.ones(T)
#scale = .1*((np.cos(np.linspace(0,5*np.pi, T)))**2 + np.linspace(0, 3, T))

def gen_synthetic(T):
    scale = .1 + .3*(np.arange(T)>T/2)
    X = np.random.gamma(1,scale=scale)
    _ = plt.plot(X)
    _ = plt.show()
    df = pd.DataFrame({'scores':X})
    df.to_csv('data/synthetic_data.csv')
    return df