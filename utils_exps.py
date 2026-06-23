#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:01:06 2026

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
from scipy.stats import gamma 
from scipy.stats import beta 
from brownian_motion import *

def table_cov(result, alphas, ws):
    cov = pd.DataFrame(result['coverage'])
    cov.columns = np.round(alphas, 2)
    df = pd.DataFrame(
                 np.abs(cov.rolling(window=ws).mean()- 1 + alphas).mean()
                )
            
    df.columns = [result['method']]
    return df

def table_set_size(result, alphas, ws):
    set_size = pd.DataFrame(result['q'])
    set_size.columns = np.round(alphas, 2)
    df = pd.DataFrame(
                 2*np.abs(set_size.rolling(window=ws).mean().mean())
                )
            
    df.columns = [result['method']]
    return df


def plot_dist_cov(result, alphas):
    
    cov = np.array(result['coverage'])
    norm = plt.Normalize(vmin=min(alphas), vmax=max(alphas))
    cmap = plt.cm.viridis
    
    K = alphas.shape[0]
    N = cov.shape[0]
    colors = [cmap(i / K) for i in range(K)]

    fig, ax = plt.subplots()
    
    miss_cov = []
    for j in range(len(alphas)):
        coverage = np.cumsum(cov[:, j]) / np.arange(1, N+1)
        miss_cov.append(np.abs(coverage - 1 + alphas[j]))
        ax.plot(
            miss_cov[-1],
            color=cmap(norm(alphas[j]))
        )
    
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    ax.set_xlabel("Time t")
    ax.set_ylabel("long-run coverage error")
    fig.colorbar(sm, ax=ax, label=r'$\alpha$')
    plt.show()
    
    return miss_cov
    
def plot_dist_cov_mean(data, result, alphas, label=''):
    
    cov = np.array(result['coverage'])
    #norm = plt.Normalize(vmin=min(alphas), vmax=max(alphas))
    #cmap = plt.cm.viridis
    
    K = alphas.shape[0]
    N = cov.shape[0]
    #colors = [cmap(i / K) for i in range(K)]

    #fig, ax = plt.subplots()
    
    miss_cov = []
    for j in range(len(alphas)):
        coverage = np.cumsum(cov[:, j]) / np.arange(1, N+1)
        miss_cov.append(np.abs(coverage - 1 + alphas[j]))
    
    plt.plot(
        data.index,
        np.mean(miss_cov, 0),
        label=label
            #color=cmap(norm(alphas[j]))
        )
    
    #sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    #sm.set_array([])
    plt.xlabel("time (t)", fontsize=15)
    plt.ylabel(r"TCE$_t$", fontsize=15)
    #fig.colorbar(sm, ax=ax, label=r'$\alpha$')
    #plt.show()
    
    return miss_cov

def plot_dist_cov_rolling_mean(data, result, alphas, ws, label=''):
    
    
    #cov = pd.DataFrame(result['coverage'])
    #cov.columns = np.round(alphas, 2)
    #df = pd.DataFrame(
    #             np.abs(cov.rolling(window=ws).mean()- 1 + alphas).mean()
    #            )
    
    #cov = np.array(result['coverage'])
    cov = pd.DataFrame(result['coverage'])
    #norm = plt.Normalize(vmin=min(alphas), vmax=max(alphas))
    #cmap = plt.cm.viridis
    
    K = alphas.shape[0]
    N = cov.shape[0]
    #colors = [cmap(i / K) for i in range(K)]

    #fig, ax = plt.subplots()
    
    #miss_cov = []
    #for j in range(len(alphas)):
        #coverage = np.cumsum(cov[:, j]) / np.arange(1, N+1)
    #    coverage = cov.rolling(ws).mean()
    #    miss_cov.append(np.abs(coverage - 1 + alphas[j]).to_numpy())
    
    miss_cov = np.abs(cov.rolling(ws).mean()-(1-alphas)).to_numpy().mean(1)
    
    plt.plot(
        data.index,
        miss_cov,
        #np.nanmean(miss_cov, 0),
        label=label
            #color=cmap(norm(alphas[j]))
        )
    
    #sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    #sm.set_array([])
    plt.xlabel("time (t)", fontsize=15)
    plt.ylabel(r"TCE_$t$", fontsize=15)
    #fig.colorbar(sm, ax=ax, label=r'$\alpha$')
    #plt.show()
    
    return miss_cov

def plot_pred_sets(data, result, alphas, filename=None):
    
    
    qs = np.array(result['q'])
    cmap = get_cmap('viridis')
    forecasts = data['forecasts']
    # Generate colors evenly spaced across the colormap
    N = alphas.shape[0]#results['q'].shape[0]
    colors = [cmap(i / N) for i in range(N)]
    window = 200
    err_list = []
    t_1 = 650
    dt = 100#qs.shape[0]
    y_max = np.array(data['y']).max()
    y_min = np.array(data['y']).min()
    #fig, ax = plt.subplots()
    
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.15)
    
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1]) #, sharex=ax_top

    
    #ax.plot(np.array(data['y']), color='black')
    #ax.plot(forecasts, color='blue')
    
    #for x in np.where(np.sum(np.round(np.diff(qs[:, ::-1], 1), 4)<0, 1)>0)[0]:#np.sum(np.diff(results['q'][:, ::-1], 1)<0, 0):
    #    ax.vlines(data.index[x], y_min-1, y_max+1, color='red', alpha=.1)
    
    #ax_top.plot(
    #    data.index,
    #    np.sum(np.round(np.diff(qs[:, ::-1], 1), 4)<0, 1)
    #)
    
    #non_monoto = pd.DataFrame(
    #    np.round(
    #        np.diff(    
    #            np.concatenate([np.zeros((qs.shape[0],1)), qs[:, ::-1]], axis=1), 
    #            1)[:,::-1], 
    #        4)).rolling(200).mean()

    #non_monoto['Date'] = data.index
    #non_monoto = non_monoto.set_index('Date')
    
    #for i in np.arange(N):#[::-1]
    #    _ = ax_top.plot(non_monoto.iloc[:,i],
    #                color=colors[i]
    #                )
        
    #_ = ax_top.hlines(0,data.index[0], data.index[-1],
    #            color='red'
    #            )
    norm = plt.Normalize(vmin=min(alphas), vmax=max(alphas))
    
    cov = np.array(result['coverage'])
    #miss_cov = []
    #for j in range(len(alphas)):
        #coverage = np.cumsum(cov[:, j]) / np.arange(1, dt+1)
    #    coverage = pd.DataFrame(cov[:, j]).rolling(window).mean()
    #    coverage = np.abs(coverage- 1 + alphas[j]) 
        #miss_cov.append(np.abs(coverage)) #- 1 + alphas[j]
    #    ax_top.plot(
    #        data.index,
    #        coverage,
    #        color=cmap(norm(alphas[j])),
    #        alpha=.8
    #    )
        
    ax_top.plot(data['y'][t_1:t_1+dt], color='black')
    ax_top.plot(data['forecasts'][t_1:t_1+dt], color='blue')
    
    ax_bot.plot(data['y'], color='black')
    ax_bot.plot(data['forecasts'], color='blue')
    for i in np.arange(N):#[::-1]:
        
        ax_top.fill_between(data.index[t_1:t_1+dt],
                         forecasts[t_1:t_1+dt] - qs[t_1:t_1+dt, i],
                         forecasts[t_1:t_1+dt] + qs[t_1:t_1+dt, i],
                 color=colors[i],
                        alpha=.5)
        
        ax_bot.fill_between(data.index,
                         forecasts - qs[:, i],
                         forecasts + qs[:, i],
                 color=colors[i],
                        alpha=.5)
    
    ax_bot.set_xlabel('time (t)', fontsize=15)
    ax_bot.set_ylabel(r'$y_t$', fontsize=15)
    ax_top.set_ylabel(r'$y_t$', fontsize=15)
    
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_top, ax_bot], label=r'$\alpha$')
    #fig.colorbar(sm, ax=ax_bot, label=r'$\alpha$')
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def gen_synthetic(T, folder, scale, alphas, save=True, plot=True):
    #scale = .1 * np.ones(T)#+ .3*(np.arange(T)>T/2)
    #scale = .1*((np.cos(np.linspace(0,5*np.pi, T)))**2 + np.linspace(0, 3, T))
    q_99 = np.array([gamma.ppf(q, a=1,  scale=scale) for q in (1-alphas[:1])]).flatten()
    q_995 = np.array([gamma.ppf(q, a=1,  scale=scale) for q in [.995]]).flatten()
    X = np.random.gamma(1, scale=scale)
    X = np.clip(X, 0, q_995)
    Q = np.array([gamma.ppf(q, a=1,  scale=scale) for q in (1-alphas)]).T
    if plot:
        _ = plt.plot(X)
        _ = plt.show()
    df = pd.DataFrame({'scores':X})
    df_Q = pd.DataFrame(Q)
    if save:
        df.to_csv(folder+'data.csv')
        df_Q.to_csv(folder+'Q.csv')
    return df, Q

def gen_synthetic_beta(T, folder, alphas, B=5, save=True, plot=True):
    #scale = .1 * np.ones(T)#+ .3*(np.arange(T)>T/2)
    a = 2 
    b = 2#np.linspace(1,10, T)
    #X = np.random.beta(a, b)
    slope = np.linspace(.1, B, T)
    X = np.random.beta(a, b, T) * slope
    Q = np.array([beta.ppf(q, a=a,  b=b) for q in (1-alphas)])* slope[:,None]

    if plot:
        _ = plt.plot(X)
        _ = plt.show()
    df = pd.DataFrame({'scores':X})
    df_Q = pd.DataFrame(Q)
    if save:
        df.to_csv(folder+'data.csv')
        df_Q.to_csv(folder+'Q.csv')
    return df, Q


def gen_synthetic_beta_2(T, folder, alphas, a, b, save=True, plot=True):
    #scale = .1 * np.ones(T)#+ .3*(np.arange(T)>T/2)
    #a = 2 
    #b = 2#np.linspace(1,10, T)
    #X = np.random.beta(a, b)
    #slope = np.linspace(.1, B, T)
    X = np.random.beta(a, b, T)# * slope
    Q = np.array([beta.ppf(q, a=a,  b=b) for q in (1-alphas)]).T#* slope[:,None]

    if plot:
        _ = plt.plot(X)
        _ = plt.show()
    df = pd.DataFrame({'scores':X})
    df_Q = pd.DataFrame(Q)
    if save:
        df.to_csv(folder+'data.csv')
        df_Q.to_csv(folder+'Q.csv')
    return df, Q


def gen_synthetic_unif(T, folder, alphas, x0, lower, upper, step_std, w, rng, save=True, plot=True):
    z = reflected_random_walk(
        T-1, 
        x0=x0, 
        lower=lower, 
        upper=upper, 
        step_std=step_std, 
        rng=rng
        )
    #b = pd.DataFrame(b).rolling(1000).mean().to_numpy().flatten()[999:]
    
    scores = z + w*(np.random.rand(T)-0.5)
    Q = (np.ones(T)*(1-alphas)[:, None]).T
    Q = z[:, None] + w*(Q-0.5)

    if plot:        
        plt.scatter(
            np.arange(scores.shape[0]),
            scores, 
            alpha=.1,
            s=5
        )
        plt.plot(Q[:,0], color='black')
        plt.plot(Q[:,-1], color='black')
        plt.show()
        
    df = pd.DataFrame({'scores':scores, 'z':z})
    df_Q = pd.DataFrame(Q)
    if save:
        df.to_csv(folder+'data.csv')
        df_Q.to_csv(folder+'Q.csv')
    return df, Q