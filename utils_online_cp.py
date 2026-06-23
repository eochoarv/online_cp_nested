#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 20:49:54 2025

@author: eochoa
"""

#import numpy as np
#from tqdm import tqdm
#from sklearn.datasets import make_classification
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression

 
import os
import numpy as np
import pandas as pd
import copy
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from tqdm import tqdm
import pdb
import warnings
from scipy.interpolate import interp1d
#import cvxpy as cp
from sklearn.isotonic import isotonic_regression

"""
    BASELINES
"""

def step_size_(q, alpha, s, eta):
    q_c = q[1:-1]
    d_prev = np.abs(np.diff(q))[1:]
    d_next = np.abs(np.diff(q[::-1]))[::-1][:-1]

    idx_dec = q_c > s
    idx_inc = q_c < s
    err = 1*idx_inc

    step_1 = (idx_inc*d_next-idx_dec*d_prev)/2
    step_2 = eta*(err-alpha[1:-1])

    step_size_1 = np.abs(step_1)
    step_size_2 = np.abs(step_2)

    step = (step_2*(step_size_2<step_size_1)+ 
            step_1*(step_size_2>step_size_1))

    return np.concatenate([[0], step, [0]])


def step_size(q, alpha, s, eta):
    df = pd.DataFrame({
        'alpha':alpha,
        'q':q,
        'prev':np.concatenate([np.abs(np.diff(q)),[10e16]]),
        'next':np.concatenate([[10e16],np.abs(np.diff(q))]),
    })
    
    idx_dec = q > s
    idx_inc = q < s
    err = 1*idx_inc
    
    step_1 = pd.concat([
        df.loc[idx_dec, 'prev'],
        df.loc[idx_inc, 'next']
             ], 
        axis=0).to_numpy()/2
    
    step_2 = eta*(err-alpha)
    
    step_size_1 = np.abs(step_1)
    step_size_2 = np.abs(step_2)
    
    step = (
        step_2*(step_size_2<step_size_1)+ 
        step_1*(step_size_2>step_size_1)*np.sign(step_2)
        )
    return step


def trailing_window(
    scores,
    alpha,
    lr, # Dummy argument
    weight_length,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    for t in (range(T_test)):
        t_pred = t - ahead + 1
        if min(weight_length, t_pred) < np.ceil(1/alpha):
            qs[t] = np.infty
        else:
            qs[t] = np.quantile(scores[max(t_pred-weight_length,0):t_pred], 1-alpha, method='higher')
    results = {"method": "Trail", "q" : qs}
    return results

def aci_clipped(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in (range(T_test)):
        t_pred = t - ahead + 1
        clip_value = scores[max(t_pred-window_length,0):t_pred].max() if t_pred > 0 else np.infty
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                qs[t] = np.infty
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr[t]*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.infty
        if qs[t] == np.infty:
            qs[t] = clip_value
    results = { "method": "ACI (clipped)", "q" : qs, "alpha" : alphas}
    return results


def aci(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in (range(T_test)):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                qs[t] = np.infty
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr[t]*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.infty
    results = { "method": "ACI", "q" : qs, "alpha" : alphas, "err":1-covereds}
    return results


def quant_tracking_multi_alpha(
    scores,
    alphas,
    lr,
    #ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    k = alphas.shape[0]
    
    B = scores[0]
    ds = np.zeros((T_test, k+1)) + B/(k+1)
    qs = np.cumsum(ds, 1)[:,::-1][:,1:]
    #qs = np.zeros((T_test, k)) + scores[0]
    #qt = 0
    covereds = np.zeros((T_test, k))
    grads = np.zeros((T_test, k))
    for t in (range(T_test)):
        #t_pred = t - ahead + 1
        t_pred = t
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update and saturation function
        grad = np.array([alpha if covereds[t_pred, i] else (alpha-1) 
                         for i, alpha in enumerate(alphas)])
        grads[t] = grad
                    
        if t < T_test - 1:
            qs[t+1] = qs[t] - lr[t] * grad

    results = { 
        "method": "quant_tracking", 
        "q" : qs.tolist(), 
        "grad" : grads.tolist(), 
        "coverage":covereds.tolist()
        }
    
    return results

def quant_tracking_simple_proj(
    scores,
    alphas,
    lr,
    #ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    k = alphas.shape[0]
    
    B = scores[0]
    ds = np.zeros((T_test, k+1)) + B/(k+1)
    qs = np.cumsum(ds, 1)[:,::-1][:,1:]
    qs_proj = np.cumsum(ds, 1)[:,::-1][:,1:]
    #qs = np.zeros((T_test, k))
    #qs_proj = np.zeros((T_test, k))
    
    covereds = np.zeros((T_test, k))
    grads = np.zeros((T_test, k))
    C = np.zeros((k-1, k))
    
    for i in range(C.shape[0]):
        C[i, C.shape[0]-i] = 1
        C[i, C.shape[0]-1-i] = -1
    
    b = np.zeros(C.shape[0])
    
    for t in (range(T_test)):
        #t_pred = t - ahead + 1
        t_pred = t
        
        #x_var = cp.Variable(k)
        #proj_problem = cp.Problem(
        #    cp.Minimize(cp.sum_squares(x_var - qs[t])), 
        #    [C @ x_var <= b]
        #    )
        #proj_problem.solve(
        #    solver=cp.OSQP, 
        #    warm_start=True,
        #    eps_abs=1e-8, 
        #    eps_rel=1e-8,
        #    verbose=True
        #    )
        qt = isotonic_regression(qs[t], y_min=0, increasing=False)#x_var.value
        qs_proj[t] = qt
        
        covereds[t] = qt >= scores[t]
        # Next, calculate the quantile update and saturation function
        grad = np.array([alpha if covereds[t_pred, i] else (alpha-1) 
                         for i, alpha in enumerate(alphas)])
        grads[t] = grad
        
        if t < T_test - 1:
            qs[t+1] = qs[t] - lr[t] * grad
            #x_var = cp.Variable(k)
            #proj_problem = cp.Problem(
            #    cp.Minimize(cp.sum_squares(x_var - qs[t+1])), 
            #    [C @ x_var <= b]
            #    )
            #proj_problem.solve(
            #    solver=cp.OSQP, 
            #    warm_start=True,
            #    eps_abs=1e-8, 
            #    eps_rel=1e-8
            #    )
            
            #qs[t+1] = x_var.value

    results = { 
        "method": "quant_tracking_simple_proj", 
        "q" : qs_proj.tolist(), 
        "grad" : grads.tolist(), 
        "coverage":covereds.tolist()
        }
    
    return results


def quant_tracking_monotonic(
    scores,
    alphas,
    lr,
    delta=0,
    #ahead,
    B_var=True,
    B=None,
    *args,
    **kwargs
):
    
    T_test = scores.shape[0]
    k = alphas.shape[0]
    #Dalphas = np.diff(list(alphas)+[1])
    Dalphas = np.diff([0] + list(alphas)+ [1])
    #if B is None:
    #    B = 1
    if B_var:
        B = scores[0]
    #B = .4
    
    #ds = np.zeros((T_test, k)) + B/k
    ds = np.zeros((T_test, k+1)) + B/(k+1)
    
    #qs = np.zeros((T_test, k))
    qs = np.cumsum(ds, 1)[:,::-1]
    covereds = np.zeros((T_test, k+1))
    dcovereds = np.zeros((T_test, k+1))
    grads = np.zeros((T_test, k+1))


    for t in (range(T_test)):
        #t_pred = t - ahead + 1
        t_pred = t
        covereds[t] = qs[t] >= scores[t]
        
        # Next, calculate the quantile update and saturation function
        if B_var and t > 0:
            B = np.max(scores[:t])
        dcovereds[t] = np.logical_and(
                np.array(list(qs[t][1:])+[0]) < scores[t],
                qs[t] >= scores[t]
                )
        
        Dgrad = np.array([-(1-dalpha) if dcovereds[t_pred, i] else dalpha 
                          for i, dalpha in enumerate(Dalphas)])
        #grad = np.array([-alpha if covereds[t_pred, i] else (1-alpha) 
        #                 for i, alpha in enumerate(alphas)])
        grads[t] = Dgrad
        
        
        if t < T_test - 1:
            ds[t+1] = ds[t] * np.exp(lr[t] * Dgrad)
            #ds[t+1,-1] = ds[t,-1] + lr * grad[-1]
            ds[t+1] = ((1-delta)*ds[t+1]/np.sum(ds[t+1]) + delta/(k+1))*B
            qs[t+1] = np.cumsum(ds[t+1][::-1])[::-1]
            #qs[t+1, 0] = B
            
            
    results = { 
        "method": "quant_tracking_monotonic",
        "q": qs[:,1:].tolist(), 
        "ds": ds.tolist(), 
        "grad": grads[:,1:].tolist(), 
        "coverage": covereds[:,1:].tolist(),
        "dcovereds": dcovereds[:,1:].tolist()
        }
    
    return results

def quant_tracking_monotonic_fixed(
    scores,
    alphas,
    lr,
    delta=0,
    #ahead,
    B_var=True,
    B=None,
    *args,
    **kwargs
):
    
    T_test = scores.shape[0]
    k = alphas.shape[0]
    #Dalphas = np.diff(list(alphas)+[1])
    alphas_ext = np.array([0] + list(alphas))
    #if B is None:
    #    B = 1
    if B_var:
        B = scores[0]
    
    ds = np.zeros((T_test, k+1)) + B/(k+1)
    #qs = np.concatenate([np.cumsum(ds, 1)[:,::-1], np.zeros((T_test, 1))],1)
    qs = np.cumsum(ds, 1)[:,::-1]
    covereds = np.zeros((T_test, k+1))
    grads = np.zeros((T_test, k+1))


    for t in (range(T_test)):
        covereds[t] = qs[t] >= scores[t]
        
        # Next, calculate the quantile update and saturation function
        if B_var and t > 0:
            B = np.max(scores[:t])
        
        #dcovereds[t] = np.logical_and(
        #        np.array(list(qs[t][1:])+[0]) < scores[t],
        #        qs[t] >= scores[t]
        #        )
        
        grad = np.array([alpha if covereds[t, i] else (alpha-1) 
                         for i, alpha in enumerate(alphas_ext)])
        
        grad = B*np.cumsum(grad)
        grads[t] = grad
        
        
        if t < T_test - 1:
            ds[t+1] = ds[t] * np.exp(-lr[t] * grad)
            ds[t+1] = ((1-delta)*ds[t+1]/np.sum(ds[t+1]) + delta/(k+1))*B
            qs[t+1] = np.cumsum(ds[t+1][::-1])[::-1]
            #qs[t+1, 0] = B
            
            
    results = { 
        "method": "quant_tracking_monotonic_fixed",
        "q": qs[:,1:].tolist(), 
        "ds": ds.tolist(), 
        "grad": grads[:,1:].tolist(), 
        "coverage": covereds[:,1:].tolist(),
        #"dcovereds": dcovereds[:,1:].tolist()
        }
    
    return results


def quant_tracking_monotonic_fixed_2(
    scores,
    alphas,
    lr,
    delta=0,
    #ahead,
    B_var=True,
    B=None,
    *args,
    **kwargs
):
    
    T_test = scores.shape[0]
    k = alphas.shape[0]
    
    if B_var:
        B = scores[0]
    
    covereds = np.zeros((T_test, k+1))
    grads = np.zeros((T_test, k+1))

    #w_t = 1/(k+1) * np.ones(k+1)
    ws = np.ones((T_test, k+1)) / (k+1)
    #qt = B*np.cumsum(w_t)[::-1]
    qs = B*np.cumsum(ws, 1)[:,::-1]

    for t in (range(T_test)):
        covereds[t] = qs[t] >= scores[t]
        #covereds_t = qs[t] >= scores[t]
        
        # Next, calculate the quantile update and saturation function
        if B_var and t > 0:
            B = np.max(scores[:t])
        
        
        
        grad = np.array([alpha if covereds[t, i] else (alpha-1) 
                             for i, alpha in enumerate([0]+list(alphas))])
            
        grad = B*np.cumsum(grad)
        grads[t] = grad
        
        if t < T_test - 1:
            ws[t+1] = ws[t] * np.exp(-lr[t] * grad)
            ws[t+1] = ((1-delta)*ws[t+1]/np.sum(ws[t+1]) + delta/(k+1))
            qs[t+1] = B*np.cumsum(ws[t+1][::-1])[::-1]
            #qs[t+1, 0] = B
            
            
    results = { 
        "method": "quant_tracking_monotonic",
        "q": qs[:,1:].tolist(), 
        "ws": ws.tolist(), 
        "grad": grads[:,1:].tolist(), 
        "coverage": covereds[:,1:].tolist(),
        #"dcovereds": dcovereds[:,1:].tolist()
        }
    
    return results


def quant_tracking_proj(
    scores,
    alphas,
    lr,
    eps=0.0001,
    #ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    k = alphas.shape[0]
    
    B = scores[0]
    ds = np.zeros((T_test, k+1)) + B/(k+1)
    qs = np.cumsum(ds, 1)[:,::-1]#[:,1:]
    qs_proj = np.cumsum(ds, 1)[:,::-1]#[:,1:]
    
    #qs = np.zeros((T_test, k))
    #qs_proj = np.zeros((T_test, k))
    #qt = 0
    covereds = np.zeros((T_test, k+1))
    grads = np.zeros((T_test, k+1))
    #C = np.zeros((k-1, k))
    
    #for i in range(C.shape[0]):
    #    C[i, C.shape[0]-i] = 1
    #    C[i, C.shape[0]-1-i] = -1
    #b = np.zeros(C.shape[0])
    
    for t in (range(T_test)):
        t_pred = t
        
        #x_var = cp.Variable(k)
        #proj_problem = cp.Problem(
        #    cp.Minimize(cp.sum_squares(x_var - qs[t])), 
        #    [C @ x_var <= b]
        #    )
        #proj_problem.solve(
        #    solver=cp.OSQP, 
        #    warm_start=True,
        #    eps_abs=1e-8, 
        #    eps_rel=1e-8
        #    )
        #qt = x_var.value
        #qs_proj[t] = qt
        
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update and saturation function
        grad = np.array([alpha if covereds[t_pred, i] else (alpha-1) 
                         for i, alpha in enumerate([0]+list(alphas))])
        grads[t] = grad
        
        if t < T_test - 1:
            #qs[t+1] = qs[t] - lr * grad
            #x_var = cp.Variable(k)
            #proj_problem = cp.Problem(
            #    cp.Minimize(cp.sum_squares(x_var - (qs[t] - lr * grad))), 
            #    [C @ x_var <= b]
            #    )
            #proj_problem.solve(
            #    solver=cp.OSQP, 
            #    warm_start=True,
            #    eps_abs=1e-8, 
            #    eps_rel=1e-8
            #    )
            qt = qs[t] - lr[t] * grad
            qt = qt - eps*np.arange(k+1)[::-1]
            
            qs[t+1] = isotonic_regression(qt, y_min=-eps, increasing=False) + eps*np.arange(k+1)[::-1]

    results = { 
        "method": "quant_tracking_proj", 
        "q" : qs[:,1:].tolist(), 
        #"q_non_proj" : qs_proj[:,1:].tolist(), 
        "grad" : grads[:,1:].tolist(), 
        "coverage":covereds[:,1:].tolist()
        }
    
    return results


def quant_tracking_bounded(
    scores,
    alphas,
    lr,
    #ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    k = alphas.shape[0]
    B = scores[0]
    qs = np.ones((T_test, k))*(1-alphas)*B
    qt = 0
    #alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test, k))
    steps = np.zeros((T_test, k))
    
    for t in (range(T_test)):
        #t_pred = t - ahead + 1
        t_pred = t
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update and saturation function
        #grad = np.array([alpha if covereds[t_pred, i] else (alpha-1) for i, alpha in enumerate(alphas)])
        step = step_size(qs[t], alphas, scores[t], lr[t])
        steps[t] = step
        
        if t < T_test - 1:
            #qs[t+1] = qs[t] - lr * grad            
            qs[t+1] = qs[t] + step
            
    results = { 
        "method": "quant_tracking_bounded", 
        "q" : qs.tolist(), 
        "step" : steps.tolist(), 
        "coverage":covereds.tolist()
        }
    
    
    return results
