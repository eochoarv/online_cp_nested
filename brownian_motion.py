#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:27:41 2026

@author: eochoa
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def brownian_motion(T=1.0, N=1000, seed=None):
    """
    Simulate 1D Brownian motion
    
    T: total time
    N: number of steps
    """
    rng = np.random.default_rng(seed)
    
    dt = T / N
    increments = rng.normal(0, np.sqrt(dt), size=N)
    
    X = np.cumsum(increments)
    X = np.insert(X, 0, 0.0)  # start at 0
    
    t = np.linspace(0, T, N+1)
    return t, X

def geometric_brownian_motion(T=1.0, N=1000, mu=0.1, sigma=0.2, S0=1.0, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / N
    
    increments = (mu - 0.5 * sigma**2) * dt + sigma * rng.normal(0, np.sqrt(dt), size=N)
    log_S = np.cumsum(increments)
    log_S = np.insert(log_S, 0, 0.0)
    
    S = S0 * np.exp(log_S)
    t = np.linspace(0, T, N+1)
    return t, S

def gbm_quantiles_exact(t, mu=0.1, sigma=0.2, S0=1.0, quantiles=[0.1, 0.5, 0.9]):
    results = {}
    
    for q in quantiles:
        z = norm.ppf(q)
        values = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)
        results[q] = values
        
    return results


import numpy as np
import matplotlib.pyplot as plt

def reflected_random_walk(
    T=500,
    x0=0.5,
    lower=0.0,
    upper=1.0,
    step_std=0.05,
    rng=None
):
    """
    Simulate a 1D random walk with reflecting boundaries.

    Parameters
    ----------
    T : int
        Number of time steps.
    x0 : float
        Initial position.
    lower : float
        Lower reflecting boundary.
    upper : float
        Upper reflecting boundary.
    step_std : float
        Standard deviation of Gaussian increments.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    x : np.ndarray
        Array of length T+1 containing the trajectory.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (lower <= x0 <= upper):
        raise ValueError("x0 must lie within [lower, upper].")
    if lower >= upper:
        raise ValueError("lower must be strictly smaller than upper.")

    x = np.empty(T + 1)
    x[0] = x0

    for t in range(T):
        eps = rng.normal(0, step_std)
        proposal = x[t] + eps

        # Reflect until the point falls inside [lower, upper]
        
        while proposal < lower or proposal > upper:
            if proposal < lower:
                proposal = 2 * lower - proposal
            elif proposal > upper:
                proposal = 2 * upper - proposal

        x[t + 1] = proposal

    return x