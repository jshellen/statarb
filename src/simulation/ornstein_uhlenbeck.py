# -*- coding: utf-8 -*-

import numpy as np

def sim_ou(X_0,k,theta,sigma,dt,N_steps):
    X    = np.zeros(N_steps)
    X[0] = X_0
    for i in range(1,N_steps):
        X[i] = X[i-1] + k*(theta - X[i-1])*dt + sigma*np.sqrt(dt)*np.random.normal(0,1)
    return X

def ornstein_uhlenbeck(X_0,k,theta,sigma,dt,N_steps,N_paths):
    size = (N_steps,N_paths)
    X    = np.zeros(size)
    for j in range(0,N_paths):
        X[:,j] = sim_ou(X_0,k,theta,sigma,dt,N_steps)
    return X