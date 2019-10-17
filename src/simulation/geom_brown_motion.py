# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:54:51 2019

@author: helleju
"""
import numpy as np

def geometrix_brownian_motion(S0,mu,sigma,dt,N_steps,N_paths):
    
    size = (N_steps,N_paths)
    S    = np.sqrt(dt)*np.random.normal(0,sigma,size) + mu*dt*np.ones(size)
    
    return np.cumprod(np.exp(S),axis=0)
