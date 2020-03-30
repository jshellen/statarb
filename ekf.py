# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:29:12 2020

@author: Juhis
"""

#%%

import numpy as np

class GBM:
    
    def __init__(self, mu, sigma, dt):
        
        self.m_mu = mu
        self.m_sigma = sigma
        self.m_dt = dt
    
    def simulate(self, s_0, n_step):
        """
        Simulate Geometric Brownian Motion
        """
        s = np.zeros(n_step)
        s[0] = s_0
        for i in range(1, n_step):
            s[i] = s[i-1] + ((self.m_mu - 0.5 * self.m_sigma**2) * self.m_dt
             + self.m_sigma * np.sqrt(self.m_dt) * np.random.normal(0, 1)   ) 
        
        return s

def f(x, dt):
    
    return np.array([[
            x[0] + x[1]*dt,
            x[1]
            ]])

def g(x, dt):
    
    return x[1]
        

def extendedKalman(y, f, g, Q, R, x0, s0, dt):
    
    N = len(y[0,:]) # number of observations
    M = np.array([[1, 0]])  # g.dx(x0.T, dt)
    [n, m] = np.shape(M)
    xPred = np.zeros((m,N))
    xUpdate = np.zeros((m,N))
    SigmaPred = np.zeros((m,m,N))
    SigmaUpdate = np.zeros((m,m,N))
    SigmaUpdate[:,:,0] = s0
    e = np.zeros((n,N))
    Sigmay = np.zeros((n,n,N))
    xUpdate[:, 0] = x0
    I = np.eye(m)
    
    for i in range(1,N):
        
        # Prediction Step
        xPred[:,i] = f(xUpdate[:,i-1].T, dt)  # f.f(xUpdate[:,i-1].T,dt)
        A = np.array([
                [1, dt],
                [0,  1]
                ])  # f.dx(xUpdate[:,i-1].T, dt)
        C = np.array([1, 0]).reshape(1, -1)  #  g.dx(xUpdate[:,i-1].T, dt)
        
        print(A.shape)
        print(C.shape)
        
        SigmaPred[:,:,i] = np.dot(A, np.dot(SigmaUpdate[:,:,i-1], A.T))# + Q
        Sigmay[:,:,i] = np.dot(C, np.dot(SigmaPred[:,:,i], C.T)) #+ R
        
        print(Sigmay[:,:,i])
        
        # Update Step
        K = np.dot(SigmaPred[:,:,i], np.dot (C.T, np.linalg.inv(Sigmay[:,:,i])))
        e[:,i] = y[:,i] - g(xPred[:,i], dt) # y[:,i] - g.f(xPred[:,i],dt)
        xUpdate[:,i] = xPred[:,i] + np.dot(K, e[:,i])
        SigmaUpdate[:,:,i] = np.dot((I - np.dot(K, C)), SigmaPred[:,:,i])
        
    return xPred, xUpdate, SigmaPred, SigmaUpdate, e, Sigmay



#%%

import matplotlib.pyplot as plt

dt = 1.0/250.0
gbm = GBM(0.05, 0.25, dt)

y = gbm.simulate(100, 500)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(y)
plt.show()


x0 = np.array([[y[0], 0]])
s0 = np.ones((2, 2))

extendedKalman(y.reshape(1, -1), f, g, None, None, x0, s0, dt)







