# -*- coding: utf-8 -*-

from numpy import roll, sqrt, log, std, ndarray
import statsmodels.api as sm


def estimate_ou_parameters(x, dt):
    
    if not isinstance(x, ndarray):
        raise TypeError(f'x needs to be type of numpy.ndarray, it was {type(x)}')
    
    if not isinstance(dt, float):
        raise TypeError('dt needs to be type of float!')
    
    if dt <= 0:
        raise ValueError('Delta time has to be positive and non-zero!')
        
    S_m = roll(x, 1)[1:]
    S_p = x[1:]
    X = sm.add_constant(S_m)
    Y = S_p
    ols_est = sm.OLS(Y, X).fit()
    a = ols_est._results.params[1]
    b = ols_est._results.params[0]
    kappa = -log(a)/dt
    theta = b/(1 - a)
    sigma = std(ols_est.resid)*(sqrt(-2*log(a)/(dt*(1-a**2))))       
    
    return (kappa, theta, sigma)