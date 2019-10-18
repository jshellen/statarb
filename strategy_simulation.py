# -*- coding: utf-8 -*-

#%%

from src.simulation.simulate_cointegrated_assets import simulate_cointegrated_assets

N_sim   = 1
N_steps = 500
B_0     = 100
mu      = 0.05
kappa   = 5.5
theta   = 0.0
eta     = 0.05
sigma_B = 0.15
dt      = 1.0/250.0

A,B,X = simulate_cointegrated_assets(N_sim,N_steps,B_0,mu,kappa,theta,eta,sigma_B,dt)

#%

import matplotlib.pyplot as plt

kws_A = {'label':'A','color':'blue' ,'alpha':0.75}
kws_B = {'label':'B','color':'black','alpha':0.75}
kws_X = {'label':'X','color':'black','alpha':0.75}

fig,ax = plt.subplots(2,1,figsize=(10,6))

ax[0].plot(A[:,0],**kws_A)
ax[0].plot(B[:,0],**kws_B)
ax[0].set_ylabel('Prices')
ax[0].legend()

ax[1].plot(X[:,0],**kws_X)
ax[1].set_ylabel('Spread')
ax[1].legend()


#%%

from src.optimal_controls.ou_params import Ornstein_Uhlenbeck_Parameters
from src.optimal_controls.ou_spread_model_parameters import OU_Spread_Model_Parameters

# Estimate spread parameters
ou_params  = Ornstein_Uhlenbeck_Parameters()
success = ou_params.estimate_using_ols(A,B,dt)
if(success):
    print('OLS estimates are:')
    print(ou_params)
else:
    print('Failed to estimate model parameters!')
    
# Create trading model parameters
nominal  = 1000000
symbol_A = 'A'
symbol_B = 'B'
horizon  = None
risk_tol = -float(250)
max_leverage = 1

model_params = OU_Spread_Model_Parameters(nominal, symbol_A, symbol_B, horizon, risk_tol, max_leverage)

#%%
from utils.plot_utils import plot_optimal_solution

plot_optimal_solution(X,ou_params,model_params)














