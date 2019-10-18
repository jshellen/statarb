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
risk_tol = -float(100)
max_leverage = 1

model_params = OU_Spread_Model_Parameters(nominal, symbol_A, symbol_B, horizon, risk_tol, max_leverage)

#%%
import numpy as np

from src.optimal_controls.ou_spread_model import OU_Spread_Model

taus = np.linspace(1,0.001,50)
xs   = np.linspace(-0.05,0.05,50)
hs   = np.zeros((len(taus),len(xs)))
for i,tau in enumerate(taus):
    for j,x in enumerate(xs):
        opt_sol = OU_Spread_Model.solve_allocation(ou_params,model_params,x,tau)
        hs[i,j] = opt_sol.alloc_a_pct_trunc
        
#%%
import numpy as np
import matplotlib.pyplot as plt
from   mpl_toolkits.axes_grid1 import make_axes_locatable

fig,ax = plt.subplots(figsize=(7,7))

# Plot the solution
im     = ax.imshow(hs, cmap=plt.cm.RdBu)  

# Set y-labels
y_rng  = np.arange(0,len(taus),10)
ax.set_ylabel('Time',fontsize=14)
ax.set_yticks(y_rng)
ax.set_yticklabels([round(taus[t],2) for t in y_rng])
    
# Set x-labels
x_rng = np.arange(0,len(xs),10)
ax.set_xlabel('Spread Level',fontsize=14)
ax.set_xticks(y_rng)
ax.set_xticklabels([round(xs[i],2) for i in x_rng])

# Plot contour lines
cset = plt.contour(hs, np.arange(-1, 1.5, 0.2), linewidths=2,colors = 'black')
plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=14)

# Set colorbar
divider = make_axes_locatable(ax)
cax     = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax = cax)  

# Set title
ax.set_title('Optimal allocation')
plt.show()

