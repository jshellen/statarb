#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import numpy as np

from src.optimal_controls.ou_spread_model import OUSpreadModelSolver

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def find_nearest(array, value):
    
    array = np.asarray(array)
    idx   = (np.abs(array - value)).argmin()
    
    return idx, array[idx]
    

def plot_optimal_solution(X,ou_params,model_params,N_points=200):

    eta   = ou_params.eta
    theta = ou_params.theta
    
    from matplotlib import rcParams
    rcParams['axes.titlepad'] = 20 
    
    # Compute optimal solution over a (t,X_t) grid
    taus = np.linspace(1,0.001,N_points)
    xs   = np.linspace(theta - 1.5*eta,theta + 1.5*eta,N_points)
    hs   = np.zeros((len(taus),len(xs)))
    for i,tau in enumerate(taus):
        for j,x in enumerate(xs):
            opt_sol = OU_Spread_Model.solve_allocation(ou_params,model_params,x,tau)
            hs[i,j] = opt_sol.alloc_a_pct_trunc
    
    # Plot the spread with the optimal solution
    t_x  = np.linspace(1,0.001,len(X))
    data = {}
    for i in range(0,len(X)):
        ix_y,v_y = find_nearest(taus,t_x[i])
        ix_x,v_x = find_nearest(xs,X[i])
        data.update({ix_y:ix_x})
    fig,ax = plt.subplots(figsize=(7,7))
    
    # Plot spread path as yellow squares
    ax.plot(list(data.values()),list(data.keys()),color='black',lw=3)
    
    # Plot heatmap of the optimal solution
    im_1 = ax.imshow(hs, cmap = plt.cm.winter)  
    
    # Set y-labels
    y_rng  = np.arange(0,len(taus),50)
    ax.set_ylabel('Trading Time Remaining',fontsize=14)
    ax.set_yticks(y_rng)
    ax.set_yticklabels([round(taus[t],2) for t in y_rng])
        
    # Set x-labels
    x_rng = np.arange(0,len(xs),50)
    ax.set_xlabel('Spread Level',fontsize=14)
    ax.set_xticks(y_rng)
    ax.set_xticklabels([round(xs[i],2) for i in x_rng])
    
    # Plot contour lines
    cset = plt.contour(hs, np.arange(-1, 1.5, 0.2), linewidths=2,colors = 'red')
    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=14)
    
    # Set colorbar
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_1, cax = cax)  
    
    # Set title
    ax.set_title('Optimal allocation')
    
    fig.tight_layout()
    
    plt.show()



#%%

