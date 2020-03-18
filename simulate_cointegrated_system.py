import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.simulation.simulate_cointegrated_assets import (
    SystemParams,
    simulate_benchmark_cointegrated_system
)

from src.optimal_controls.z_spread_model_parameters import  (
    ZSpreadModelParameters
)

from src.optimal_controls.z_spread_model_solver import (
    ZSpreadModelSolver
)

def main():

    corr_mat = np.array([[1.0, 0.6, 0.6],
                         [0.6, 1.0, 0.6],
                         [0.6, 0.6, 1.0]])

    params = {
        'A': SystemParams(s_0=100, rho_0=0.7, mu_i=0.25, sigma_i=0.25, beta_i=-10, delta_i=1, a=1, b=0),
        'B': SystemParams(s_0=100, rho_0=0.7, mu_i=0.25, sigma_i=0.25, beta_i=-15, delta_i=2, a=2, b=0),
        'C': SystemParams(s_0=100, rho_0=0.7, mu_i=0.25, sigma_i=0.25, beta_i=-5, delta_i=3, a=3, b=0),
    }

    ln_s_0, ln_s_i, z = simulate_benchmark_cointegrated_system(params, 100, 0, 0.15, 1/250, corr_mat, 10000)

    params = ZSpreadModelParameters.estimate_from_ln_prices(ln_s_0, ln_s_i)

    print(" ")

    model = ZSpreadModelSolver.solve(params, 50, 10000)


    n_calc = 500
    holding = np.zeros_like(ln_s_i)
    for i in range(0, n_calc):
        holding[i, :] = model.optimal_portfolio(z[i, :].reshape(-1, 1), 25).reshape(1, -1)

    dln_s_i = np.diff(ln_s_i, 1, axis=0)
    pnl = holding[0:-1, :]*dln_s_i

    fig, ax = plt.subplots(4, 1, figsize=(8, 6))
    ax[0].plot(ln_s_i[0:n_calc, 0], color='blue')
    ax_0_2 = ax[0].twinx()
    ax_0_2.plot(ln_s_0[0:n_calc], color='red')
    ax[1].plot(z[0:n_calc, 0])
    ax[2].plot(holding[0:n_calc, :])
    ax[3].plot(np.cumsum(pnl, axis=0)[0:n_calc, :])
    ax[3].plot(np.sum(np.cumsum(pnl, axis=0)[0:n_calc, :],axis=1), color='black')
    plt.show()



if __name__ == '__main__':
    main()