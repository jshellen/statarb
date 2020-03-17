import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from src.simulation.simulate_cointegrated_assets import (
    SystemParams,
    simulate_benchmark_cointegrated_system
)

from src.estimation.parameter_estimation import (
    estimate_ln_coint_params
)

from src.estimation.coint_johansen import Johansen


def main():

    corr_mat = np.array([[1.0, 0.6, 0.6],
                         [0.6, 1.0, 0.6],
                         [0.6, 0.6, 1.0]])

    params = {
        'A': SystemParams(s_0=100, rho_0=0.5, mu_i=0.25, sigma_i=0.25, beta_i=-30, delta_i=2, a=0, b=0),
        'B': SystemParams(s_0=100, rho_0=0.7, mu_i=0.00, sigma_i=0.15, beta_i=-15, delta_i=1, a=0, b=0),
        'C': SystemParams(s_0=100, rho_0=0.5, mu_i=0.00, sigma_i=0.10, beta_i=-20, delta_i=1, a=0, b=0),
    }

    ln_s_0, ln_s_i, z = simulate_benchmark_cointegrated_system(params, 100, 0, 0.15, 1/250, corr_mat, 10000)

    #fig, ax = plt.subplots(3, 1, figsize=(8, 4))
    #ax[0].plot(ln_s_i[:, 0], color='blue')
    #ax[1].plot(ln_s_0 + params['A'].beta_i*ln_s_i[:, 0], color='green')
    #ax[2].plot(z[:, 0])
    #plt.show()

    #ols_est = sm.OLS(ln_s_0.reshape(-1, 1), sm.add_constant(ln_s_i[:, 0].reshape(-1, 1))).fit()
    #print(ols_est.summary())

    #estimator = Johansen(np.concatenate([ln_s_0.reshape(-1, 1), ln_s_i[:, 0].reshape(-1, 1)], axis=1), model=2, significance_level=0)
    #e_, r = estimator.johansen()
    #e = e_[:, 0] / e_[0, 0]

    delta, beta, kappa = estimate_ln_coint_params(ln_s_0, ln_s_i[:, 0], 1/250)

    print(beta, kappa, delta)

    #print(" ")

    delta, beta, kappa = estimate_ln_coint_params(ln_s_0, ln_s_i[:, 1], 1/250)

    print(beta, kappa, delta)


if __name__ == '__main__':
    main()