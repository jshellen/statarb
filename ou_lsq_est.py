import matplotlib.pyplot as plt
import numpy as np
<<<<<<< HEAD:ou_lsq_est.py
=======
import pandas as pd
>>>>>>> a3507c187b5b0d726d1e5d3ff8642e728fccec0a:out_lsq_est.py

from src.simulation.ornstein_uhlenbeck import (
    sim_ou
)

from src.optimal_controls.estimation.parameter_estimation import (
    estimate_ou_parameters_using_lsq
)


def sample_estimation_error(kappa_true, sigma, dt, n_grid):

    n_samples = 50
    bias_n = np.zeros(len(n_grid))
    kappa_n = np.zeros(len(n_grid))
    for i in range(0, len(n_grid)):
        bias_sum = 0
        kappa_sum = 0
        # Sample estimation error
        for j in range(0, n_samples):
            # Simulate ou process
            x = sim_ou(0, kappa_true, 0, sigma, dt, n_grid[i])
            # Estimate parameters
            kappa_est, theta_est, sigma_est = estimate_ou_parameters_using_lsq(x, dt)
<<<<<<< HEAD:ou_lsq_est.py
            # Error
            bias = kappa_est - kappa_true
            bias_sum += bias
            kappa_sum += kappa_est
=======


            if kappa_est is not None:
                # Error
                bias = kappa_est - kappa_true
                bias_sum += bias
                kappa_sum += kappa_est
>>>>>>> a3507c187b5b0d726d1e5d3ff8642e728fccec0a:out_lsq_est.py
        # Compute mean error
        bias_n[i] = bias_sum / float(n_samples)
        kappa_n[i] = kappa_sum / float(n_samples)

    return kappa_n, bias_n



def ou_bias(n, dt):

    if not isinstance(n, (int, np.int32, np.int64)):
        raise ValueError(f'n has to be integer. {type(n)}')

    return 1372.96281*(1 + 1.0/n) - 1373.2467


def ou_bias2(n):

    if not isinstance(n, (int, np.int32, np.int64)):
        raise ValueError(f'n has to be integer. {type(n)}')

    return 29.70381061*(1/(n*0.0189243637761786))

def main():

    n_grid = np.arange(50, 1500, 50)

    # Sample estimation error with different kappa parameters
    dt = 1.0/250.0
    k_1, e_1 = sample_estimation_error(1.5, 0.25, dt, n_grid)
    k_2, e_2 = sample_estimation_error(0.5, 0.5, dt, n_grid)

    #k_1 = pd.DataFrame(data=k_1)
    #k_2 = pd.DataFrame(data=k_2)

    #k_1.to_excel('k_1.xlsx')
    #k_2.to_excel('k_2.xlsx')

    k_1_unbiased = np.array([k_1[i] - ou_bias2(n_grid[i]) for i in range(0, len(n_grid))])
    k_2_unbiased = np.array([k_2[i] - ou_bias2(n_grid[i]) for i in range(0, len(n_grid))])

    #params, _ = curve_fit(ou_bias, n_grid, e_1)
    #a = params[0]
    #b = params[1]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].plot(n_grid, [ou_bias2(n) for n in n_grid], color='black')
    ax[0].scatter(n_grid, e_1, color='blue')
    ax[0].scatter(n_grid, e_2, color='red')

    ax[1].scatter(n_grid, k_1, color='blue')
    ax[1].scatter(n_grid, k_2, color='red')

    ax[2].scatter(n_grid, k_1_unbiased, color='blue')
    ax[2].axhline(y=np.mean(k_1_unbiased), color='blue')
    ax[2].scatter(n_grid, k_2_unbiased, color='red')
    ax[2].axhline(y=np.mean(k_2_unbiased), color='red')

    plt.show()

if __name__ == '__main__':
    main()
