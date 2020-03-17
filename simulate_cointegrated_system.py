import numpy as np
import matplotlib.pyplot as plt

from src.simulation.simulate_cointegrated_assets import (
    SystemParams,
    simulate_benchmark_cointegrated_system
)

def main():

    corr_mat = np.array([[1.0, 0.6, 0.6],
                         [0.6, 1.0, 0.6],
                         [0.6, 0.6, 1.0]])

    params = {
        'A': SystemParams(s_0=100, rho_0=0.5, mu_i=0.25, sigma_i=0.25, beta_i=-10, delta_i=0.1, a=0, b=0),
        'B': SystemParams(s_0=100, rho_0=0.7, mu_i=0, sigma_i=0.15, beta_i=-15, delta_i=0.1, a=0, b=0),
        'C': SystemParams(s_0=100, rho_0=0.5, mu_i=0, sigma_i=0.10, beta_i=-20, delta_i=0.1, a=0, b=0),
    }

    ln_s_0, ln_s_i, z = simulate_benchmark_cointegrated_system(params, 100, 0, 0.15, 1/500, corr_mat, 10000)

    fig, ax = plt.subplots(3, 1, figsize=(8, 4))
    ax[0].plot(ln_s_i[:, 0], color='blue')
    ax[1].plot(ln_s_0 + params['A'].beta_i*ln_s_i[:, 0], color='green')
    ax[2].plot(z[:, 0])
    plt.show()




if __name__ == '__main__':
    main()