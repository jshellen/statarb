
from src.simulation.ornstein_uhlenbeck import (
    simulate_one_ornstein_uhlenbeck_path
)

from src.optimal_controls.estimation.seqols import (
    SequentialLinearRegression
)


def main():

    import numpy as np
    import matplotlib.pyplot as plt

    n_step = 5000
    dt = 1.0/250.0
    x_1 = simulate_one_ornstein_uhlenbeck_path(0, 1.5, 0, 0.14, dt, n_step)
    x_2 = simulate_one_ornstein_uhlenbeck_path(x_1[-1], 1.5, 0, 0.14, dt, n_step)

    x = np.concatenate([x_1, x_2], axis=0)

    seqols = SequentialLinearRegression(0.999, True, 1)

    x_ = np.roll(x, 1)[1:]
    y_ = x[1:]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_, y_)
    plt.show()

    coefs = np.zeros((len(x_), 2))
    params = np.zeros((len(x_), 2))
    for i in range(1, len(x_)):

        seqols.add_obs(x_[i].reshape(-1, 1), y_[i])

        coefs[i, :] = seqols.m_coefs.flatten()

        a = seqols.m_coefs[0][1]
        b = seqols.m_coefs[0][0]

        kappa = None
        if a > 0:
            kappa = -np.log(a)/dt
        theta = b/(1 - a)
        params[i, 0] = kappa
        params[i, 1] = theta

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    ax[0].plot(coefs[200:, 0])
    ax[1].plot(coefs[200:, 1])

    fig, ax = plt.subplots(3, 1, figsize=(6, 6))
    ax[0].plot(seqols.m_e[250:])
    ax[1].plot(params[250:, 0])
    ax[2].plot(params[250:, 1])
    plt.show()


if __name__ == '__main__':
    main()