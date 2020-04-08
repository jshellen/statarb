import numpy as np
import pandas as pd
import padasip

from src.simulation.ornstein_uhlenbeck import (
    simulate_one_ornstein_uhlenbeck_path
)

from src.estimation.rls import (
    RLSFilter2,
    system_identification_setup
)


def RLS(x, d, y, n, mu, s):

    a = np.zeros([n, d])
    r_1 = np.linalg.inv(np.matmul(x[0:s, :].T, x[0:s, :]))
    r_2 = np.matmul(x[0:s, :].T, y[0:s].reshape(-1, 1))
    w_0 = np.dot(r_1, r_2)
    a[0] = w_0.flatten()
    P = np.zeros([n, d, d])
    P[0] = np.linalg.inv(np.dot(x[0:s, :].T, x[0:s, :]))
    for t in range(1, n):
        xt = np.reshape(x[t], [1, d])
        e = y[t] - np.dot(xt, np.reshape(a[t - 1], [d, 1]))
        k = np.dot(P[t - 1], xt.T) / (mu + np.linalg.multi_dot([xt, P[t - 1], xt.T]))
        a[t] = a[t - 1] + np.dot(k, e).T
        P[t] = (1 / mu) * (P[t - 1] - np.linalg.multi_dot([k, xt, P[t - 1]]))
    return a


def test1():

    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    np.random.seed(5)

    n_steps = 500
    dt = 1.0 / n_steps
    x = np.zeros((n_steps, 1))
    for i in range(2, n_steps):
        x[i] = 0.01 * dt + 0.99 * x[i - 1] + 0.25 * np.sqrt(dt) * np.random.normal(0, 1)

    x_ = sm.add_constant(np.roll(x, 1)[1:])
    y_ = x[1:]

    b = RLS(x_, 2, y_, len(x_), 0.94, 20)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(b, label='RLS', color='red')
    plt.suptitle('Recursive Least Squares')
    plt.legend()
    plt.show()


def main():

    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    """
    Simulate Ornstein-Uhlenbeck process
    """
    #n_step = 1500
    #dt = 1.0/n_step
    #x = simulate_one_ornstein_uhlenbeck_path(0, 5.5, 0, 0.14, dt, n_step)
    n_steps = 500
    dt = 1.0 / n_steps
    x = np.zeros((n_steps, 1))
    for i in range(2, n_steps):
        x[i] = 0.01 * dt + 0.5 * x[i - 1] + 0.25 * np.sqrt(dt) * np.random.normal(0, 1)

    """
    Parameter estimation
    """
    x_ = sm.add_constant(np.roll(x, 1)[1:])
    y_ = x[1:]

    # Padasip filter
    F_1 = padasip.filters.FilterRLS(2, mu=0.997, eps=1e-2, w="zeros")
    y, e, w_hat_1 = F_1.run(y_, x_)

    # Feedforward implementation
    F_2 = RLSFilter2(p=1, lmbda=0.997, delta=1e-2)
    w_hat_2 = np.array([F_2.ff_fb(x_i[0]) for x_i in x])

    # Naive implementation
    w_hat_3 = RLS(x_, 2, y_, len(x_), 0.997, 10)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.title('Time-Dependent ARMA RLSE')
    ax.plot(w_hat_1, color='blue', ls='-', lw=2)
    ax.plot(w_hat_2, color='green', ls='-', lw=2)
    ax.plot(w_hat_3, color='red', ls='-', lw=2)
    ax.set_ylabel(r'$\beta(t)$', fontsize=15)
    plt.show()


if __name__ == '__main__':
    main()