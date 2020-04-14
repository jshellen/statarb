import numpy as np

from src.estimation.rls import (
    RLSFilter
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


def main():

    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    n_steps = 1500
    dt = 1.0 / 250.0
    x = np.zeros((n_steps, 1))
    for i in range(1, n_steps):
        x[i] = 0.1 + 0.75 * x[i - 1] + 0.05 * np.random.normal(0, 1)

    x_ = sm.add_constant(np.roll(x, 1)[1:])
    y_ = x[1:]

    # Padasip filter
    w_0 = np.array([1, 0.1])
    F_1 = RLSFilter(2, mu=0.997, eps=0.99999, w_0=w_0.reshape(-1, 1))
    w_1 = np.zeros((len(y_), 2))
    w_1[0, :] = w_0
    for i in range(1, len(y_)):
        F_1.update(y_[i][0], x_[i])
        w_1[i, :] = F_1.w.flatten()

    # Naive implementation
    w_3 = RLS(x_, 2, y_, len(x_), 0.997, 10)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(w_1[10:, 0], color='blue', ls='-', lw=2, label='IG-RLS')
    ax[0].plot(w_3[10:, 0], color='red', ls='-', lw=2, label='NIG-RLS')

    ax[1].plot(w_1[10:, 1], color='blue', ls='-', lw=2, label='IG-RLS')
    ax[1].plot(w_3[10:, 1], color='red', ls='-', lw=2, label='NIG-RLS')
    plt.show()


if __name__ == '__main__':
    main()