
from src.simulation.ornstein_uhlenbeck import (
    simulate_one_ornstein_uhlenbeck_path
)

from src.estimation.kalman_filter import (
    kalman_filter_predict,
    kalman_filter_update
)


def main():

    import numpy as np
    import matplotlib.pyplot as plt

    n_step = 250
    dt = 1.0/250.0
    x = simulate_one_ornstein_uhlenbeck_path(0, 1.5, 0, 0.14, dt, n_step)

    x_ = np.roll(x, 1)[1:]
    y_ = x[1:]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_, y_)
    plt.show()

    # ---

    # Initialization of state matrices
    X = np.array([[1.0, 0.0]]).reshape(-1, 1)
    delta = 0.9
    P = np.eye(2)
    A = np.array([[1, 0],
                  [0, 1]])
    Q = np.zeros(X.shape)
    B = np.zeros(X.shape).reshape(1, -1)
    U = np.zeros((X.shape[0], 1))
    R = 5
    X_t = np.zeros((n_step-1, 2))
    for i in np.arange(0, n_step-1):
        Y = np.array([[y_[i]]])
        H = np.array([x_[i], 1]).reshape(1, -1)
        (X, P) = kalman_filter_predict(X, P, A, Q, B, U)
        (X, P, K, IM, IS, LH) = kalman_filter_update(X, P, Y, H, R)
        X_t[i, :] = X.flatten()

    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    ax[0, 0].plot(X_t[:, 0], color='blue', lw=4)
    ax[0, 1].plot(-np.log(X_t[:, 0])/dt, color='blue', lw=4)
    ax[1, 0].plot(X_t[:, 1], color='blue', lw=4)
    ax[1, 1].plot(X_t[:, 1]/(1 - X_t[:, 0]), color='blue', lw=4)
    plt.show()



if __name__ == '__main__':
    main()