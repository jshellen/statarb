
import numpy as np


def sim_ou(x_0, k, theta, sigma, dt, n_steps):
    """
    Simulate Ornstein-Uhlenbeck process
    """
    x = np.zeros(n_steps)
    x[0] = x_0

    for i in range(1, n_steps):
        x[i] = x[i-1] + k*(theta - x[i-1])*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1)

    return x


def ornstein_uhlenbeck(X_0, k, theta, sigma, dt, n_steps, n_paths):
    """
    Simulate Ornstein-Uhlenbeck process
    """
    size = (n_steps, n_paths)
    x = np.zeros(size)

    for j in range(0, n_paths):
        x[:, j] = sim_ou(X_0, k, theta, sigma, dt, n_steps)

    return x