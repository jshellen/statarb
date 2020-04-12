import numpy as np
import pandas as pd
import padasip

from src.simulation.ornstein_uhlenbeck import (
    simulate_one_ornstein_uhlenbeck_path
)

class KalmanFilter(object):

    def __init__(self, dt, u, std_acc, std_meas):
        self.dt = dt
        self.u = u
        self.std_acc = std_acc
        self.A = np.array([[1, self.dt],
                           [0,       1]
                           ])

        self.B = np.array([[(self.dt**2)/2], [self.dt]])
        self.H = np.array([[1, 0]])
        self.Q = np.array([[(self.dt**4)/4, (self.dt**3)/2],
                            [(self.dt**3)/2, self.dt**2]]) * self.std_acc**2
        self.R = std_meas**2
        self.P = np.eye(self.A.shape[1])
        self.x = np.array([[0], [0]])

    def predict(self):
        # Ref :Eq.(9) and Eq.(10)
        # Update time state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Calculate error covariance
        # P= A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        # Ref :Eq.(11) , Eq.(11) and Eq.(13)
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Eq.(11)
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))  # Eq.(12)
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P  # Eq.(13)

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
    F_1 = padasip.filters.FilterRLS(2, mu=0.997, eps=0.99999, w_0=w_0.reshape(-1, 1))
    w_1 = np.zeros((len(y_), 2))
    w_1[0, :] = w_0
    for i in range(1, len(y_)):
        F_1.update(y_[i][0], x_[i])
        w_1[i, :] = F_1.w.flatten()

    u = 0
    std_acc = 5.00  # we assume that the standard deviation of the acceleration is 0.25 (m/s^2)
    std_meas = 0.0001  # and standard deviation of the measurement is 1.2 (m)
    kf = KalmanFilter(dt, u, std_acc, std_meas)
    predictions = []
    for i in range(0, len(w_1)):
        predictions.append(kf.predict()[0])
        kf.update(w_1[i, 0])

    # Naive implementation
    w_3 = RLS(x_, 2, y_, len(x_), 0.997, 10)

    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    ax[0].plot(w_1[10:, 0], color='blue', ls='-', lw=2)
    ax[0].plot(w_3[10:, 0], color='red', ls='-', lw=2)

    ax[1].plot(w_1[10:, 1], color='blue', ls='-', lw=2)
    ax[1].plot(w_3[10:, 1], color='red', ls='-', lw=2)
    ax[2].plot(predictions, color='red', ls='-', lw=2)
    plt.show()


if __name__ == '__main__':
    main()