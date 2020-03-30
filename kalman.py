
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_one_ornstein_uhlenbeck_path(x_0, k, theta, sigma, dt, n_steps):
    """
    Simulate Ornstein-Uhlenbeck process
    """
    x = np.zeros(n_steps)
    x[0] = x_0

    for i in range(1, n_steps):
        x[i] = x[i-1] + k*(theta - x[i-1])*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1)

    return x


def main():

    dt = 1.0 / 250.0
    sample_size = 5000

    x_ = simulate_one_ornstein_uhlenbeck_path(5, 5.5, 0, 0.5, dt, sample_size)

    sigma_e = 0.001

    x = np.roll(x_, 1)[1:]
    y = x_[1:]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y)

    N = int(len(y)/2)
    # initial value
    theta_0_0 = np.array([[2.5], [2.5]])  # 2x1 array
    W = np.array([[0, 0], [0, 0]])  # 2x2 array
    P_0_0 = W
    lr = 0.1
    results = np.zeros([N, 2])
    for k in range(N):  # 250 pairs
        print('step {}'.format(k))
        # A-Priori prediction
        # first step, let k = 1
        theta_1_0 = theta_0_0
        P_1_0 = P_0_0 + W

        # After observing two points (x1, y1) and (x2, y2)
        x1 = x[2 * k + 0]
        x2 = x[2 * k + 1]
        y1 = y[2 * k + 0]
        y2 = y[2 * k + 1]
        y_1 = np.array([y1, y2]).reshape(2, 1)
        F_1 = np.array([[1, x1], [1, x2]])
        y_1_tilde = y_1 - np.dot(F_1, theta_1_0)

        # residual covariance
        V_1 = np.array([[sigma_e, 0], [0, sigma_e]])
        S_1 = np.dot(np.dot(F_1, P_1_0), np.transpose(F_1)) + V_1

        # Kalman Gain
        K_1 = np.dot(np.dot(P_1_0, np.transpose(F_1)), np.linalg.inv(S_1))

        # Posterior
        theta_1_1 = theta_1_0 + lr * np.dot(K_1, y_1_tilde)
        P_1_1 = np.eye(2) - np.dot(np.dot(K_1, F_1), P_1_0)

        # assign for next iteration
        results[k, :] = theta_1_1.reshape(2, )
        theta_0_0 = theta_1_1
        P_0_0 = P_1_1

    results = pd.DataFrame(data=results, columns=['a', 'b'])

    # present the results
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax1.plot(results['a'])

    ax2 = fig.add_subplot(122)
    ax2.plot(results['b'])
    plt.show()

if __name__ == '__main__':
    main()