
import matplotlib.pyplot as plt
import numpy as np

from src.simulation.ornstein_uhlenbeck import (
    simulate_one_ornstein_uhlenbeck_path
)

from src.optimal_controls.estimation.parameter_estimation import (

    estimate_ou_parameters_using_lsq
)


def main():

    sample_size = 50000
    sigma_e = 3.0  # true value of parameter error sigma
    random_num_generator = np.random.RandomState(0)
    x = 10.0 * random_num_generator.rand(sample_size)
    e = random_num_generator.normal(0, sigma_e, sample_size)
    a = 1.0
    b = 2.0
    y = a + b * x + e  # a = 1.0; b = 2.0; y = a + b*x

    a_0 = 0.5
    b_0 = 0.5
    sigma_a_0 = 0.25
    sigma_b_0 = 0.5
    beta_0 = np.array([[a_0], [b_0]])
    sigma_beta_0 = np.array([[sigma_a_0 * sigma_a_0, 0], [0, sigma_b_0 * sigma_b_0]])

    beta_recorder = []  # record parameter beta
    beta_recorder.append(beta_0)
    for pair in range(int(sample_size/2)):  # 500 points means 250 pairs
        x1 = x[pair * 2]
        x2 = x[pair * 2 + 1]
        y1 = y[pair * 2]
        y2 = y[pair * 2 + 1]
        mu_y = np.array([[(x1 * y2 - x2 * y1) / (x1 - x2)], [(y1 - y2) / (x1 - x2)]])
        sigma_y = np.array([[(np.square(x1 / (x1 - x2)) + np.square(x2 / (x1 - x2))) * np.square(sigma_e), 0],
                            [0, 2 * np.square(sigma_e / (x1 - x2))]])
        sigma_beta_1 = np.linalg.inv(np.linalg.inv(sigma_beta_0) + np.linalg.inv(sigma_y))
        beta_1 = sigma_beta_1.dot(np.linalg.inv(sigma_beta_0).dot(beta_0) + np.linalg.inv(sigma_y).dot(mu_y))

        # assign beta_1 to beta_0
        beta_0 = beta_1
        sigma_beta_0 = sigma_beta_1
        beta_recorder.append(beta_0)

    print('pamameters: %.7f, %.7f' % (beta_0[0], beta_0[1]))

    # plot the Beyesian dynamics
    xfit = np.linspace(0, 10, sample_size)
    ytrue = 2.0 * xfit + 1.0  # we know the true value of slope and intercept
    plt.plot(xfit, ytrue, label='true line', linewidth=3)
    y0 = beta_recorder[0][1] * xfit + beta_recorder[0][0]
    plt.plot(xfit, y0, label='initial belief', linewidth=1)
    y1 = beta_recorder[1][1] * xfit + beta_recorder[1][0]
    plt.plot(xfit, y1, label='1st update', linewidth=1)
    y10 = beta_recorder[10][1] * xfit + beta_recorder[10][0]
    plt.plot(xfit, y10, label='10th update', linewidth=1)
    y100 = beta_recorder[100][1] * xfit + beta_recorder[100][0]
    plt.plot(xfit, y100, label='100th update', linewidth=1)
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot([beta_recorder[i][0] for i in range(0, len(beta_recorder))])
    ax[0].axhline(y=a, lw=2, color='red')

    ax[1].plot([beta_recorder[i][1] for i in range(0, len(beta_recorder))])
    ax[1].axhline(y=b, lw=2, color='red')
    plt.show()


def ou_test():

    dt = 1.0/250.0
    sample_size = 50000

    x_ = simulate_one_ornstein_uhlenbeck_path(5, 5.5, 0, 0.5, dt, sample_size)

    sigma_e = 0.1*np.std(x_)

    kappa_, theta_, sigma_, a_, b_ = estimate_ou_parameters_using_lsq(x_, dt)

    x = np.roll(x_, 1)[1:]
    y = x_[1:]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y)

    a_0 = 0.02
    b_0 = 0.02
    sigma_a_0 = 2.25
    sigma_b_0 = 2.25
    beta_0 = np.array([[a_0], [b_0]])
    sigma_beta_0 = np.array([[sigma_a_0 * sigma_a_0, 0], [0, sigma_b_0 * sigma_b_0]])

    param_path = []
    beta_recorder = []  # record parameter beta
    beta_recorder.append(beta_0)
    for pair in range(int(len(y) / 2)):  # 500 points means 250 pairs
        x1 = x[pair * 2]
        x2 = x[pair * 2 + 1]
        y1 = y[pair * 2]
        y2 = y[pair * 2 + 1]
        mu_y = np.array([[(x1 * y2 - x2 * y1) / (x1 - x2)], [(y1 - y2) / (x1 - x2)]])
        sigma_y = np.array([[(np.square(x1 / (x1 - x2)) + np.square(x2 / (x1 - x2))) * np.square(sigma_e), 0],
                            [0, 2 * np.square(sigma_e / (x1 - x2))]])
        sigma_beta_1 = np.linalg.inv(np.linalg.inv(sigma_beta_0) + np.linalg.inv(sigma_y))
        beta_1 = sigma_beta_1.dot(np.linalg.inv(sigma_beta_0).dot(beta_0) + np.linalg.inv(sigma_y).dot(mu_y))

        # assign beta_1 to beta_0
        beta_0 = beta_1
        sigma_beta_0 = sigma_beta_1
        beta_recorder.append(beta_1)

        a = beta_1[1]
        b = beta_1[0]
        kappa = -np.log(a) #/ dt
        theta = b / (1 - a)

        param_path.append([kappa, theta])


    fig, ax = plt.subplots(2, 2, figsize=(12, 4))

    ax[0, 0].plot([beta_recorder[i][0] for i in range(500, len(beta_recorder))])
    ax[0, 0].axhline(y=b_)

    ax[0, 1].plot([beta_recorder[i][1] for i in range(500, len(beta_recorder))])
    ax[0, 1].axhline(y=b_)

    plt.show()

    print(" ")


if __name__ == '__main__':
    ou_test()