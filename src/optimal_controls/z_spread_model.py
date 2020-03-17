import numpy as np
import matplotlib.pyplot as plt


class MultiSpreadModelParameters:

    def __init__(self, gamma, rho, rho_0, sigma_0, sigma, mu_0, mu, beta, delta, b, a):

        self.m_gamma = gamma
        self.m_rho = rho
        self.m_rho_0 = rho_0
        self.m_sigma_0 = sigma_0
        self.m_sigma = sigma
        self.m_mu_0 = mu_0
        self.m_mu = mu
        self.m_beta = beta
        self.m_delta = delta
        self.m_b = b
        self.m_a = a

        self.m_kappa = np.zeros_like(self.m_delta)
        for i in range(0, len(self.m_kappa)):
            self.m_kappa[i] = - self.m_beta[i] * self.m_delta[i]

        self.m_sigma_1 = np.matmul(np.matmul(np.diag(self.m_sigma.flatten()), self.m_rho), np.diag(self.m_sigma.flatten()))

        self.m_sigma_2 = np.zeros_like(self.m_sigma_1)
        for i in range(0, self.m_sigma_1.shape[0]):
            for j in range(0, self.m_sigma_1.shape[1]):
                self.m_sigma_2[i, j] = self.m_sigma_0 * self.m_sigma[i] * self.m_rho_0[i] \
                                + self.m_beta[j] * self.m_sigma_1[i, j]

        self.m_sigma_3 = np.zeros_like(self.m_sigma_1)
        for i in range(0, self.m_sigma_1.shape[0]):
            for j in range(0, self.m_sigma_1.shape[1]):
                self.m_sigma_3[i, j] = (
                        self.m_sigma_0**2
                        + self.m_sigma_0 * self.m_sigma[i]
                        * self.m_beta[i] * self.m_rho_0[i]
                        + self.m_sigma_0 * self.m_sigma[j]
                        * self.m_beta[j] * self.m_rho_0[j]
                        + self.m_sigma[i] * self.m_beta[i]
                        * self.m_sigma[j] * self.m_beta[j] * self.m_rho[i, j])

        self.m_theta = np.zeros_like(self.m_delta)
        for i in range(0, self.m_sigma_1.shape[0]):
            #self.m_theta[i] = a[i]
             self.m_theta[i] = (self.m_beta[i] + self.m_mu_0 + self.m_beta[i] * self.m_mu[i]
                                - 0.5 * (self.m_sigma_0**2 + self.m_beta[i] * self.m_sigma[i]**2))/(self.m_beta[i]*self.m_delta[i])

        print(" ")


class MultiSpreadModelSolver:

    @staticmethod
    def solve(params, T, n_step):
        """

        :param params:
        :param T:
        :param n_step:
        :return:
        """
        if not isinstance(params, MultiSpreadModelParameters):
            raise TypeError('Parameters have to be type of <MultiSpreadModelParameters>.')

        if not isinstance(T, (int, float)):
            raise TypeError('T has to be type of <int> or <float>.')

        if not isinstance(n_step, int):
            raise TypeError('n_step has to be type of <int>.')

        if T <= 0:
            raise ValueError('T has to be non-negative.')

        if n_step <= 0:
            raise ValueError('n_step has to be non-negative.')

        c_t = MultiSpreadModelSolver._solve_c(params, T, n_step)

        b_t = MultiSpreadModelSolver._solve_b(params, c_t, T, n_step)

        return MultiSpreadModel(params, b_t, c_t, T, n_step)

    @staticmethod
    def _solve_b(params, c_t, T, n_step):
        """

        :param params:
        :param c:
        :param T:
        :param n_step:
        :return:
        """

        if not isinstance(params, MultiSpreadModelParameters):
            raise TypeError('Parameters have to be type of <MultiSpreadModelParameters>.')

        if not isinstance(T, (int, float)):
            raise TypeError('T has to be type of <int> or <float>.')

        if not isinstance(n_step, int):
            raise TypeError('n_step has to be type of <int>.')

        if not isinstance(c_t, dict):
            raise TypeError('c_t has to be type of <dict>.')

        if T <= 0:
            raise ValueError('T has to be non-negative.')

        if n_step <= 0:
            raise ValueError('n_step has to be non-negative.')

        dt = T / float(n_step)

        b_T = np.zeros(params.m_rho.shape[0]).reshape(-1, 1)

        b_t = {}
        b_t.update({n_step: b_T})

        Q_t = params.m_gamma/(1.0 - params.m_gamma) * np.matmul(
            np.matmul(params.m_sigma_2.T,
                      np.linalg.inv(params.m_sigma_1)), params.m_sigma_2
        ) + params.m_sigma_3

        A_t = params.m_gamma/(1.0 - params.m_gamma) * np.matmul(
            np.matmul(np.diag(params.m_delta.flatten()),
                      np.linalg.inv(params.m_sigma_1)), params.m_sigma_2
        ) - np.diag(params.m_kappa)

        P_t = params.m_gamma/(1.0 - params.m_gamma) * np.matmul(
            np.matmul(params.m_sigma_2.T, np.linalg.inv(params.m_sigma_1)
                      ), params.m_mu
        ) + np.matmul(np.diag(params.m_kappa.flatten()), params.m_theta)

        O_t = params.m_gamma/(1.0 - params.m_gamma) * np.matmul(
            np.matmul(np.diag(params.m_delta.flatten()),
                      np.linalg.inv(params.m_sigma_1)), params.m_mu )

        # Compute b_t
        b_t_ = b_T
        for i in range(n_step - 1, -1, -1):

            db_t = -2 * np.matmul(np.matmul(c_t[i], Q_t), b_t_)\
                   - np.matmul(A_t, b_t_)\
                   - 2*np.matmul(c_t[i], P_t) - O_t

            b_t_ = b_t_ + db_t * dt

            b_t.update({i: b_t_})

        return b_t

    @staticmethod
    def _solve_c(params, T, n_step):
        """

        :param params:
        :param T:
        :param n_step:
        :return:
        """

        if not isinstance(params, MultiSpreadModelParameters):
            raise TypeError('Parameters have to be type of <MultiSpreadModelParameters>.')

        if not isinstance(T, (int, float)):
            raise TypeError('T has to be type of <int> or <float>.')

        if not isinstance(n_step, int):
            raise TypeError('n_step has to be type of <int>.')

        if T <= 0:
            raise ValueError('T has to be non-negative.')

        if n_step <= 0:
            raise ValueError('n_step has to be non-negative.')

        dt = T / float(n_step)

        c_T = np.zeros_like(params.m_rho)

        c_t = {}
        c_t.update({n_step: c_T})

        Q_u = (2 * params.m_gamma)/(1.0-params.m_gamma) * np.matmul(
                            np.matmul(params.m_sigma_2.T,
                                      np.linalg.inv(params.m_sigma_1)), params.m_sigma_2
        ) + 2 * params.m_sigma_3

        A_u = params.m_gamma/(1.0-params.m_gamma) * np.matmul(
                    np.matmul(params.m_sigma_2.T,
                              np.linalg.inv(params.m_sigma_1)), np.diag(params.m_delta.flatten())
        ) - np.diag(params.m_kappa)

        P_u = params.m_gamma/(2.0 * (1.0-params.m_gamma)) * np.matmul(
            np.matmul(np.diag(params.m_delta.flatten()),
                      np.linalg.inv(params.m_sigma_1)), np.diag(params.m_delta.flatten()))

        # Compute C_t
        c_t_ = c_T
        for i in range(n_step - 1, -1, -1):

            dc_t = - np.matmul(A_u.T, c_t_) - np.matmul(c_t_, A_u)\
                 - np.matmul(np.matmul(c_t_, Q_u), c_t_) - P_u

            c_t_ = c_t_ + dc_t * dt


            c_t.update({i: c_t_})

        return c_t


class MultiSpreadModel:

    def __init__(self, params, b_t, c_t, T, n_step):

        self.m_params = params
        self.m_b_t = b_t
        self.m_c_t = c_t
        self.m_time_grid = np.linspace(0, T, n_step)

    def optimal_portfolio(self, z, t):

        b_t = self.get_b_t(t)
        c_t = self.get_c_t(t)

        if b_t is None:
            raise ValueError('b_t was None.')
        if c_t is None:
            raise ValueError('c_t was None.')

        A = 1.0/(1.0 - self.m_params.m_gamma) * np.matmul(
            np.linalg.inv(self.m_params.m_sigma_1),
            (self.m_params.m_mu + np.matmul(np.diag(self.m_params.m_delta.flatten()), z)))

        B = 1.0/(1.0 - self.m_params.m_gamma) * np.matmul(np.matmul(np.linalg.inv(self.m_params.m_sigma_1),
                      self.m_params.m_sigma_2), 2*np.matmul(c_t, z) + b_t)

        result = A + B

        return result

    def get_b_t(self, t):
        """

        :param t:
        :return:
        """
        result = None
        for i, t_p in enumerate(self.m_time_grid):
            if t_p >= t:
                result = self.m_b_t[i]
        return result

    def get_c_t(self, t):
        """

        :param t:
        :return:
        """
        result = None
        for i, t_p in enumerate(self.m_time_grid):
            if t_p >= t:
                result = self.m_c_t[i]
        return result

    def plot_c_t(self):

        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(0, self.m_params.m_rho.shape[0]):
            for j in range(0, self.m_params.m_rho.shape[0]):
                ax.plot([c_t[i, j] for k, c_t in self.m_c_t.items()])
        plt.show()

    def plot_b_t(self):

        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(0, self.m_params.m_rho.shape[0]):
            ax.plot([c_t[i] for k, c_t in self.m_b_t.items()])
        plt.show()



def main():

    mu_0 = 0
    sigma_0 = 0.22

    rho = np.array([
        [1.0, 0.6, 0.6],
        [0.6, 1.0, 0.6],
        [0.6, 0.6, 1.0]
    ])

    rho_0 = np.array([0.5, 0.5, 0.5])
    sigma = np.array([0.25, 0.25, 0.25])
    beta = np.array([-15, -10, -5])
    mu = np.array([0, 0, 0])
    delta = np.array([0.1, 0.1, 0.1])

    kappa = np.zeros_like(delta)
    for i in range(0, len(kappa)):
        kappa[i] = - beta[i] * delta[i]

    b = np.array([0, 0, 0])
    a = np.array([0, 0, 0])


    params = MultiSpreadModelParameters(-1, rho, rho_0, sigma_0, sigma, mu_0, mu, beta, delta, b, a)

    model = MultiSpreadModelSolver.solve(params, 50, 2000)


    model.plot_b_t()

    model.plot_c_t()

    print(model.get_c_t(1.0))

    z = np.array([10, 0, 0])

    print(model.optimal_portfolio(z, 1))


if __name__ == '__main__':
    main()