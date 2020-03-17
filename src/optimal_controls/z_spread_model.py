import numpy as np
import matplotlib.pyplot as plt


class ZSpreadModel:

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



# def main():
#
#     mu_0 = 0
#     sigma_0 = 0.22
#
#     rho = np.array([
#         [1.0, 0.6, 0.6],
#         [0.6, 1.0, 0.6],
#         [0.6, 0.6, 1.0]
#     ])
#
#     rho_0 = np.array([0.5, 0.5, 0.5])
#     sigma = np.array([0.25, 0.25, 0.25])
#     beta = np.array([-15, -10, -5])
#     mu = np.array([0, 0, 0])
#     delta = np.array([0.1, 0.1, 0.1])
#
#     kappa = np.zeros_like(delta)
#     for i in range(0, len(kappa)):
#         kappa[i] = - beta[i] * delta[i]
#
#     b = np.array([0, 0, 0])
#     a = np.array([0, 0, 0])
#
#
#     params = ZSpreadModelParameters(-1, rho, rho_0, sigma_0, sigma, mu_0, mu, beta, delta, b, a)
#
#     model = ZSpreadModelSolver.solve(params, 50, 2000)
#
#
#     model.plot_b_t()
#
#     model.plot_c_t()
#
#     print(model.get_c_t(1.0))
#
#     z = np.array([10, 0, 0])
#
#     print(model.optimal_portfolio(z, 1))
#
#
# if __name__ == '__main__':
#     main()