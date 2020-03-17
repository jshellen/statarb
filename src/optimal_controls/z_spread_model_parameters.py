import numpy as np

from .estimation.parameter_estimation import (
    estimate_ln_coint_params
)


class ZSpreadModelParameters:

    def __init__(self, gamma=None, rho=None, rho_0=None, sigma_0=None, sigma=None, mu_0=None, mu=None, beta=None, delta=None, b=None, a=None):

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
                        self.m_sigma_0 ** 2
                        + self.m_sigma_0 * self.m_sigma[i]
                        * self.m_beta[i] * self.m_rho_0[i]
                        + self.m_sigma_0 * self.m_sigma[j]
                        * self.m_beta[j] * self.m_rho_0[j]
                        + self.m_sigma[i] * self.m_beta[i]
                        * self.m_sigma[j] * self.m_beta[j] * self.m_rho[i, j])

        self.m_theta = np.zeros_like(self.m_delta)
        for i in range(0, self.m_sigma_1.shape[0]):
            self.m_theta[i] = (self.m_beta[i] + self.m_mu_0 + self.m_beta[i] * self.m_mu[i]
                               - 0.5 * (self.m_sigma_0 ** 2 + self.m_beta[i] * self.m_sigma[i] ** 2)) / (
                                          self.m_beta[i] * self.m_delta[i])

    @staticmethod
    def estimate_from_ln_prices(ln_s_0, ln_s_i, gamma=-1):

        mu_0 = 250 * np.mean(np.diff(ln_s_0, 1, axis=0))
        sigma_0 = np.sqrt(250) * np.std(np.diff(ln_s_0, 1, axis=0))

        n_assets = ln_s_i.shape[1]

        # Estimate correlation structure
        cor = np.corrcoef(np.concatenate([np.diff(ln_s_0, 1).reshape(-1, 1),
                                          np.diff(ln_s_i, 1, axis=0)], axis=1), rowvar=False)

        # Correlations between assets and the the benchmark
        rho_0 = np.zeros((n_assets, 1))
        for i in range(0, n_assets):
            rho_0[i, 0] = cor[i + 1, 0]

        # Correlations between assets
        rho = cor[1:, 1:]

        # Estimate annual drift rates
        mu = np.zeros((n_assets, 1))
        for i in range(0, n_assets):
            mu[i] = 250 * np.mean(np.diff(ln_s_i, 1, axis=0))

        # Estimate annual volatilities
        sigma = np.zeros((n_assets, 1))
        for i in range(0, n_assets):
            sigma[i] = np.sqrt(250) * np.std(np.diff(ln_s_i[:, i], 1, axis=0))

        # Estimate beta, kappa and delta
        d = np.zeros((n_assets, 1))
        a = np.zeros((n_assets, 1))
        b = np.zeros((n_assets, 1))
        k = np.zeros((n_assets, 1))
        beta = np.zeros((n_assets, 1))
        for i in range(0, n_assets):
            d[i], beta[i], k[i], a[i] = estimate_ln_coint_params(ln_s_0, ln_s_i[:, i], 1 / 250)

        return ZSpreadModelParameters(gamma, rho, rho_0, sigma_0, sigma, mu_0, mu, beta, d, b, a)

