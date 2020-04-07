import numpy as np
import pandas as pd

from src.estimation.ou_parameter_estimation import (
    estimate_ln_coint_params
)


class ZSpreadModelParameters:

    def __init__(self, gamma=None, rho=None, rho_0=None, sigma_0=None, sigma=None,
                 mu_0=None, mu=None, beta=None, delta=None, b=None, a=None,
                 sigma_1=None, sigma_2=None, sigma_3=None,
                 kappa=None, theta=None, symbols=None):

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

        self.m_symbols = None
        if symbols is not None:
            self.m_symbols = symbols

        if kappa is None:
            self.m_kappa = np.zeros_like(self.m_delta)
            for i in range(0, len(self.m_kappa)):
                self.m_kappa[i] = - self.m_beta[i] * self.m_delta[i]
        else:
            self.m_kappa = kappa

        if sigma_1 is None:
            self.m_sigma_1 = np.matmul(
                np.matmul(np.diag(self.m_sigma.flatten()),
                          self.m_rho), np.diag(self.m_sigma.flatten()))
        else:
            self.m_sigma_1 = sigma_1

        if sigma_2 is None:
            self.m_sigma_2 = np.zeros_like(self.m_sigma_1)
            for i in range(0, self.m_sigma_1.shape[0]):
                for j in range(0, self.m_sigma_1.shape[1]):
                    self.m_sigma_2[i, j] = self.m_sigma_0 * self.m_sigma[i] * self.m_rho_0[i] \
                                           + self.m_beta[j] * self.m_sigma_1[i, j]
        else:
            self.m_sigma_2 = sigma_2

        if sigma_3 is None:
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
        else:
            self.m_sigma_3 = sigma_3

        if theta is None:
            self.m_theta = np.zeros_like(self.m_delta)
            for i in range(0, self.m_sigma_1.shape[0]):
                 self.m_theta[i] = -(self.m_b[i] + self.m_mu_0 + self.m_beta[i] * self.m_mu[i]
                        - 0.5 * (self.m_sigma_0 ** 2 + self.m_beta[i] * self.m_sigma[i] ** 2)) / (
                               self.m_beta[i] * self.m_delta[i])
        else:
            self.m_theta = theta

    @staticmethod
    def estimate_from_ln_prices(ln_s_0, ln_s_i, gamma=-1, kappa_min=None):
        """
        Run parameter estimation from ln-prices.

        """
        if not isinstance(ln_s_0, pd.core.frame.DataFrame):
            raise ValueError('ln_s_0 has to be <pd.core.frame.DataFrame>.')

        if not isinstance(ln_s_i, pd.core.frame.DataFrame):
            raise ValueError('ln_s_i has to be <pd.core.frame.DataFrame>.')

        n_assets = ln_s_i.shape[1]
        symbols = list(ln_s_i.columns)
        ln_s_0_ = np.copy(ln_s_0.values)
        ln_s_i_ = np.copy(ln_s_i.values)

        # If kappa min is used, run pre kappa estimation in order to select indices
        if kappa_min:
            if kappa_min < 0:
                raise ValueError('kappa_min has to be non-negative.')
            kappa = np.zeros((n_assets, 1))
            for i in range(0, n_assets):
                _, _, kappa[i], _ = estimate_ln_coint_params(ln_s_0_, ln_s_i_[:, i], 1.0 / 250.0)
            selected_columns = np.where(kappa > kappa_min)[0]
            if len(selected_columns) < 1:
                raise ValueError('No selected columns.')
            symbols = list(ln_s_i.columns[selected_columns])
            n_assets = len(selected_columns)
            ln_s_i_ = ln_s_i_[:, selected_columns]

        # Estimate beta, kappa and delta
        a = np.zeros((n_assets, 1))
        b = np.zeros((n_assets, 1))
        beta = np.zeros((n_assets, 1))
        delta = np.zeros((n_assets, 1))
        kappa = np.zeros((n_assets, 1))
        for i in range(0, n_assets):
            delta[i], beta[i], kappa[i], a[i] = estimate_ln_coint_params(ln_s_0_, ln_s_i_[:, i], 1 / 250)

        mu_0 = 250 * np.mean(np.diff(ln_s_0_, 1, axis=0))
        sigma_0 = np.sqrt(250) * np.std(np.diff(ln_s_0_, 1, axis=0))

        series_all_ = np.concatenate([ln_s_0_, ln_s_i_], axis=1)
        cor = np.corrcoef(np.diff(series_all_, axis=0), rowvar=False)

        # Correlations between assets and the the benchmark
        rho_0 = np.zeros((n_assets, 1))
        for i in range(0, n_assets):
            rho_0[i, 0] = cor[i + 1, 0]

        # Correlations between assets
        rho = cor[1:, 1:]

        # Estimate annual drift rates
        mu = np.zeros((n_assets, 1))
        for i in range(0, n_assets):
            # NOTE: estimate is scaled down by 50 %
            mu[i] = 0.5 * 250 * np.mean(np.diff(ln_s_i_[:, i], 1, axis=0))

        # Estimate annual volatilities
        sigma = np.zeros((n_assets, 1))
        for i in range(0, n_assets):
            sigma[i] = np.sqrt(250) * np.std(np.diff(ln_s_i_[:, i], 1, axis=0))

        # Estimate sigmas for riccati equations
        sigma_1 = np.matmul(
            np.matmul(np.diag(sigma.flatten()), rho),
            np.diag(sigma.flatten())
        )

        sigma_2 = np.zeros_like(sigma_1)
        for i in range(0, sigma_1.shape[0]):
            for j in range(0, sigma_1.shape[1]):
                sigma_2[i, j] = sigma_0 * sigma[i] * rho_0[i] \
                                       + beta[j] * sigma_1[i, j]

        sigma_3 = np.zeros_like(sigma_1)
        for i in range(0, sigma_1.shape[0]):
            for j in range(0, sigma_1.shape[1]):
                sigma_3[i, j] = (
                        sigma_0 ** 2
                        + sigma_0 * sigma[i]
                        * beta[i] * rho_0[i]
                        + sigma_0 * sigma[j]
                        * beta[j] * rho_0[j]
                        + sigma[i] * beta[i]
                        * sigma[j] * beta[j] * rho[i, j])

        theta = np.zeros_like(delta)
        for i in range(0, sigma_1.shape[0]):
            theta[i] = -(b[i] + mu_0 + beta[i] * mu[i]
                         - 0.5 * (sigma_0 ** 2 + beta[i] * sigma[i] ** 2)) / (
                           beta[i] * delta[i])

        return ZSpreadModelParameters(gamma, rho, rho_0, sigma_0, sigma,
                 mu_0, mu, beta, delta, b, a, sigma_1, sigma_2, sigma_3,
                 kappa, theta, symbols)

