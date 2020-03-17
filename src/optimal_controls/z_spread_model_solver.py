import numpy as np

from .z_spread_model_parameters import (
    ZSpreadModelParameters
)

from .z_spread_model import (
    ZSpreadModel
)


class ZSpreadModelSolver:

    @staticmethod
    def solve(params, T, n_step):
        """

        :param params:
        :param T:
        :param n_step:
        :return:
        """
        if not isinstance(params, ZSpreadModelParameters):
            raise TypeError('Parameters have to be type of <ZSpreadModelParameters>.')

        if not isinstance(T, (int, float)):
            raise TypeError('T has to be type of <int> or <float>.')

        if not isinstance(n_step, int):
            raise TypeError('n_step has to be type of <int>.')

        if T <= 0:
            raise ValueError('T has to be non-negative.')

        if n_step <= 0:
            raise ValueError('n_step has to be non-negative.')

        c_t = ZSpreadModelSolver._solve_c(params, T, n_step)

        b_t = ZSpreadModelSolver._solve_b(params, c_t, T, n_step)

        return ZSpreadModel(params, b_t, c_t, T, n_step)

    @staticmethod
    def _solve_b(params, c_t, T, n_step):
        """

        :param params:
        :param c:
        :param T:
        :param n_step:
        :return:
        """

        if not isinstance(params, ZSpreadModelParameters):
            raise TypeError('Parameters have to be type of <ZSpreadModelParameters>.')

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

        if not isinstance(params, ZSpreadModelParameters):
            raise TypeError('Parameters have to be type of <ZSpreadModelParameters>.')

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