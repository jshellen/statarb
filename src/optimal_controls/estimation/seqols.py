import numpy as np


class SequentialLinearRegression:

    def __init__(self, alpha, constant, n_vars):

        if not isinstance(alpha, float):
            raise TypeError('alpha has to be type of <float>.')

        if not isinstance(constant, bool):
            raise TypeError('constant has to be type of <bool>.')

        if not isinstance(n_vars, int):
            raise TypeError('n_vars has to be type of <int>.')

        if alpha <= 0:
            raise ValueError(f'alpha has to be positive float, was: {float}')

        if n_vars <= 0:
            raise ValueError(f'n_vars has to be positive constant, was : {n_vars}')

        self.m_alpha = alpha
        self.m_constant = constant
        self.m_n_vars = n_vars

        if constant:
            self.m_dim = n_vars + 1
        else:
            self.m_dim = n_vars

        self.m_M_t = None
        self.m_V_t = None
        self.m_M_p = np.zeros((self.m_dim, self.m_dim))
        self.m_V_p = np.zeros((self.m_dim, 1))
        self.m_coefs = None

    def add_obs(self, x, y):
        """
        Add observation and re-evaluate.
        @param: x vector of observations
        """

        if isinstance(x, (int, float)):
            xy = np.array([[y]])

        if isinstance(y, (int, float)):
            y = np.array([[y]])

        if not isinstance(x, np.ndarray):
            raise TypeError('x has to be type of <np.ndarray>.')

        if not isinstance(y, np.ndarray):
            raise TypeError('y has to be type of <np.ndarray>.')

        if len(x.shape) != 2:
            raise ValueError('x is non-dimensional')

        if x.shape[1] != self.m_n_vars:
            raise ValueError(f'x has shape {x.shape} while n_vars: {self.m_n_vars}.')

        x_ = x.copy()
        if self.m_constant:
            x_ = np.concatenate([np.array([[1]]), x_], axis=1)

        self.m_M_t = (1-self.m_alpha)*x_.T.dot(x_) + self.m_alpha * self.m_M_p
        self.m_V_t = (1-self.m_alpha)*x_.T.dot(y).reshape(-1, 1) + self.m_alpha * self.m_V_p

        self.m_M_p = self.m_M_t
        self.m_V_p = self.m_V_t

        M_inv = np.linalg.pinv(self.m_M_t)

        self.m_coefs = np.matmul(M_inv, self.m_V_t).reshape(1, -1)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n_obs = 500

    a = 0.5
    b = 1.2
    c = 2.6

    # Set 1
    x_1 = np.random.normal(0, 1, (n_obs, 2))
    y_1 = (a + b * x_1[:, 0] + c * x_1[:, 1]).reshape(-1, 1)
    y_1 += np.random.normal(0, 0.25, (n_obs, 1))

    a = 0.5
    b = 2.2
    c = 3.6

    # Set 2 with different parameters
    x_2 = np.random.normal(0, 1, (n_obs, 2))
    y_2 = (a + b * x_2[:, 0] + c * x_2[:, 1]).reshape(-1, 1)
    y_2 += np.random.normal(0, 0.25, (n_obs, 1))

    # Concatenate data series together
    y = np.concatenate([y_1, y_2], axis=0)
    x = np.concatenate([x_1, x_2], axis=0)

    betas = np.zeros((len(x), 3))

    seq_reg = SequentialLinearRegression(0.99, True, 2)

    for i in range(0, len(x)):
        x_ = x[i, :].reshape(1, -1)
        y_ = y[i]
        seq_reg.add_obs(x_, y_)
        betas[i, :] = seq_reg.m_coefs

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(betas[10:, :])
    plt.show()