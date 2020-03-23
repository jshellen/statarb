import matplotlib.pyplot as plt
import numpy as np

import statsmodels.api as sm

from .coint_johansen import Johansen

def ou_bias_correction(n):

    if not isinstance(n, (int, np.int32, np.int64)):
        raise ValueError(f'n has to be integer. {type(n)}')

    return 29.70381061*(1/(n*0.0189243637761786))


def estimate_ou_parameters_using_lsq(x, dt, bias_corretion=False):
    """
    Estimates parameters of Ornstein-Uhlenbeck style stochastic process:

    dX_t = kappa*(theta - X_t)*dt + sigma*dW_t

    NOTE: The standard least squares estimation is very upward biased. Therefore, we need to adjust it down.

    :param x:
    :param dt:
    :return: kappa, theta, sigma
    """

    if not isinstance(x, np.ndarray):
        raise TypeError(f'x needs to be type of numpy.ndarray, it was {type(x)}')
    
    if not isinstance(dt, float):
        raise TypeError('dt needs to be type of float!')
    
    if dt <= 0:
        raise ValueError('Delta time has to be positive and non-zero!')
        
    S_m = np.roll(x, 1)[1:]
    S_p = x[1:]
    X = sm.add_constant(S_m)
    Y = S_p
    ols_est = sm.OLS(Y, X).fit()

    a = ols_est._results.params[1]
    b = ols_est._results.params[0]

    if a < 0:
        kappa = 0  # we cannot take log from negative number
        theta = 0
        sigma = 0
    else:

        kappa = -np.log(a)/dt
        theta = b/(1 - a)
        sigma = np.std(ols_est.resid)*(np.sqrt(-2*np.log(a)/(dt*(1-a**2))))

        if bias_corretion:
            kappa = kappa - ou_bias_correction(len(x))

    return kappa, theta, sigma


def estimate_ln_coint_params(x, y, dt):
    """
    Estimates log-price sensitivity "delta" w.r.t cointegrating factor:

    dln(S_t) = (mu - 1/2*sigma^2 + delta * Z_t)*dt + sigma*dW_t

    Dynamic Optimal Portfolios for Multiple Co-Integrated Assets

    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f'x needs to be type of numpy.ndarray, it was {type(x)}')

    if not isinstance(y, np.ndarray):
        raise TypeError(f'y needs to be type of numpy.ndarray, it was {type(y)}')

    if not isinstance(dt, float):
        raise TypeError('dt needs to be type of float!')

    if len(x.shape) != 2:
        x = x.reshape(-1, 1)

    if len(y.shape) != 2:
        y = y.reshape(-1, 1)

    # Estimate cointegration factor beta_i
    estimator = Johansen(np.concatenate([x, y], axis=1), model=2, significance_level=0)
    e_, r = estimator.johansen()
    e = e_[:, 0] / e_[0, 0]
    beta = e[1]

    # Compute Z_t - a_i = ln(s_0) + beta_i * ln(s_i)
    z_minus_a = x + beta * y

    # Estimate a_i
    a = -np.mean(z_minus_a)

    # Recompute Z_t
    z = a + x + beta * y

    # Estimate Ornstein-Uhlenbeck parameters
    kappa, theta, sigma = estimate_ou_parameters_using_lsq(z, dt, True)

    delta = None
    if (kappa != None) and (theta != None) and (sigma != None):

        # Compute delta from mean-reversion speed and beta_i
        delta = kappa/(-beta)

    return delta, beta, kappa, a

