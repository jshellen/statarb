import matplotlib.pyplot as plt
from numpy import mean, roll, sqrt, log, std, ndarray, concatenate

import statsmodels.api as sm

from .coint_johansen import Johansen

def estimate_ou_parameters(x, dt):
    """
    Estimates parameters of Ornstein-Uhlenbeck style stochastic process:

    dX_t = kappa*(theta - X_t)*dt + sigma*dW_t

    :param x:
    :param dt:
    :return: kappa, theta, sigma
    """

    if not isinstance(x, ndarray):
        raise TypeError(f'x needs to be type of numpy.ndarray, it was {type(x)}')
    
    if not isinstance(dt, float):
        raise TypeError('dt needs to be type of float!')
    
    if dt <= 0:
        raise ValueError('Delta time has to be positive and non-zero!')
        
    S_m = roll(x, 1)[1:]
    S_p = x[1:]
    X = sm.add_constant(S_m)
    Y = S_p
    ols_est = sm.OLS(Y, X).fit()
    a = ols_est._results.params[1]
    b = ols_est._results.params[0]
    kappa = -log(a)/dt
    theta = b/(1 - a)
    sigma = std(ols_est.resid)*(sqrt(-2*log(a)/(dt*(1-a**2))))       
    
    return kappa, theta, sigma


def estimate_ln_coint_params(x, y, dt):
    """
    Estimates log-price sensitivity "delta" w.r.t cointegrating factor:

    dln(S_t) = (mu - 1/2*sigma^2 + delta * Z_t)*dt + sigma*dW_t

    Dynamic Optimal Portfolios for Multiple Co-Integrated Assets

    """
    if not isinstance(x, ndarray):
        raise TypeError(f'x needs to be type of numpy.ndarray, it was {type(x)}')

    if not isinstance(y, ndarray):
        raise TypeError(f'y needs to be type of numpy.ndarray, it was {type(y)}')

    if not isinstance(dt, float):
        raise TypeError('dt needs to be type of float!')

    # Estimate cointegration factor beta_i
    estimator = Johansen(concatenate([x.reshape(-1, 1),
                                      y.reshape(-1, 1)], axis=1), model=2, significance_level=0)
    e_, r = estimator.johansen()
    e = e_[:, 0] / e_[0, 0]
    beta = e[1]

    z_minus_a = x + beta * y

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(z_minus_a)
    plt.show()

    a = -mean(z_minus_a)

    z = a + x + beta * y

    kappa, theta, sigma = estimate_ou_parameters(z, dt)

    print('a:', a, 'kappa:', kappa, 'theta:', theta)

    delta = kappa/(-beta)

    return delta, beta, kappa, a
