# -*- coding: utf-8 -*-

from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandas import concat
from scipy.stats.stats import pearsonr
from numpy import log, sqrt, ndarray, log

from src.estimation.ou_parameter_estimation import estimate_ou_parameters_using_lsq


class OrnsteinUhlenbeckProcessParameters:

    def __init__(self, kappa, theta, eta, sigma_b, rho, mu_b, x_0, b_0):

        def check_numeric(arg, arg_name):
            if not isinstance(arg, (int, float)):
                raise TypeError('{} has to be type of <int> or <float>.'.format(arg_name))

        check_numeric(kappa, 'kappa')
        check_numeric(theta, 'theta')
        check_numeric(eta, 'eta')
        check_numeric(sigma_b, 'sigma_b')
        check_numeric(rho, 'rho')
        check_numeric(mu_b, 'mu_b')
        check_numeric(x_0, 'x_0')
        check_numeric(b_0, 'b_0')

        # Initialize values to none
        self.m_eta = eta
        self.m_sigma_b = sigma_b
        self.m_rho = rho
        self.m_theta = theta
        self.m_kappa = kappa
        self.m_mu_b = mu_b
        self.m_x_0 = x_0
        self.m_b_0 = b_0

    @property
    def kappa(self):
        """
        Mean-reversion speed of the process.
        """
        return self.m_kappa
    
    @property
    def theta(self):
        """
        Long-term average level of the spread
        """
        return self.m_theta
        
    @property
    def rho(self):
        """
        Correlation coefficient between the spread and the price series
        """
        return self.m_rho

    @property
    def eta(self):
        """
        Volatility of the spread.
        """
        return self.m_eta
    
    @property
    def sigma_b(self):
        """
        Volatility of asset "B" (GBM)
        """
        return self.m_sigma_b

    @property
    def mu_b(self):
        """
        Drift of asset "B" (GBM)
        """
        return self.m_x_0           

    @property
    def x_0(self):
        """
        Initial OU Process value
        """
        return self.m_x_0      

    @property
    def b_0(self):
        """
        Initial value of asset "B" (GBM)
        """
        return self.m_b_0        

    @classmethod
    def ols_parameter_estimation(cls, a_data, b_data, dt):

        if not isinstance(a_data, (DataFrame, Series, ndarray)):
            raise TypeError('a_data has invalid data type')

        if not isinstance(b_data, (DataFrame, Series, ndarray)):
            raise TypeError('b_data has invalid data type')
        
        if not isinstance(dt, (int, float)):
            raise TypeError('dt has to be type of float.')
        
        if dt <= 0:
            raise ValueError('Delta time has to be positive and non-zero!')
        
        if isinstance(a_data, Series):
            a_data = a_data.to_frame(name=0)
            
        if isinstance(b_data, Series):
            b_data = b_data.to_frame(name=0)

        if isinstance(a_data, ndarray):
            a_data = DataFrame(data=a_data)
            
        if isinstance(b_data, ndarray):
            b_data = DataFrame(data=b_data)
                
        # Compute logarithmic spread level
        x = log(a_data) - log(b_data)
        
        # Estimate OU parameters
        pars = estimate_ou_parameters_using_lsq(x.values, dt)
        kappa_est = pars[0]
        theta_est = pars[1]
        eta_est = pars[2]            

        # Compute correlation between asset a and spread level
        a = b_data.pct_change(1)
        b = x.diff(1)
        c = concat([a, b], axis=1).dropna()
        rho_est = pearsonr(c.iloc[:, 0], c.iloc[:, 1])[0]

        # Compute scaled volatility for asset b
        sigma_est = a.std().values[0]*sqrt(1.0/dt)
        
        x_0 = a_data.iloc[0, 0]
        b_0 = b_data.iloc[0, 0]

        N = b_data.shape[0]
        sigma_est = a.std().values[0]*sqrt(1.0/dt)
        mu_est = log(b_data.iloc[-1, 0] / b_data.iloc[0, 0]) / (dt * N) +0.5 * sigma_est**2
    
        return cls(kappa_est, theta_est, eta_est, sigma_est, rho_est, mu_est, x_0, b_0)
    
    def __str__(self):
        
        if isinstance(self.kappa, float):
            kappa = round(self.kappa, 2)
        else:
            kappa = ''
        
        if isinstance(self.theta, float):
            theta = round(self.theta, 2)
        else:
            theta = ''
        
        if isinstance(self.rho, float):
            rho = round(self.rho, 2)
        else:
            rho = ''
        
        if isinstance(self.eta, float):
            eta = round(self.eta, 2)
        else:
            eta = ''
        
        return f"Ornstein-Uhlenbeck Parameters: Kappa = {kappa}, Theta = {theta}, Rho = {rho}, Eta = {eta}"
