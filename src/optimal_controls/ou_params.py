# -*- coding: utf-8 -*-

from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandas import concat
from scipy.stats.stats import pearsonr
from numpy import log, sqrt, ndarray

from src.estimation.parameter_estimation import estimate_ou_parameters


class OrnsteinUhlenbeckProcessParameters:

    def __init__(self, kappa, theta, eta, sigma_b, rho):

        if not isinstance(kappa, (int, float)):
            raise TypeError('kappa has to be type of <int> or <float>.')

        if not isinstance(theta, (int, float)):
            raise TypeError('theta has to be type of <int> or <float>.')

        if not isinstance(eta, (int, float)):
            raise TypeError('eta has to be type of <int> or <float>.')

        if not isinstance(sigma_b, (int, float)):
            raise TypeError('sigma_b has to be type of <int> or <float>.')

        if not isinstance(rho, (int, float)):
            raise TypeError('rho has to be type of <int> or <float>.')

        # Initialize values to none
        self.m_eta = eta
        self.m_sigma_b = sigma_b
        self.m_rho = rho
        self.m_theta = theta
        self.m_kappa = kappa

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
        Volatility of asset "B"
        """
        return self.m_sigma_b

    def ols_parameter_estimation(self, a_data, b_data, dt):

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
        
        try:
                
            # Compute logarithmic spread level
            x = log(a_data) - log(b_data)
            
            # Estimate OU parameters
            pars = estimate_ou_parameters(x.values, dt)
            
            self.m_kappa = pars[0]
            self.m_theta = pars[1]
            self.m_eta = pars[2]

            # Compute correlation between asset a and spread level
            a = b_data.pct_change(1)
            b = x.diff(1)
            c = concat([a, b], axis=1).dropna()
            self.m_rho = pearsonr(c.iloc[:, 0], c.iloc[:, 1])[0]

            # Compute scaled volatility for asset b
            self.m_sigma_b = a.std().values[0]*sqrt(1.0/dt)
        
            return 1
        
        except Exception as e:
            
            print(e)
            
            return 0
    
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
