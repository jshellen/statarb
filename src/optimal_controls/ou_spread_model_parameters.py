# -*- coding: utf-8 -*-


class OU_Spread_Model_Parameters:
    
    def __init__(self, nominal, asset_a_symbol, asset_b_symbol, horizon_date, risk_tolerance, maximum_leverage = 1.0):
        
        self.m_nominal          = nominal
        self.m_asset_a_symbol   = asset_a_symbol
        self.m_asset_b_symbol   = asset_b_symbol
        self.m_horizon_date     = horizon_date
        self.m_risk_tolerance   = risk_tolerance
        self.m_maximum_leverage = maximum_leverage

    @property
    def nominal(self):
        return self.m_nominal

    @property
    def risk_tolerance(self):
        return self.m_risk_tolerance

    @property
    def horizon_date(self):
        return self.m_horizon_date

    @property
    def maximum_leverage(self):
        return self.m_maximum_leverage