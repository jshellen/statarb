# -*- coding: utf-8 -*-


class OUSpreadModelStrategyParameters:
    
    def __init__(self, nominal, asset_a_symbol, asset_b_symbol, trading_horizon, risk_tolerance, maximum_leverage=1.0):

        if not isinstance(nominal, (int, float)):
            raise TypeError('nominal has to be type of <int> or <float>.')

        if not isinstance(asset_a_symbol, str):
            raise TypeError('asset_a_symbol has to be type of <str>.')

        if not isinstance(asset_b_symbol, str):
            raise TypeError('asset_b_symbol has to be type of <str>.')

        if not isinstance(trading_horizon, (int, float)):
            raise TypeError('trading_horizon has to be type of <int> or <float>.')

        if not isinstance(risk_tolerance, float):
            raise TypeError('risk_tolerance has to be type of <float>.')

        if risk_tolerance == 1.0:
            raise ValueError('risk_tolerance has to be smaller than 1.0 .')

        if not isinstance(maximum_leverage, (int, float)):
            raise TypeError('maximum_leverage has to be type of <int> or <float>.')

        self.m_nominal = nominal
        self.m_asset_a_symbol = asset_a_symbol
        self.m_asset_b_symbol = asset_b_symbol
        self.m_trading_horizon = trading_horizon
        self.m_risk_tolerance = risk_tolerance
        self.m_maximum_leverage = maximum_leverage

    @property
    def nominal(self):
        return self.m_nominal

    @property
    def risk_tolerance(self):
        return self.m_risk_tolerance

    @property
    def trading_horizon(self):
        return self.m_trading_horizon

    @property
    def maximum_leverage(self):
        return self.m_maximum_leverage

    @property
    def symbol_a(self):
        return self.m_asset_a_symbol

    @property
    def symbol_b(self):
        return self.m_asset_b_symbol
