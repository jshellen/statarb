from copy import deepcopy
import pandas as pd
from .utilities import infer_trade_action


class Position2:
    
    def __init__(self, contract):     
        
        self.m_contract = contract
        self.m_multiplier = contract.multiplier
        
        self.m_net_position = 0
        self.m_avg_open_price = 0
        self.m_net_investment = 0
        
        self.m_realized_pnl = 0
        self.m_unrealized_pnl = 0
        self.m_total_pnl = 0
        self.m_commissions = 0
          
        self.m_bid = None
        self.m_ask = None

        self.m_records = {}

    @property
    def contract(self):
        return self.m_contract
    
    @property
    def market_value(self):
        
        # Compute mid-price
        mid_price = 0.5*(self.m_bid + self.m_ask)
        
        return self.m_net_position * self.m_multiplier * mid_price
    
    @property
    def quantity(self):
        
        return abs(self.m_net_position)

    @property
    def net_position(self):
        return deepcopy(self.m_net_position)

    # buy_or_sell: 1 is buy, 2 is sell
    def add_trade(self, trade):
        
        action = infer_trade_action(trade.action)
        traded_price = trade.price
        commission = trade.commission
        traded_quantity = trade.quantity

        # buy: positive position, sell: negative position
        signed_quantity = traded_quantity if action == 1 else (-1) * traded_quantity
        
        # Check if the trade will revert the direction of the position
        is_still_open = (self.m_net_position * signed_quantity) >= 0

        # Update Realizd and Total PnL
        if not is_still_open:
            
            # Remember to keep the sign as the net position
            self.m_realized_pnl += self.m_multiplier *( traded_price - self.m_avg_open_price ) \
                                    * min(abs(signed_quantity),
                                          abs(self.m_net_position)        ) \
                                    * ( abs(self.m_net_position) /
                                            self.m_net_position ) 
                
        # total pnl
        self.m_total_pnl = self.m_realized_pnl + self.m_unrealized_pnl

        # Commissions
        self.m_commissions += commission

        # Update Average Openin Price
        if is_still_open:
            #print("Still open")
            # Update average open price
            self.m_avg_open_price = ( ( self.m_avg_open_price * self.m_net_position ) 
                                    + ( traded_price * signed_quantity ) ) \
                                    / ( self.m_net_position + signed_quantity )
        
        else:
            #print("Not open")
            # Check if it is close-and-open
            if traded_quantity > abs(self.m_net_position):
                self.m_avg_open_price = traded_price

        # Update net position
        self.m_net_position += signed_quantity

        # net investment
#        self.m_net_investment = max( self.m_net_investment,
#                                    abs( self.m_multiplier * self.m_net_position * self.m_avg_open_price  ) )
        
        self.m_net_investment = self.m_multiplier * self.m_net_position * self.m_avg_open_price     
        
        # Update Unrealized and Total PnL
        #self.update_market_value(bid,ask)

    def update_market_value(self, bid, ask, time=None):
        
        if (bid > 0) and (ask > 0):
            mid = 0.5*(bid + ask)
            self.m_unrealized_pnl = self.m_multiplier * (mid - self.m_avg_open_price) * self.m_net_position
            self.m_total_pnl = self.m_realized_pnl + self.m_unrealized_pnl - self.m_commissions
        else:
            raise ValueError('Prices have to be positive')

        record = {}
        record.update({'NET_POSITION': self.m_net_position})
        record.update({'NET_INVESTMENT': self.m_net_investment})
        record.update({'REALIZED_PNL': self.m_realized_pnl})
        record.update({'UNREALIZED_PNL': self.m_unrealized_pnl})
        record.update({'TOTAL_PNL': self.m_total_pnl})
        record.update({'TOTAL_COM': self.m_commissions})

        if time is not None:
            self.m_records.update({time: record})
        else:
            i = len(self.m_records)
            self.m_records.update({i+1: record})

    def generate_report_frame(self):

        return pd.DataFrame.from_dict(self.m_records, orient='index')