import pandas


class PositionInfo:

    def __init__(self, position):
        
        self.m_position = position
        self.m_trades = {}
        self.m_records = {}

    def add_trade(self, trade):
        """
        Log a trade for TCA.
        
        """
        self.m_position.add_trade(trade)
        self.m_trades.update({trade.time: trade})

    def log_position_status(self, time):
        """
        Log position status for PNL-analysis. 
        
        """
        record = {}
        record.update({'NET_POSITION': self.m_position.m_net_position})
        record.update({'NET_INVESTMENT': self.m_position.m_net_investment})
        record.update({'REALIZED_PNL': self.m_position.m_realized_pnl})
        record.update({'UNREALIZED_PNL': self.m_position.m_unrealized_pnl})
        record.update({'TOTAL_PNL': self.m_position.m_total_pnl})
        record.update({'TOTAL_COM': self.m_position.m_commissions})
        
        self.m_records.update({time: record})
 
    def generate_pnl_report(self, formate='frame'):
        """
        Returns a PNL report either as a dictionary or as a Pandas DataFrame.
        """
        if formate == 'frame':
            report = pandas.DataFrame.from_dict(self.m_records, orient='index')
        elif formate == 'dict':
            report = self.m_records
        else:
            raise ValueError('Formate has to be either "frame" or "dict"!')
        return report
    
