
import pandas as pd
import yfinance as api


class DataHandler:

    @staticmethod
    def download_historical_closings(symbols):

        if not isinstance(symbols, list):
            symbols = [symbols]

        symbol_data = {}
        for symbol in symbols:

            # Download data
            data = api.Ticker(symbol).history(period="max")

            symbol_data.update({symbol: data.to_dict() })

        return symbol_data

