
import pandas as pd
import yfinance as api


class DataHandler:

    @staticmethod
    def download_historical_closings(symbols):

        if not isinstance(symbols, list):
            symbols = [symbols]

        data = []
        for symbol in symbols:
            historical_sample = api.Ticker(symbol).history(period="max")
            field_data = historical_sample['Close'].to_frame()
            field_data.index = pd.DatetimeIndex(field_data.index)
            field_data.columns = [symbol]
            data.append(field_data)
        data = pd.concat(data, axis=1)

        return data

