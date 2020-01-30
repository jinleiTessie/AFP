from datetime import date, timedelta
import datetime
import pandas as pd

class DataStore:
    def __init__(self, stdt, endt):
        self.startdate        = pd.to_datetime(stdt)
        self.enddate        = pd.to_datetime(endt)
        self.ticker_list = self.get_tickers()
    
    def loadData(self, tag):
        return df
