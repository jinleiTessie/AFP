import os
import numpy as np
import pickle
import pdb
import sys
from datetime import datetime
import pandas as pd
import bs4 as bs
import pickle
import requests
from datetime import timedelta
from itertools import combinations

class DataStore:
    
    def __init__(self, startDate = '19900101', endDate = '20200101', adjusted = True):
        self.startdate = pd.to_datetime(startDate, format='%Y%m%d', errors='ignore')
        self.enddate = pd.to_datetime(endDate, format='%Y%m%d', errors='ignore')
        self.adjusted = adjusted
        
        self.ticker_list = self.save_sp500_tickers() 
        self.data = self.load_data()
        self.pair_list = self.get_pairs()
    
    def save_sp500_tickers(self):
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            tickers.append(ticker)
        symbols = [s.split('\n')[0] for s in tickers]
        symbols = [s.replace('.','/') for s in symbols]
#         symbols_df = pd.DataFrame(map(lambda x: x.replace('.','/') + ' US EQUITY',symbols),columns = ['ticker'])
#         symbols_df.to_excel('ticker.xlsx','Sheet1')
        return symbols
    
    def load_data(self):
        if self.adjusted:
            columns = pd.MultiIndex.from_product([self.ticker_list, ['Volume', 'Last', 'Low', 'High', 'Close']],
                                     names=['Ticker', 'type'])
            data = pd.read_csv('ticker_adjusted.csv',header = 5, index_col ='Dates').iloc[:,2:]
            data.columns = columns
        else:
            columns = pd.MultiIndex.from_product([self.ticker_list, ['Volume', 'Last', 'Low', 'High', 'Close']],
                                     names=['Ticker', 'type'])
            data = pd.read_csv('ticker_non_adjusted.csv',header = 5,index_col = 'Dates').iloc[:,3:]
            data.columns = columns
        return data
    
    def get_pairs(self):
        Pairs = combinations(self.ticker_list, 2) 
        return list(Pairs)
        
    def get_data(self, ticker = ['AAPL']):
        df = pd.DataFrame()
        for symbol in ticker:
            df = pd.concat([df, self.data[symbol]],axis = 1)
        columns = pd.MultiIndex.from_product([ticker, ['Volume', 'Last', 'Low', 'High', 'Close']],
                                     names=['Ticker', 'type'])   
        df.columns = columns
        return df
        
    def data_split(self, train = 0.5, validate = 0.3, test = 0.2):
        
        row_num = self.data.shape[0]
        cut_index_1 = int(tow_num * train)
        cut_index_2 = int(row_num * (train + validate))
        train_set = self.data.iloc[0:cut_index_1, :]
        validate_set = self.data.iloc[cut_index_1:cut_index_2, :]
        test_set = self.data.iloc[cut_index_2:, :]
        return train_set, validate_set, test_set
