import pandas as pd
import datetime as dt

def get_prices(symbols, start_date, end_date):
    
#    names=['date','ibovespa']
#    dtypes = {'date':'str','ibovespa':'float'}
#    parse_dates = ['date']
#
#    result = pd.read_csv('./data/BVSP.csv', sep=',',header=0, names=names, usecols=[0,5], dtype=dtypes, parse_dates=parse_dates, index_col=0)
#    
#    for element in symbols:
#        names=['date', element]
#        dtypes = {'date':'str', element:'float'}
#        parse_dates = ['date']
#
#        df = pd.read_csv('./data/'+element+'.csv', sep=',',header=0, names=names, usecols=[0,5], dtype=dtypes, parse_dates=parse_dates, index_col=0)
#        result = pd.concat([result, df], axis=1, join='inner')


    df = pd.read_csv("stocks_close.csv", index_col="Dates")
    df = df.rename(columns={col: col.replace(" US EQUITY", "") for col in df.columns})
    df = df[symbols]
    df.index = pd.to_datetime(df.index)
    
    df.sort_index(inplace=True)
    df = df.loc[start_date:end_date]
    df.dropna(inplace=True)
    
    print (df.head())
    return df