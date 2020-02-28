import numpy as np
import pandas as pd
import util
import datetime as dt
import csv
import matplotlib.pyplot as plt

"""
Code based on previous code written by Yichen Shen and Yiding Zhao 
( <https://github.com/shenyichen105/Deep-Reinforcement-Learning-in-Stock-Trading> )

"""


# Simulates the market. It accept the agent's choice of action, executes the respective operations (buying, selling),
# atualizes the portfolio composition and value and returns a reward for the action and the new state, dependent on
# next day's stock prices. 

class Simulator(object):

    def __init__(self, symbols,
                 start_date=dt.datetime(2008, 1, 1),
                 end_date=dt.datetime(2009, 1, 1),
                 test_date=dt.datetime(2008, 8, 1), ):

        self.dates_range = pd.date_range(start_date, end_date)
        self.test_date = test_date
        self.check = 0
        self.tickers = symbols
        #self.cum_reward = 0


        self.date_start = 1

        # for visualization
        self.data_out = []
        self.costs = []
        self.cum_return = []

        # preprocessing time series
        # stock symbol data
        stock_symbols = symbols[:]

        # price data
        prices_all = util.get_prices(symbols, start_date, end_date)

        self.stock_A = stock_symbols[0]
        self.stock_B = stock_symbols[1]

        # first trading day start from day 100
        self.dateIdx = 100
        self.date = prices_all.index[0]
        self.start_date = start_date
        self.end_date = end_date

        self.prices = prices_all[stock_symbols]
#        print (self.prices)
        #self.prices_ibovespa = prices_all['ibovespa']

        self.portfolio = {'longA':0,'longB':0}

        print('========1.init environment===================')
        #print('Environment is', self.dates_range, self.portfolio, self.port_val, self.port_val_market)

    def init_state(self, lookback=50):
        """
        return initial states of the market
        """
        print('===========2.get state=====================')

        lookback_states = []
        #lookback_states.append(self.get_state(self.date))
        day1 = self.dateIdx-100
        for _ in range(lookback - 1):
            day1 += 1
            self.date = self.prices.index[day1]
            lookback_states.append(self.get_state(self.date))


        lookback_mean = np.array(lookback_states).mean()
        lookback_std = np.array(lookback_states).std()

        spread_zscores =[]
        day2 = self.dateIdx-50
        for _ in range(lookback - 1):
            day2 +=1
            self.date = self.prices.index[day2]
            spread_zscores.append(list((self.get_state(self.date)-lookback_mean)/lookback_std))


#        print('returned states are', spread_zscores)
        return spread_zscores

    def step(self, action, prev_action):

        """
        Take an action and and move the date forward.

        There are 3 actions: buy, sell and hold

        returns reward, next day's market status (the "state"), the day and the restrictions ("boundary")
        """
#        print('===============6.in each step study action in environment=======================')

        position_A = 1
        position_B = 1
        price_A = self.prices.ix[self.date, self.stock_A]
        price_B = self.prices.ix[self.date, self.stock_B]

        #same_action = (action == previous_action)

        if (action == 'buy'):
            # long A short B
            A_cost = position_A*price_A
            B_earning = position_B * price_B
            net = B_earning - A_cost
            self.costs.append(net)
            daily_return = net/self.costs[0]
            self.cum_return.append(daily_return)



        if (action == 'sell'):
            # long B short A
            A_earning = position_A * price_A
            B_cost = position_B * price_B
            net= A_earning - B_cost
            self.costs.append(net)
            daily_return = net / self.costs[0]
            self.cum_return.append(daily_return)


        if (action =='hold'):
            #no action
            net =price_A-price_B
            self.costs.append(net)
            if (len(self.cum_return)==0):
                daily_return = 1
                self.cum_return.append(daily_return)

            else:
                if (prev_action=='buy'):
                    daily_return = self.cum_return[-1]
                    daily_return -=(price_A - price_B)/self.costs[0]
                if (prev_action=='sell'):
                    daily_return = -self.cum_return[-1]
                    daily_return += (price_A - price_B)/self.costs[0]
                if (prev_action=='hold'):
                    daily_return = self.cum_return[-1]
                self.cum_return.append(daily_return)


        # The reward function
        #reward = np.tanh(100 * (self.port_val - old_port_val) / self.init_cash)
        reward = np.tanh(100*self.cum_return[-1])

        #self.cum_reward += reward


        self.data_out.append(self.date.isoformat()[0:10] + ',' +str(action)+','+str(price_A)+','+str(price_B)+','+str(price_A-price_B)+','+str(self.cum_return[-1]))

        self.dateIdx += 1
        if self.dateIdx < len(self.prices.index):
            self.date = self.prices.index[self.dateIdx]

        state = self.get_state(self.date)

        # The following function applies limitations to the closing of certain positions and to the opening of too many ones
        #boundary = self.get_boundary()

        # resets the portfolio values when the simulation enters the testing period
        # (in the present version, the 11th year)
        #        if self.date >= self.dates_range[-1] - dt.timedelta(days=365) and self.check == 0:
        if self.date >= self.test_date and self.check == 0:
            # self.portfolio = {'cash': self.init_cash, 'a_vol': [], 'a_price': [], 'b_vol': [], 'b_price': [],
            #                   'longA': 0, 'longB': 0}
            self.check = 1
            #self.cum_cash = self.init_cash
            self.date_start = self.dateIdx
            self.costs=[]
            self.cum_return=[]
            #self.cum_operations = 0

        #return (reward, state, self.dateIdx, boundary)
        return (reward, state, self.dateIdx)


    def get_state(self, date):
        """
        returns state of next day's market.
        """
        if date not in self.dates_range:
            if verbose: print('Date was out of bounds.')
            if verbose: print(date)
            exit

        #print ('===========================prices index=======================',self.prices.index[-1])

        if (date == self.prices.index[-1]):
            file_name = "data_for_vis_%s.csv" % dt.datetime.now().strftime("%H-%M-%S")
            print ('======================file names====================',file_name)

            file = open(file_name, 'w');
            file.write(
                'date,action,A_price,B_price,spread,cum_reward')
            file.write('\n')
            for line in self.data_out:
                file.write(line);
                file.write('\n')
            file.close()
            
            plot_df = pd.read_csv(file_name)
            plot_df['date'] = pd.to_datetime(plot_df['date'])
            plot_df = plot_df.set_index('date')
            
#            train_df = plot_df
#            [plot_df.index<self.test_date]
#            self.plot_res(train_df.copy(), 'training')
            train_df = plot_df[plot_df.index>=self.test_date]
            
            # training    
            fig, ((ax1,ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
            train_df[['cum_reward']].plot(title='cumulative return', ax=ax1)
            train_df[['cum_reward']].pct_change().plot(title='return', ax=ax2)
    #        df[['port_val']].plot(title='port_val', ax=ax2)
            train_df[['A_price','B_price']].plot(title='prices', ax=ax3)
            
            train_df['max'] = train_df['cum_reward'].cummax()
            train_df['drawdown'] = train_df['cum_reward'] - train_df['max'] 
            train_df[['drawdown']].plot(title='drawdown', ax=ax4)
    
            act_A = train_df[['action']].replace('hold',0).replace('buy',1).replace('sell',-1).replace('buyB',0).replace('sellB',0)
#            act_B = train_df[['action']].replace('hold',0).replace('buyA',0).replace('sellA',0).replace('buyB',1).replace('sellB',-1)
#            acts=pd.concat([act_A, act_B], axis=1)
#            acts.columns=['A','B']
            act_A.plot(title='actions', ax=ax5)
            
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            names='testing'+ str(self.tickers)
            
            fig.suptitle(names)
            fig.savefig(names+'.png')



        # Calculates the mean between stocks A, B and C's normalized prices on the day corresponding to the variable "date".
        # A_zscore = self.prices.ix[date, self.stock_A] / self.prices.ix[0, self.stock_A]
        # B_zscore = self.prices.ix[date, self.stock_B] / self.prices.ix[0, self.stock_B]
        # prices_mean = np.mean([A_zscore, B_zscore])
        #spread = (self.prices.ix[date, self.stock_A] - self.prices.ix[date, self.stock_B])/(self.prices.ix[0, self.stock_A] - self.prices.ix[0, self.stock_B]) - 1
        spread = (self.prices.ix[date, self.stock_A] - self.prices.ix[date, self.stock_B])
        # returns state of next day's market, i.e. the difference between each stock's normalized price and the mean of the three
        # normalized prices.
        #return [A_zscore - prices_mean, B_zscore - prices_mean]
        #return [A_zscore-B_zscore]
        return [spread]




    def has_more(self):
        if ((self.dateIdx < len(self.prices.index)) == False):
            print('\n\n\n*****')
            # # Average daily cash account
            # print(self.cum_cash / (self.dateIdx - self.date_start + 1))
            # print('*****\n\n\n')
            # # Final portfolio value in the testing year
            # print(self.port_val)
            # print(self.port_val_market)
            # print('*****\n\n\n')
            # # Number of positions opened in the testing year
            # print(self.cum_operations)
            # print('*****\n\n\n')
        return self.dateIdx < len(self.prices.index)