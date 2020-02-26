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
        self.cum_reward = 0

        # initialize portfolio's cash
        self.init_cash = 100000

        # value of each position (long or short)
        self.buy_volume = 8000

        self.cum_operations = 0
        self.cum_cash = self.init_cash
        self.date_start = 1

        # for visualization
        self.data_out = []

        # preprocessing time series
        # stock symbol data
        stock_symbols = symbols[:]

        # price data
        prices_all = util.get_prices(symbols, start_date, end_date)

        self.stock_A = stock_symbols[0]
        self.stock_B = stock_symbols[1]

        # first trading day
        self.dateIdx = 0
        self.date = prices_all.index[0]
        self.start_date = start_date
        self.end_date = end_date

        self.prices = prices_all[stock_symbols]
        self.prices_ibovespa = prices_all['ibovespa']

        # keep track of portfolio value as a series.
        #########################
        # the variables 'longA', 'longB' and 'longC' keep track of the number
        # of opened positions on each stock. When a stock is bought, the variables are increased 1.
        # When a stock is sold, the variables are decreased 1. When the variables' values are
        # zero, no positions are opened.
        #########################
        # variable 'a_vol' keeps track of the number of A shares traded
        # (either bought or sold) in the form of a list.
        # Whenever a position is opened, it is feeded.
        # Whenever a position is closed, the respective register is deleted. The
        # positions are closed beginning with the last ones.
        #########################
        # variable 'a_price' is a list that keeps track of the price of the
        # stock A when a trade occurs. Likewise, Whenever a position is opened, it is feeded.
        # Whenever a position is closed, the respective register is deleted.
        #########################
        # The same observations apply to 'b_vol', 'c_vol', 'b_price' and 'c_price'.
        # All those variables are needed for the computation of returns when the positions are closed.
        self.portfolio = {'cash': self.init_cash, 'a_vol': [], 'a_price': [], 'b_vol': [], 'b_price': [], 'longA': 0,
                          'longB': 0}
        self.port_val = self.port_value()
        self.port_val_market = self.port_value_for_output()
        print('========1.init environment===================')
        print('Environment is', self.dates_range, self.portfolio, self.port_val, self.port_val_market)

    def init_state(self, lookback=50):
        """
        return initial states of the market
        """
        print('===========2.get state=====================')
        states = []
        states.append(self.get_state(self.date))
        for _ in range(lookback - 1):
            self.dateIdx += 1
            self.date = self.prices.index[self.dateIdx]
            states.append(self.get_state(self.date))
        print('returned states are', states)
        return states

    def step(self, action):

        """
        Take an action and and move the date forward.

        There are 7 actions: buyA, sellA, buyB, sellB, buyC, sellC and hold

        returns reward, next day's market status (the "state"), the day and the restrictions ("boundary")
        """
        print('===============6.in each step study action in environment========================')
        buy_volume = self.buy_volume
        abs_return_A = 0
        pct_return_A = 0
        abs_return_B = 0
        pct_return_B = 0
        A_cost = 0
        B_cost = 0
        A_return = 0
        B_return = 0
        temp = 0

        print('before updating, portfolio is ', self.portfolio)
        # This parameter has been used on a former version of the reward function.
        # On the present version of the program, it only serves to help calculate
        # one of the program's output for informative purposes (percentage returns).
        cost_reward = 1.0
        position_A = 4
        position_B = 1

        if (action == 'buyA'):

            if (self.portfolio['longA'] >= 0):  # i.e., the agent wants to buy A
                # and it is already bought on A. Thus it will open a new position.
                # Whenever a position is OPENED, costs are computed
                print('buyA and longA >0')
                A_cost = position_A * buy_volume
                #self.portfolio['a_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_A])
                #self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['longA'] += position_A
                pct_return_A = -position_A * cost_reward
                self.cum_operations += 1
                print('after', self.portfolio)


            else:  # longA < 0, i.e., the agent wants to buy A but it is sold on A. Thus it will close a (short) position.
                # Whenever a position is CLOSED, returns are computed
                print('buyA and longA <0')
                short_initial_1 = buy_volume
                abs_return_A = buy_volume - self.portfolio['a_vol'][-1] * self.prices.ix[self.date, self.stock_A]
                A_return = buy_volume
                A_return += (buy_volume - self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].pop()
                pct_return_A = float(abs_return_A) / short_initial_1
                self.portfolio['longA'] += position_A
                print('after', self.portfolio)

                if (self.portfolio['longA'] >= 0):
                    print('buyA and longA <0 then long A >0')
                    A_cost = buy_volume
                    self.portfolio['a_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_A])
                    self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                    self.portfolio['longA'] += position_A
                    pct_return_A += -position_A * cost_reward
                    self.cum_operations += 1
                    print('after', self.portfolio)

                else:  # longA < 0
                    print('buyA and longA <0 but long A still < 0')
                    short_initial_2 = buy_volume
                    abs_return_A = (buy_volume - self.portfolio['a_vol'][-1] * self.prices.ix[self.date, self.stock_A])
                    temp = buy_volume
                    temp += (buy_volume - self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A])
                    A_return += temp
                    self.portfolio['a_price'].pop()
                    # pct_return_A = float(abs_return_A)/(short_initial_1 + short_initial_2)
                    pct_return_A += float(abs_return_A) / (short_initial_2)
                    self.portfolio['longA'] += position_A
                    print('after', self.portfolio)

            if (self.portfolio['longB'] > 0):
                print('buyA and longB >0')
                long_initial = buy_volume
                B_return = self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B]
                abs_return_B = B_return - long_initial
                pct_return_B = float(abs_return_B) / long_initial
                self.portfolio['b_price'].pop()
                self.portfolio['longB'] -= position_B
                print('after', self.portfolio)
            else:  # longB <= 0
                print('buyA and longB <0')
                B_cost = buy_volume
                self.portfolio['b_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longB'] -= position_B
                pct_return_B = -position_B * cost_reward
                self.cum_operations += 1
                print('after', self.portfolio)

            # if (self.portfolio['longC'] > 0):
            #
            #     long_initial = buy_volume
            #     C_return = self.portfolio['c_vol'].pop() * self.prices.ix[self.date, self.stock_C]
            #     abs_return_C = C_return - long_initial
            #     pct_return_C = float(abs_return_C)/long_initial
            #     self.portfolio['c_price'].pop()
            #     self.portfolio['longC'] -= 1
            #
            # else: #longC <= 0
            #
            #     C_cost = buy_volume
            #     self.portfolio['c_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_C])
            #     self.portfolio['c_price'].append(self.prices.ix[self.date, self.stock_C])
            #     self.portfolio['longC'] -= 1
            #     pct_return_C = -1.0 * cost_reward
            #     self.cum_operations += 1


        elif (action == 'sellA'):

            if (self.portfolio['longA'] > 0):
                print('sellA and longA >0')
                long_initial_1 = buy_volume
                A_return = self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A]
                abs_return_A = A_return - long_initial_1
                pct_return_A = float(abs_return_A) / long_initial_1
                self.portfolio['a_price'].pop()
                self.portfolio['longA'] -= position_A
                print('after', self.portfolio)

                if (self.portfolio['longA'] > 0):
                    print('sellA and longA >0 and longA still >0')
                    long_initial_2 = buy_volume
                    temp = self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A]
                    A_return += temp
                    abs_return_A = (A_return - long_initial_2)
                    # pct_return_A = float(abs_return_A)/(long_initial_1 + long_initial_2)
                    pct_return_A += float(abs_return_A) / (long_initial_2)
                    self.portfolio['a_price'].pop()
                    self.portfolio['longA'] -= position_A
                    print('after', self.portfolio)

                else:  # longA <= 0
                    print('sellA and longA >0 then longA <0')
                    A_cost = buy_volume
                    self.portfolio['a_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_A])
                    self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                    self.portfolio['longA'] -= position_A
                    pct_return_A += -position_A * cost_reward
                    self.cum_operations += 1
                    print('after', self.portfolio)

            else:  # longA <= 0
                print('sellA and longA <0')
                A_cost = 1 * buy_volume
                #self.portfolio['a_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_A])
                #self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['longA'] -= position_A
                pct_return_A = -position_A * cost_reward
                self.cum_operations += 1
                print('after', self.portfolio)

            if (self.portfolio['longB'] >= 0):
                print('sellA and longB >=0')
                B_cost = buy_volume
                self.portfolio['b_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longB'] += position_B
                pct_return_B = -position_B * cost_reward
                self.cum_operations += 1
                print('after', self.portfolio)

            else:  # longB < 0
                print('sellA and longB < 0')
                short_initial = buy_volume
                abs_return_B = buy_volume - self.portfolio['b_vol'][-1] * self.prices.ix[self.date, self.stock_B]
                B_return = buy_volume
                B_return += (buy_volume - self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].pop()
                pct_return_B = float(abs_return_B) / short_initial
                self.portfolio['longB'] += position_B
                print('after', self.portfolio)


        elif (action == 'buyB'):

            if (self.portfolio['longB'] >= 0):
                print('buyB and longB >= 0')
                B_cost = position_B * buy_volume
                #self.portfolio['b_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_B])
               # self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longB'] += position_B
                pct_return_B = -position_B * cost_reward
                self.cum_operations += 1
                print('after', self.portfolio)


            else:  # longB < 0
                print('buyB and longB <0')
                short_initial_1 = buy_volume
                abs_return_B = buy_volume - self.portfolio['b_vol'][-1] * self.prices.ix[self.date, self.stock_B]
                B_return = buy_volume
                B_return += (buy_volume - self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].pop()
                pct_return_B = float(abs_return_B) / short_initial_1
                self.portfolio['longB'] += position_B
                print('after', self.portfolio)

                if (self.portfolio['longB'] >= 0):
                    print('buyB and longB <0 then longB >=0')
                    B_cost = buy_volume
                    self.portfolio['b_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_B])
                    self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                    self.portfolio['longB'] += position_B
                    pct_return_B += -position_B * cost_reward
                    self.cum_operations += 1
                    print('after', self.portfolio)

                else:  # longB < 0
                    print('buyB and longB <0 then longB <0')
                    short_initial_2 = buy_volume
                    abs_return_B = (buy_volume - self.portfolio['b_vol'][-1] * self.prices.ix[self.date, self.stock_B])
                    temp = buy_volume
                    temp += (buy_volume - self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B])
                    B_return += temp
                    self.portfolio['b_price'].pop()
                    # pct_return_B = float(abs_return_B)/(short_initial_1 + short_initial_2)
                    pct_return_B += float(abs_return_B) / (short_initial_2)
                    self.portfolio['longB'] += position_B
                    print('after', self.portfolio)

            if (self.portfolio['longA'] > 0):
                print('buyB and longA >0')
                long_initial = buy_volume
                A_return = self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A]
                abs_return_A = A_return - long_initial
                pct_return_A = float(abs_return_A) / long_initial
                self.portfolio['a_price'].pop()
                self.portfolio['longA'] -= position_A
                print('after', self.portfolio)

            else:  # longA <= 0
                print('buyB and longA <0')
                A_cost = buy_volume
                self.portfolio['a_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['longA'] -= position_A
                pct_return_A = -position_A * cost_reward
                self.cum_operations += 1
                print('after', self.portfolio)

        elif (action == 'sellB'):

            if (self.portfolio['longB'] > 0):
                print('sellB and longB>0')
                long_initial_1 = buy_volume
                B_return = self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B]
                abs_return_B = B_return - long_initial_1
                pct_return_B = float(abs_return_B) / long_initial_1
                self.portfolio['b_price'].pop()
                self.portfolio['longB'] -= position_B
                print('after', self.portfolio)

                if (self.portfolio['longB'] > 0):
                    print('sellB and longB>0 and long B still >0 ')
                    long_initial_2 = buy_volume
                    temp = self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B]
                    B_return += temp
                    abs_return_B = (B_return - long_initial_2)
                    # pct_return_B = float(abs_return_B)/(long_initial_1 + long_initial_2)
                    pct_return_B += float(abs_return_B) / (long_initial_2)
                    self.portfolio['b_price'].pop()
                    self.portfolio['longB'] -= position_B
                    print('after', self.portfolio)

                else:  # longB <= 0
                    print('sellB and longB>0 then long B <0 ')
                    B_cost = buy_volume
                    self.portfolio['b_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_B])
                    self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                    self.portfolio['longB'] -= position_B
                    pct_return_B += -position_B * cost_reward
                    self.cum_operations += 1
                    print('after', self.portfolio)


            else:  # longB <= 0
                print('sellB and longB<0')
                B_cost = 1 * buy_volume
                #self.portfolio['b_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_B])
                #self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longB'] -= position_B
                pct_return_B = -position_B * cost_reward
                self.cum_operations += 1
                print('after', self.portfolio)

            if (self.portfolio['longA'] >= 0):
                print('sellB and longA >0')
                A_cost = buy_volume
                self.portfolio['a_vol'].append(buy_volume / self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['longA'] += position_A
                pct_return_A = -position_A * cost_reward
                self.cum_operations += 1
                print('after', self.portfolio)

            else:  # longA < 0
                print('sellB and longA <0')
                short_initial = buy_volume
                abs_return_A = buy_volume - self.portfolio['a_vol'][-1] * self.prices.ix[self.date, self.stock_A]
                A_return = buy_volume
                A_return += (buy_volume - self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].pop()
                pct_return_A = float(abs_return_A) / short_initial
                self.portfolio['longA'] += position_A
                print('after', self.portfolio)

        # The portfolio cash receives the returns of closed positions and pays for the newly opened ones.
        print('after the cases, A_return, B_return, A_cost, B_cost', A_return, B_return, A_cost, B_cost)
        self.portfolio['cash'] = self.portfolio['cash'] + A_return + B_return - A_cost - B_cost

        # This variable accumulates the daily value of the portfolio's cash. In the end of the program's execution it will be used
        # to calculate the average daily value of the cash account
        self.cum_cash += self.portfolio['cash']

        old_port_val = self.port_val
        print('after the cases, the old portfolio value', old_port_val)
        # This is the portfolio evaluation method chosen for the calculation of the reward function. The stocks in the portfolio
        # are evaluated by the prices they had when the positions were opened. This precludes the simulator of rewarding the agent
        # for an increase in the market value of a specific asset. Otherwise the agent would be reinforced to accumulate
        # well-performing stocks instead of opening and closing positions pursuant to the pairs trading strategy
        self.port_val = self.port_value()
        print('after the cases, the new portfolio value', self.port_val)

        # This is an alternate portfolio evaluation method, based on the assets' current prices. I include it in the output files
        # only for comparison purposes
        self.port_val_market = self.port_value_for_output()
        print('after the cases, the market portfolio values', self.port_val_market)

        # The reward function
        reward = np.tanh(100 * (self.port_val - old_port_val) / self.init_cash)

        self.cum_reward += reward

        self.data_out.append(self.date.isoformat()[0:10] + ',' + str(self.portfolio['cash']) + ',' + str(
            self.prices.ix[self.date, self.stock_A]) + ',' + str(
            self.prices.ix[self.date, self.stock_B]) + ',' + action + ',' + str(abs_return_A) + ',' + str(
            pct_return_A) + ',' + str(abs_return_B) + ',' + str(pct_return_B) + ',' + str(
            self.prices_ibovespa.loc[self.date]) + ',' + str(self.cum_reward / self.dateIdx) + ',' + str(
            self.port_val) + ',' + str(self.port_val_market))
        print(' for this step in environment, the data out', self.data_out)

        self.dateIdx += 1
        if self.dateIdx < len(self.prices.index):
            self.date = self.prices.index[self.dateIdx]

        state = self.get_state(self.date)

        # The following function applies limitations to the closing of certain positions and to the opening of too many ones
        boundary = self.get_boundary()

        # resets the portfolio values when the simulation enters the testing period
        # (in the present version, the 11th year)
        #        if self.date >= self.dates_range[-1] - dt.timedelta(days=365) and self.check == 0:
        if self.date >= self.test_date and self.check == 0:
            self.portfolio = {'cash': self.init_cash, 'a_vol': [], 'a_price': [], 'b_vol': [], 'b_price': [],
                              'longA': 0, 'longB': 0}
            self.check = 1
            self.cum_cash = self.init_cash
            self.date_start = self.dateIdx
            self.cum_operations = 0

        return (reward, state, self.dateIdx, boundary)

    #    def plot_res(self, df, title):
    #
    #        fig, ((ax1,ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
    #        df[['port_val_market']].plot(title='cumulative return', ax=ax1)
    #        df[['port_val_market']].pct_change().plot(title='return', ax=ax2)
    # #        df[['port_val']].plot(title='port_val', ax=ax2)
    #        df[['A_price','B_price']].plot(title='prices', ax=ax3)
    #
    #        df['max'] = df['port_val_market'].cummax()
    #        df['drawdown'] = df['port_val_market'] - df['max']
    #        df[['drawdown']].plot(title='drawdown', ax=ax4)
    #
    #        act_A = df[['action']].replace('hold',0).replace('buyA',1).replace('sellA',-1).replace('buyB',0).replace('sellB',0)
    #        act_B = df[['action']].replace('hold',0).replace('buyA',0).replace('sellA',0).replace('buyB',1).replace('sellB',-1)
    #        acts=pd.concat([act_A, act_B], axis=1)
    #        acts.columns=['A','B']
    #        acts.plot(title='actions', ax=ax5)
    #        fig.suptitle(title)
    #        del df

    def get_state(self, date):
        """
        returns state of next day's market.
        """
        if date not in self.dates_range:
            if verbose: print('Date was out of bounds.')
            if verbose: print(date)
            exit

        print ('===========================prices index=======================',self.prices.index[-1])

        if (date == self.prices.index[-1]):
            file_name = "data_for_vis_%s.csv" % dt.datetime.now().strftime("%H-%M-%S")
            print ('======================file names====================',file_name)

            file = open(file_name, 'w');
            file.write(
                'date,cash,A_price,B_price,action,abs_return_A,pct_return_A,abs_return_B,pct_return_B,prices_ibovespa,cum_reward,port_val,port_val_market')
            file.write('\n')
            for line in self.data_out:
                file.write(line);
                file.write('\n')
            file.close()

            plot_df = pd.read_csv(file_name)
            plot_df['date'] = pd.to_datetime(plot_df['date'])
            plot_df = plot_df.set_index('date')

            train_df = plot_df[plot_df.index < self.test_date]
            #            self.plot_res(train_df.copy(), 'training')
            test_df = plot_df[plot_df.index >= self.test_date]

            # training
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
            train_df[['port_val_market']].plot(title='cumulative return', ax=ax1)
            train_df[['port_val_market']].pct_change().plot(title='return', ax=ax2)
            #        df[['port_val']].plot(title='port_val', ax=ax2)
            train_df[['A_price', 'B_price']].plot(title='prices', ax=ax3)

            train_df['max'] = train_df['port_val_market'].cummax()
            train_df['drawdown'] = train_df['port_val_market'] - train_df['max']
            train_df[['drawdown']].plot(title='drawdown', ax=ax4)

            act_A = train_df[['action']].replace('hold', 0).replace('buyA', 1).replace('sellA', -1).replace('buyB',
                                                                                                            0).replace(
                'sellB', 0)
            act_B = train_df[['action']].replace('hold', 0).replace('buyA', 0).replace('sellA', 0).replace('buyB',
                                                                                                           1).replace(
                'sellB', -1)
            acts = pd.concat([act_A, act_B], axis=1)
            acts.columns = ['A', 'B']
            acts.plot(title='actions', ax=ax5)
            fig.suptitle('training')

            # testing
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
            test_df[['port_val_market']].plot(title='cumulative return', ax=ax1)
            test_df[['port_val_market']].pct_change().plot(title='return', ax=ax2)
            #        df[['port_val']].plot(title='port_val', ax=ax2)
            test_df[['A_price', 'B_price']].plot(title='prices', ax=ax3)

            test_df['max'] = test_df['port_val_market'].cummax()
            test_df['drawdown'] = test_df['port_val_market'] - test_df['max']
            test_df[['drawdown']].plot(title='drawdown', ax=ax4)

            act_A = test_df[['action']].replace('hold', 0).replace('buyA', 1).replace('sellA', -1).replace('buyB',
                                                                                                           0).replace(
                'sellB', 0)
            act_B = test_df[['action']].replace('hold', 0).replace('buyA', 0).replace('sellA', 0).replace('buyB',
                                                                                                          1).replace(
                'sellB', -1)
            acts = pd.concat([act_A, act_B], axis=1)
            acts.columns = ['A', 'B']
            acts.plot(title='actions', ax=ax5)
            fig.suptitle('testing')

        # Calculates the mean between stocks A, B and C's normalized prices on the day corresponding to the variable "date".
        A_zscore = self.prices.ix[date, self.stock_A] / self.prices.ix[0, self.stock_A]
        B_zscore = self.prices.ix[date, self.stock_B] / self.prices.ix[0, self.stock_B]
        prices_mean = np.mean([A_zscore, B_zscore])

        # returns state of next day's market, i.e. the difference between each stock's normalized price and the mean of the three
        # normalized prices.
        #return [A_zscore - prices_mean, B_zscore - prices_mean]
        return [A_zscore-B_zscore]

    def get_boundary(self):

        boundary = np.array([1, 1, 1, 1, 1])
        default = [0, 0]
        max_positions = 4
        min_return = 0

        # Forbids the opening of more than "max_positions" positions in any share and in any direction (long or short).
        # If max_positions is large enough, no such limits are imposed on the agent.

        if self.portfolio['longA'] >= max_positions:
            boundary[0] = 0
            boundary[3] = 0
        if self.portfolio['longA'] <= -max_positions:
            boundary[1] = 0
            boundary[2] = 0
        if self.portfolio['longB'] >= max_positions:
            boundary[1] = 0
            boundary[2] = 0
        if self.portfolio['longB'] <= -max_positions:
            boundary[0] = 0
            boundary[3] = 0

        a_vol = default + self.portfolio['a_vol']
        b_vol = default + self.portfolio['b_vol']

        # Forbids the closing of operations with returns inferior to the thresold defined by the variable "min_return". In a true
        # life situation the setting of "min_return" may take into consideration the existence of transaction costs. If set to zero
        # it simply precludes the closing of operations with loss.

        if (self.portfolio['longA'] < 0) * (self.buy_volume - a_vol[-1] * self.prices.ix[self.date, self.stock_A]) + (
                self.portfolio['longA'] + 1 < 0) * (
                self.buy_volume - a_vol[-2] * self.prices.ix[self.date, self.stock_A]) + (
                self.portfolio['longB'] > 0) * (
                b_vol[-1] * self.prices.ix[self.date, self.stock_B] - self.buy_volume) < min_return:
            boundary[0] = 0

        if (self.portfolio['longA'] > 0) * (a_vol[-1] * self.prices.ix[self.date, self.stock_A] - self.buy_volume) + (
                self.portfolio['longA'] - 1 > 0) * (
                a_vol[-2] * self.prices.ix[self.date, self.stock_A] - self.buy_volume) + (
                self.portfolio['longB'] < 0) * (
                self.buy_volume - b_vol[-1] * self.prices.ix[self.date, self.stock_B]) < min_return:
            boundary[1] = 0

        if (self.portfolio['longB'] < 0) * (self.buy_volume - b_vol[-1] * self.prices.ix[self.date, self.stock_B]) + (
                self.portfolio['longB'] + 1 < 0) * (
                self.buy_volume - b_vol[-2] * self.prices.ix[self.date, self.stock_B]) + (
                self.portfolio['longA'] > 0) * (
                a_vol[-1] * self.prices.ix[self.date, self.stock_A] - self.buy_volume) < min_return:
            boundary[2] = 0

        if (self.portfolio['longB'] > 0) * (b_vol[-1] * self.prices.ix[self.date, self.stock_B] - self.buy_volume) + (
                self.portfolio['longB'] - 1 > 0) * (
                b_vol[-2] * self.prices.ix[self.date, self.stock_B] - self.buy_volume) + (
                self.portfolio['longA'] < 0) * (
                self.buy_volume - a_vol[-1] * self.prices.ix[self.date, self.stock_A]) < min_return:
            boundary[3] = 0

        return boundary

    # calculates portfolio based on the prices of acquisition
    def port_value(self):
        value = self.portfolio['cash']
        value += self.buy_volume * abs(self.portfolio['longA'])
        value += self.buy_volume * abs(self.portfolio['longB'])
        return value

    # calculates portfolio based on current market prices
    def port_value_for_output(self):
        buy_volume = self.buy_volume
        value = self.portfolio['cash']

        if (self.portfolio['longA'] > 0):
            for i in range(len(self.portfolio['a_vol'])):
                value += (self.portfolio['a_vol'][i] * self.prices.ix[self.date, self.stock_A])

        if (self.portfolio['longA'] < 0):
            for i in range(len(self.portfolio['a_vol'])):
                value += buy_volume
                value += (buy_volume - self.portfolio['a_vol'][i] * self.prices.ix[self.date, self.stock_A])

        if (self.portfolio['longB'] > 0):
            for i in range(len(self.portfolio['b_vol'])):
                value += (self.portfolio['b_vol'][i] * self.prices.ix[self.date, self.stock_B])

        if (self.portfolio['longB'] < 0):
            for i in range(len(self.portfolio['b_vol'])):
                value += buy_volume
                value += (buy_volume - self.portfolio['b_vol'][i] * self.prices.ix[self.date, self.stock_B])

        return value

    def has_more(self):
        if ((self.dateIdx < len(self.prices.index)) == False):
            print('\n\n\n*****')
            # Average daily cash account
            print(self.cum_cash / (self.dateIdx - self.date_start + 1))
            print('*****\n\n\n')
            # Final portfolio value in the testing year
            print(self.port_val)
            print(self.port_val_market)
            print('*****\n\n\n')
            # Number of positions opened in the testing year
            print(self.cum_operations)
            print('*****\n\n\n')
        return self.dateIdx < len(self.prices.index)