class Backtest:
    
    def __init__(self, engine):  # trained engine
        
        self.price = engine.price
        
        self.train_period = engine.train_period
        self.valid_period = engine.valid_period
        self.test_period = engine.test_period
        
        self.y_train_pred = engine.y_train_pred
        self.y_valid_pred = engine.y_valid_pred
        self.y_test_pred = engine.y_test_pred
    
    
    def init_trade(self, which):
        '''initialize trade
        '''
        assert which in ['train', 'valid', 'test']
        
        # Account
        self.cash = self.init_cap = 1.0e8
        self.position = {}
        
        # trade date, open/close signal
        if which == 'train':
            self._trade_date = self.train_period
            self._y_pred = self.y_train_pred
        elif which == 'valid':
            self._trade_date = self.valid_period
            self._y_pred = self.y_valid_pred
        else:
            self._trade_date = self.test_period
            self._y_pred = self.y_test_pred
        
        self._price_list = self.price.loc[self._trade_date].values
        
        self._get_signal()
        
        self.__trade_date_iter = iter(self._trade_date)
        self.__price_list_iter = iter(self._price_list)
        
        self.__open_date = next(self._open_signal_iter)
        self.__close_date = next(self._close_signal_iter)
        
        # risk
        self.tdays = len(self._trade_date)
        self.total_returns = []
        self.risk_indicators_dict = {}
    
    def _get_signal(self):
        '''open and close signal
        '''
        raise NotImplementedError
        
    def __update_position_info(self, price):
        '''update position infomation daily
        '''
        if self.position:
            self.position['Bitcoin']['price'] = price
            self.position['Bitcoin']['cap'] = self.position['Bitcoin']['price'] * self.position['Bitcoin']['number']
            self.position['Bitcoin']['holding days'] += 1
    
    def __open(self, price):
        num = self.cash / price
        self.position['Bitcoin'] = {'price': price, 'number': num, 'cap': self.cash, 'holding days': 0}
        self.cash = 0
        
    def __close(self):
        self.cash = self.position['Bitcoin']['cap']
        self.position.pop('Bitcoin')
        
        
    # Risks
    def __cal_total_returns(self):
        self.total_cap = self.position['Bitcoin']['cap'] if self.position else self.cash
        self.total_returns.append(np.log(self.total_cap) - np.log(self.init_cap))
        
    def __cal_total_annualized_returns(self):
        self.total_annualized_returns = self.total_returns[-1] * 250.0 / self.tdays
        self.risk_indicators_dict['Annualized Return'] = self.total_annualized_returns
    
    def __cal_daily_returns(self):
        self.daily_returns = np.array(self.total_returns)[1:] - np.array(self.total_returns)[:-1]
    
    def __cal_annualized_volatility(self):
        self.annualized_volatility = self.daily_returns.std() * np.sqrt(250)
        self.risk_indicators_dict['Annualized Volatility'] = self.annualized_volatility

    def __cal_sharpe(self):
        self.sharpe = self.total_annualized_returns / self.annualized_volatility
        self.risk_indicators_dict['Sharpe Ratio'] = self.sharpe
        
    def __cal_max_drawdown(self):
        caps = self.init_cap * np.exp(self.total_returns)
        start = end = 0
        i = 0
        mdd = 0
        cap_max = caps[0]
        for k, cap in enumerate(caps):
            if k > i and 1 - cap / cap_max > mdd:
                mdd = 1 - cap / cap_max
                end = k
                start = i
            if cap > cap_max:
                cap_max = cap
                i = k
        if mdd == 0:
            self.max_drawdown = 0
            self.mdd_start = None
            self.mdd_end = None
        else:
            self.max_drawdown = mdd
            self.mdd_start = self._trade_date[start]
            self.mdd_end = self._trade_date[end]
        self.risk_indicators_dict['Max DrawDown'] = self.max_drawdown
    
    def __do_daily_calculation(self):
        self.__cal_total_returns()
        
    def get_risk_indicators(self):
        self.__cal_daily_returns()
        self.__cal_total_annualized_returns()
        self.__cal_annualized_volatility()
        self.__cal_sharpe()
        self.__cal_max_drawdown()
    
    def visualize_PNL(self, size=(12, 8), linewidth=('2', '3'), color=('r', 'black')):
        total_returns = pd.Series(self.total_returns, index=self._trade_date)
        fig = plt.figure(figsize=size)
        ax1 = fig.add_subplot(111)
        ax1.plot(total_returns, linewidth=linewidth[0], color=color[0])
        plt.axhline(0, linewidth=linewidth[1], color=color[1])
        ax1.set_xlim(left=self._trade_date[0])
        ax1.plot()
        ax1.yaxis.grid(True)

    
    def __iter__(self):
        return self
    
    def __next__(self):
        date = next(self.__trade_date_iter)
        price = next(self.__price_list_iter)

        self.__update_position_info(price)
        
        self.__do_daily_calculation()
        
        if date == self.__close_date:
            # close position
            self.__close()
            self.__close_date = next(self._close_signal_iter)
        
        if date == self.__open_date:
            # open position
            self.__open(price)
            self.__open_date = next(self._open_signal_iter)
        return
    
    def Run(self):
        for _ in self:
            pass
        self.get_risk_indicators()
