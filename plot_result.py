import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
file_name = 'data_for_vis_16-59-45.csv'
test_date=dt.datetime(2008, 8, 1)
def plot_result(file_name, test_date):
            plot_df = pd.read_csv(file_name)
            plot_df['date'] = pd.to_datetime(plot_df['date'])
            plot_df = plot_df.set_index('date')

            train_df = plot_df[plot_df.index < test_date]
            #            self.plot_res(train_df.copy(), 'training')
            test_df = plot_df[plot_df.index >= test_date]

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
            #plt.show()

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
            plt.show()

plot_result(file_name, test_date)