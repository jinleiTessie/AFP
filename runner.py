import datetime as dt
import numpy as np
from simulator import Simulator
from actor_critic_agents import ActorAgent

"""
        
        Code based on previous code written by Yichen Shen and Yiding Zhao 
        ( <https://github.com/shenyichen105/Deep-Reinforcement-Learning-in-Stock-Trading> )
        
"""

def main():
    actions = ["buy", "sell", "hold"]
    hist_actions = []

    stock_close = pd.read_csv('stock_close.csv')

    lvt = stock_close['LVT US EQUITY']
    awk = stock_close[ 'AWK US EQUITY']

    environment = Simulator(['ITSA4', 'ITUB3'], start_date = dt.datetime(2007, 1, 1), end_date= dt.datetime(2018, 1, 1),test_date = dt.datetime(2012,1,1))

    # Creates instance of agent
    agent = ActorAgent(lookback = environment.init_state())

    # Choice of first action
    action = agent.init_query()
    #print ('choice of first action',action)


    step = 0





    while environment.has_more():

            # Maps action from id to name
            action = actions[action]
            hist_actions.append(action)

            print ('action from action list',action)

            # Simulation step (trading day). Apart from reward and state, the simulator provides the agent with the current day
            # and information concerning restrictions (boundary). The agent uses the first one in the computation of a kind of
            # epsilon-greedy policy; the second piece of information serves to limit the ability of the agent to open or close
            # positions in certain scenarios
            step +=1
            print ('for step', step)
            #reward, state, day, boundary = environment.step(action)
            reward, state, day = environment.step(action, hist_actions[-1])

            # Agent takes action
            #action = agent.query(state, reward, day, boundary)
            action = agent.query(state,reward,day)

            hist_actions.append(actions[action])




if __name__ == '__main__':
    main()


#reward, state, pair result, how RNN works
#five action currently