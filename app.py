import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import *

if __name__ == "__main__":
    st.title("DQN Trading Bot Simulation")

    # Load the data
    data = pd.read_csv("multi_stock_data.csv", index_col=0)

    st.sidebar.header("Bot Settings")
    initial_balance = st.sidebar.number_input("Initial Balance", 1000)

    if st.sidebar.button("Run Bot"):
        env = MultiStockTradingEnv(data, initial_balance=initial_balance)
        agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
        
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        
        total_rewards = []
        
        for time in range(len(data) - 1):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_rewards.append(reward)
            
            if done:
                break
        
        st.write(f"Final Balance: ${env.balance:.2f}")
        st.write(f"Total Profit: ${env.total_profit:.2f}")
        st.line_chart(total_rewards)
