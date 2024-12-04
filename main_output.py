import pandas as pd
import numpy as np
from dqn_agent import DQNAgent
from dqn_agent import MultiStockTradingEnv

def main():
    data = pd.read_csv("multi_stock_data.csv", index_col=0)

    initial_balance = 1000
    env = MultiStockTradingEnv(data, initial_balance=initial_balance)
    
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    
    episodes = 100  # Number of training episodes
    for e in range(episodes):
        # Reset environment and reshape state
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        total_rewards = 0
        done = False
        
        # Print debug information
        print(f"Starting Episode {e + 1}/{episodes}")
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_rewards += reward
            
            # Print debug information for each step
            # print(f"  Step reward: {reward:.2f}, Balance: {env.balance:.2f}")

        # Training after each episode
        agent.replay(100)  
        
        # Print results after each episode

    # Print final balance
    print(f"Final Balance after {episodes} episodes: ${env.balance:.2f}")

if __name__ == "__main__":
    main()
