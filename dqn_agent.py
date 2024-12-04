import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces
import pandas as pd

# Assume symbols is defined somewhere, representing the stock symbols
symbols = ["AAPL", "GOOGL", "MSFT"]

import numpy as np
import pandas as pd
import gym
from gym import spaces
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class MultiStockTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=1000):
        super(MultiStockTradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 15  # Start from the 15th step to ensure moving averages are available
        self.balance = initial_balance
        self.shares_held = {symbol: 0 for symbol in symbols}
        self.total_profit = 0
        self.action_space = spaces.Discrete(len(symbols) * 3)
        # Observation space: price, shares held, 5-day MA, 15-day MA, RSI, MACD, ATR per stock
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(symbols) * 7,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 15
        self.shares_held = {symbol: 0 for symbol in symbols}
        self.total_profit = 0
        return self._next_observation()

    def _next_observation(self):
        obs = []
        for symbol in symbols:
            obs.extend(self._get_indicators(symbol))
            obs.append(self.shares_held[symbol])  # Shares held for each stock
        return np.array(obs)

    def _get_indicators(self, symbol):
        """Calculate indicators for each stock based on current_step data."""
        close_prices = self.data[f"{symbol}_('Close', 'GOOGL')"].iloc[:self.current_step]

        # 5-day and 15-day Moving Averages
        ma_5 = close_prices.rolling(window=5).mean().iloc[-1] / 1000  # Normalized
        ma_15 = close_prices.rolling(window=15).mean().iloc[-1] / 1000  # Normalized

        # RSI Calculation (14-day)
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 0
        rsi /= 100  # Normalized

        # MACD Calculation (12-day EMA - 26-day EMA)
        ema_12 = close_prices.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = close_prices.ewm(span=26, adjust=False).mean().iloc[-1]
        macd = (ema_12 - ema_26) / 1000  # Normalized

        # ATR Calculation (14-day)
        high = self.data[f"{symbol}_('High', 'GOOGL')"].iloc[:self.current_step]
        low = self.data[f"{symbol}_('Low', 'GOOGL')"].iloc[:self.current_step]
        atr = (high - low).rolling(window=14).mean().iloc[-1] / 1000  # Normalized

        return [ma_5, ma_15, rsi, macd, atr]

    def step(self, action):
        total_reward = 0
        current_prices = {symbol: self.data[f"{symbol}_('Close', 'GOOGL')"].iloc[self.current_step] for symbol in symbols}

        for i, symbol in enumerate(symbols):
            stock_action = action % 3  # Determines buy/hold/sell per stock
            action = action // 3

            if stock_action == 1:  # Buy
                if self.balance >= current_prices[symbol]:
                    self.shares_held[symbol] += 1
                    self.balance -= current_prices[symbol]
                    total_reward += 1
            elif stock_action == 2:  # Sell
                if self.shares_held[symbol] > 0:
                    self.shares_held[symbol] -= 1
                    self.balance += current_prices[symbol]
                    total_reward += 1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self._next_observation()

        return obs, total_reward, done, {}

# DQN Model for Trading Agent
def build_dqn_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(action_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
    return model

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_dqn_model(state_size, action_size)
        self.target_model = build_dqn_model(state_size, action_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
