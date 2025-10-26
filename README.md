# TradeGym-RL

## Overview
The goal of this project is to show how a reinforcement learning (RL) agent can learn to trade stocks or other financial assets using price data. The agent interacts with a simulated market environment, making decisions to buy, sell, or hold based on observed data, with the goal of maximizing profit.

## How It Works
1. Load data — import a CSV file containing price data (e.g., Open, High, Low, Close, Volume).
2. Create environment — wrap the data in a gym-anytrading environment.
3. Train RL agent — use an algorithm such as PPO or DQN to train the agent.
4. Evaluate — test the trained model and visualize trades and profit.

## Quick Start

### 1. Clone the repository
git clone https://github.com/ines-besrour/TradeGym-RL.git

cd TradeGym-RL

### 2. Install dependencies
pip install gym-anytrading stable-baselines3 pandas matplotlib numpy jupyterlab

### 3. Run a basic training example
import gym
import gym_anytrading
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

df = pd.read_csv('data/your_data.csv', parse_dates=['Date'], index_col='Date')

env = gym.make('stocks-v0', df=df, frame_bound=(50, len(df)), window_size=10)
venv = DummyVecEnv([lambda: env])

model = PPO('MlpPolicy', venv, verbose=1)
model.learn(total_timesteps=10000)

env.render()
