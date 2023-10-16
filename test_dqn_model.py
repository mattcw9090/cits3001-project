from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from gym.wrappers import GrayScaleObservation
import os
import numpy as np
import csv

# CONSTANTS
MODEL_DIR = "./train"

# 1. Set up your environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Grayscale Conversion
env = GrayScaleObservation(env, keep_dim=True)

# Vectorize and Frame Stack
vec_env = DummyVecEnv([lambda: env])  # Vectorize environment
vec_env = VecFrameStack(vec_env, n_stack=4)  # Stack 4 frames

# Assuming there's a model saved; otherwise, add a check
model = DQN.load('./train/310000_model.zip', env=vec_env)

# vec_env = model.get_env()
obs = vec_env.reset()

for step in range(50000):
    action, _ = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)

env.close()
