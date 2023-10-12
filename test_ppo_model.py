from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import os

# CONSTANTS
MODEL_DIR = "./train"

# 1. Set up your environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Fix for JoypadSpace (based on your training script)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Assuming there's a model saved; otherwise, add a check
model = PPO.load('./train/2140000_model.zip', env=env)

vec_env = model.get_env()
obs = vec_env.reset()

for step in range(50000):
    action, _ = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)

env.close()