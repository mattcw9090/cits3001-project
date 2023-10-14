from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np

# CONSTANTS
TIMESTEPS = 20000
STEPS_TO_SAVE_AND_LOG_MODEL = 10000
MODEL_DIR = "./train"
LOG_DIR = "./log"
LEARNING_RATE = 1e-5
ENT_COEF = 0.05

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, env, check_freq, save_path, verbose=0):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'{self.num_timesteps}_model')
            self.model.save(model_path)

        return True
class MarioRewardShaping(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_info = {}

    def reset(self, **kwargs):
        self.prev_info = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        # Encourage forward movement
        if info['x_pos'] > self.prev_info.get('x_pos', 0):
            reward += 1

        # Add reward for getting coins
        if info['coins'] > self.prev_info.get('coins', 0):
            reward += 5

        # Reward for level completion
        if info['flag_get']:
            reward += 100

        # Penalty for taking damage
        if self.prev_info.get('status', '') == 'big' and info['status'] == 'small':
            reward -= 10

        # Reward for getting a power-up
        if self.prev_info.get('status', '') == 'small' and info['status'] == 'big':
            reward += 10

        # Time-based penalty
        reward -= 0.05

        # Penalty for falling into pits
        if info['y_pos'] - self.prev_info.get('y_pos', info['y_pos']) > 10:
            reward -= 5

        # Penalize for death
        if info['life'] < self.prev_info.get('life', 2):
            reward -= 50

        # Penalize not progressing
        if info['x_pos'] == self.prev_info.get('x_pos', 0):
            reward -= 0.1

        self.prev_info = info
        return state, reward, done, truncated, info


env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True)
env = MarioRewardShaping(env)
env = Monitor(env)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

if os.path.exists(MODEL_DIR):  # If train folder exists
    zip_files = [file for file in os.listdir(MODEL_DIR) if file.endswith('.zip')]
    if zip_files:  # If something in train folder
        largest_zip_file = max(zip_files, key=lambda file: int(file.split('_')[0]))
        model_path = os.path.join(MODEL_DIR, largest_zip_file)
        model = PPO.load(model_path, env=env)
        print("LOAD SUCCESSFUL\n")
    else:  # If nothing in train folder
        model = PPO("MlpPolicy", env, learning_rate=LEARNING_RATE, tensorboard_log=LOG_DIR, ent_coef=ENT_COEF)
else:  # If train folder doesn't exist
    os.mkdir(MODEL_DIR)
    model = PPO("MlpPolicy", env, learning_rate=LEARNING_RATE, tensorboard_log=LOG_DIR, ent_coef=ENT_COEF)

# Create the callback
callback_instance = TrainAndLoggingCallback(env=env, check_freq=STEPS_TO_SAVE_AND_LOG_MODEL, save_path=MODEL_DIR)

while True:
    env.reset()
    model.learn(total_timesteps=TIMESTEPS, callback=callback_instance, reset_num_timesteps=False)
