from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from gym.wrappers import GrayScaleObservation
import os
import numpy as np
import csv

# CONSTANTS
TIMESTEPS = 20000
STEPS_TO_SAVE_AND_LOG_MODEL = 5000
MODEL_DIR = "./train"
LOG_DIR = "./log"
LEARNING_RATE = 1e-4

# Global variables for tracking rewards
reward_sum = 0
reward_counts = 0
mean_rewards_graph = []


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, env, check_freq, save_path, verbose=0):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        global reward_sum, reward_counts, mean_rewards_graph  # declare them as global
        if self.num_timesteps % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'{self.num_timesteps}_model')
            self.model.save(model_path)

            # Calculate mean reward
            mean_reward = reward_sum / reward_counts if reward_counts != 0 else 0

            # Append to CSV file
            with open('mean_rewards.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.num_timesteps, mean_reward])

            # Reset reward_sum and reward_counts
            reward_sum = 0
            reward_counts = 0

        return True


class MarioRewardShaping(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_info = {}

    def reset(self, **kwargs):
        self.prev_info = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        global reward_sum, reward_counts  # declare them as global
        state, reward, done, truncated, info = self.env.step(action)

        # Encourage forward movement
        if info['x_pos'] > self.prev_info.get('x_pos', 0):
            reward += 0.5
        elif info['x_pos'] < self.prev_info.get('x_pos', 0):
            reward -= 0.4
        elif info['x_pos'] == self.prev_info.get('x_pos', 0):
            reward -= 0.05

        # Add reward for getting coins
        if info['coins'] > self.prev_info.get('coins', 0):
            reward += 1
        
        # Death Penalty
        if info['life'] < self.prev_info.get('life', 2):
            reward-= 5

        if info['score'] > self.prev_info.get('score', 0):
            reward+= 0.5

        # Reward for level completion
        if info['flag_get']:
            reward += 10

        # Reward for getting a power-up
        if self.prev_info.get('status', '') == 'small' and info['status'] == 'big':
            reward += 1

        self.prev_info = info

        reward_sum += reward
        reward_counts += 1

        return state, reward, done, truncated, info


env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True)
env = MarioRewardShaping(env)
env = Monitor(env)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Grayscale Conversion
env = GrayScaleObservation(env, keep_dim=True)

# Vectorize and Frame Stack
vec_env = DummyVecEnv([lambda: env])  # Vectorize environment
vec_env = VecFrameStack(vec_env, n_stack=4)  # Stack 4 frames

if os.path.exists(MODEL_DIR):  # If train folder exists
    zip_files = [file for file in os.listdir(MODEL_DIR) if file.endswith('.zip')]
    if zip_files:  # If something in train folder
        largest_zip_file = max(zip_files, key=lambda file: int(file.split('_')[0]))
        model_path = os.path.join(MODEL_DIR, largest_zip_file)
        model = DQN.load(model_path, env=vec_env)
        print("LOAD SUCCESSFUL\n")
    else:  # If nothing in train folder
        model = DQN("CnnPolicy", vec_env, learning_rate=LEARNING_RATE, buffer_size = 1000, tensorboard_log=LOG_DIR)
else:  # If train folder doesn't exist
    os.mkdir(MODEL_DIR)
    model = DQN("CnnPolicy", vec_env, learning_rate=LEARNING_RATE, buffer_size = 1000, tensorboard_log=LOG_DIR)

# Create the callback
callback_instance = TrainAndLoggingCallback(env=vec_env, check_freq=STEPS_TO_SAVE_AND_LOG_MODEL, save_path=MODEL_DIR)


while True:
    vec_env.reset()
    model.learn(total_timesteps=TIMESTEPS, callback=callback_instance, reset_num_timesteps=False)
