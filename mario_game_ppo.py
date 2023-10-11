from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=0):
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


env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# CONSTANTS
TIMESTEPS = 5000
MODEL_DIR = "./train"
LOG_DIR = "./log"

if os.path.exists(MODEL_DIR):  # If train folder exists
    zip_files = [file for file in os.listdir(MODEL_DIR) if file.endswith('.zip')]
    if zip_files:  # If something in train folder
        largest_zip_file = max(zip_files, key=lambda file: int(file.split('_')[0]))
        trained_steps = int(largest_zip_file.split('_')[0])
        model_path = os.path.join(MODEL_DIR, largest_zip_file)
        model = PPO.load(model_path, env=env)
        print("LOAD SUCCESSFUL\n")
    else:  # If nothing in train folder
        model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1, tensorboard_log=LOG_DIR)
else:  # If train folder doesn't exist
    os.mkdir(MODEL_DIR)
    model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1, tensorboard_log=LOG_DIR)

# Create the callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path="./train")

model.learn(total_timesteps=TIMESTEPS, callback=callback, reset_num_timesteps=False)

# vec_env = model.get_env()
# obs = vec_env.reset()
#
# for step in range(500):
#     action, _ = model.predict(obs)
#     obs, reward, done, info = vec_env.step(action)
#
# env.close()
