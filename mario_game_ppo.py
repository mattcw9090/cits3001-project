from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# PARAMETERS TO ADJUST
TIMESTEPS = 10000

if os.path.exists("ppo_mario"):
    zip_files = [file for file in os.listdir("ppo_mario") if file.endswith('.zip')]
    largest_zip_file = max(zip_files, key=lambda file: int(file.split('.')[0]))
    trained_steps = int(largest_zip_file.split('.')[0])

    model = PPO.load(f"ppo_mario/{largest_zip_file}", env=env)
    print("LOAD SUCCESSFUL\n")
else:
    model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1)
    trained_steps = 0

model.learn(total_timesteps=TIMESTEPS)
model.save(f"ppo_mario/{trained_steps + TIMESTEPS}")

vec_env = model.get_env()
obs = vec_env.reset()

for step in range(500):
    action, _ = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
env.close()
