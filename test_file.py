from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from gym.wrappers import GrayScaleObservation
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])

done = True
state = env.reset()

for step in range(5000):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    done = terminated or truncated
    if done:
       state = env.reset()
env.close()