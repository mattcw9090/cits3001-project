import os
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID "
os.environ["CUDA_VISIBLE_DEVICES"] = ""

env = gym.make('CartPole-v1', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
sampling_batch_size = 32
num_episodes = 10
output_dir = 'model_output/cartpole'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_memory = deque(maxlen=2000)
        self.discount_rate = 0.99
        self.exploration_rate = 1
        self.exploration_decay_rate = 0.995
        self.exploration_min = 0.01
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=legacy.Adam(learning_rate=self.learning_rate))

        return model

    def store_experience_in_replay_memory(self, state, action, reward, new_state, done):
        self.replay_memory.append((state, action, reward, new_state, done))

    def decide_action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        else:
            action_values = self.model.predict(state)
            return np.argmax(action_values[0])

    def sample_experiences_to_train_model(self, sampling_batch_size):
        sample_of_experiences = random.sample(self.replay_memory, sampling_batch_size)

        for experience in sample_of_experiences:
            state, action, reward, new_state, done = experience
            if done:
                target_q_value = reward
            else:
                target_q_value = reward + self.discount_rate * (np.amax(self.model.predict(new_state)[0]))
            output_q_value = self.model.predict(state)
            output_q_value[0][action] = target_q_value

            self.model.fit(state, output_q_value, epochs=1, verbose=0)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay_rate

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)

agent = DQNAgent(state_size, action_size)

# Training Phase
done = False

for episode in range(num_episodes):
    state = env.reset()[0]
    state = np.reshape(state, [1, state_size])

    for step in range(5):
        env.render()
        action = agent.decide_action(state)
        new_state, reward, done, truncated, info = env.step(action)
        reward = reward if not done else -10
        new_state = np.reshape(new_state, [1, state_size])
        agent.store_experience_in_replay_memory(state, action, reward, new_state, done)
        state = new_state

        if done:
            print("episode: {}/{}, score: {}, e:{:.2f}".format(episode, num_episodes, step, agent.exploration_rate))
            break

        if len(agent.replay_memory) > sampling_batch_size:
            agent.sample_experiences_to_train_model(sampling_batch_size)

        if episode % 50 == 0:
            agent.save_model(output_dir + "weights_" + '{:04d}'.format(episode) + ".hd5")