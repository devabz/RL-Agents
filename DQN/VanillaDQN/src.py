from collections import deque
import os
import gym
import random
import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm
from collections import defaultdict
from moviepy.editor import ImageSequenceClip


class ExperienceReplayBuffer:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def save(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, k=min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self, action_size, memory_size):
        self._h1 = tf.keras.layers.Dense(64, activation='relu')
        self._h2 = tf.keras.layers.Dense(64, activation='relu')

        self.targetNetwork = self._create_network(output_size=action_size)
        self.valueNetwork = self._create_network(output_size=action_size)

        self.buffer = ExperienceReplayBuffer(memory_size)

        self.gamma = 1
        self.logs = defaultdict(list)



    def _create_network(self, output_size):
        model = tf.keras.models.Sequential([
            self._h1,
            self._h2,
            tf.keras.layers.Dense(output_size, activation=None)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def learn(self):
        pass

    def _retrain_value_network(self, batch_size, epochs):
        if len(self.buffer) >= batch_size:
            batch = self.buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
            target = self.targetNetwork.predict(next_states, verbose=0)
            td_target = rewards + self.gamma * np.max(target, axis=1) * (1 - dones)
            td_target = td_target.reshape((-1, 1))
            loss = self.valueNetwork.fit(states, td_target, epochs=epochs)
            self.logs['loss'].append(loss)

    def _retrain_target_network(self, batch_size, epochs):
        if len(self.buffer) >= batch_size:
            batch = self.buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
            target = self.valueNetwork.predict(next_states, verbose=0)
            td_target = rewards + self.gamma * np.max(target, axis=1) * (1 - dones)
            td_target = td_target.reshape((-1, 1))
            loss = self.targetNetwork.fit(states, td_target, epochs=epochs)
            self.logs['loss'].append(loss)

    def train(self, env: gym.Env, n_episodes, batch_size, destination, epochs=1, rn_episodes=6, fps=10, pad_frames=16):
        episode_rewards = []
        eps = 1
        eps_d = 2 / n_episodes
        frames = []
        for episode in tqdm(range(n_episodes)):
            done = False
            state, *_ = env.reset()
            rewards = []
            while not done:
                if episode % (n_episodes//rn_episodes) == 0:
                    frames.append(env.render())

                action = env.action_space.sample()
                if np.random.random() < 1 - eps:
                    predictions = self.valueNetwork.predict(state.reshape(1, -1), verbose=0)
                    action = np.argmax(predictions)

                state_, reward, done, *_ = env.step(action)
                self.buffer.save((state, action, reward, state_, done))
                self._retrain_value_network(batch_size=batch_size, epochs=epochs)
                state = state_
                rewards.append(reward)

                if done:
                    for _ in range(pad_frames):
                        frames.append(env.render())
                        action = env.action_space.sample()
                        env.step(action)

            episode_rewards.append(sum(rewards))
            eps -= eps_d

            for _ in range(5):
                self._retrain_target_network(batch_size=batch_size, epochs=epochs)

        dir_, _ = os.path.split(destination)
        os.makedirs(dir_, fps, exist_ok=True)
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(destination, codec='libx264', logger=None)
        return episode_rewards

    def record(self, env: gym.Env, n_episodes):
        frames = []
        for episode in tqdm(range(n_episodes)):
            done = False
            state, *_ = env.reset()
            while not done:
                frames.append(env.render())
                predictions = self.valueNetwork.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(predictions)
                state, reward, done, *_ = env.step(action)
            frames.append(env.render())

        os.makedirs("test", exist_ok=True)
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(f"test/eval.mp4", codec='libx264', logger=None)
