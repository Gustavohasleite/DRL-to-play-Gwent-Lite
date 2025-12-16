import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.memory = deque(maxlen=100000) # buffer maior
        self.gamma = 0.99    # foco maior no longo prazo
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997 # exploração mais longa
        self.learning_rate = 0.0005 # lr menor para rede maior
        self.batch_size = 128 # eficiência na gpu

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(self.hidden_size, activation='relu'))
        model.add(Dense(self.hidden_size, activation='relu'))
        model.add(Dense(self.hidden_size, activation='relu')) # 3 camadas de 256
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, use_epsilon=True):
        if use_epsilon and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0][0] for i in minibatch])
        next_states = np.array([i[3][0] for i in minibatch])

        current_qs = self.model.predict(states, verbose=0)
        next_qs = self.target_model.predict(next_states, verbose=0)

        X = []
        y = []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(next_qs[i])

            current_q = current_qs[i]
            current_q[action] = target

            X.append(state[0])
            y.append(current_q)

        self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0, batch_size=self.batch_size)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)