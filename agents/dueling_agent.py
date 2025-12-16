import numpy as np
import random
import os
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization
from tensorflow.keras.optimizers import Adam

class DuelingAgent:
    def __init__(self, state_size, action_size, double_dqn=False):
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = double_dqn
        
        # hiperparâmetros otimizados para longo treino
        self.memory = deque(maxlen=20000) # buffer maior
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995 # decaimento mais lento para 10k episódios
        self.learning_rate = 0.00025 # learning rate mais refinado
        self.batch_size = 128 # maior para aproveitar a a100
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # dueling dqn architecture
        inputs = Input(shape=(self.state_size,))
        
        # feature extraction
        x = Dense(512, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # stream de valor (v)
        value = Dense(64, activation='relu')(x)
        value = Dense(1, activation='linear')(value)
        
        # stream de vantagem (a)
        advantage = Dense(64, activation='relu')(x)
        advantage = Dense(self.action_size, activation='linear')(advantage)
        
        # combinar v e a
        # isso garante estabilidade
        q_values = Add()([value, Subtract()([advantage, Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)])])
        
        model = Model(inputs=inputs, outputs=q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # previsão otimizada
        state = np.array(state).reshape(1, -1)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        # conversão em massa para numpy arrays
        states = np.array([i[0] for i in minibatch])
        if states.ndim == 3: states = np.squeeze(states, axis=1)
        
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        
        next_states = np.array([i[3] for i in minibatch])
        if next_states.ndim == 3: next_states = np.squeeze(next_states, axis=1)
        
        dones = np.array([i[4] for i in minibatch])

        # batch predictions
        # current q-states
        target = self.model.predict(states, verbose=0)
        
        # next q-states
        target_next = self.target_model.predict(next_states, verbose=0)
        
        if self.double_dqn:
            # ddqn logic
            online_next = self.model.predict(next_states, verbose=0)
            best_actions = np.argmax(online_next, axis=1)
            
            for i in range(self.batch_size):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    # q(s,a) = r + gamma * target_q(s', argmax(online_q(s', a')))
                    target[i][actions[i]] = rewards[i] + self.gamma * target_next[i][best_actions[i]]
        else:
            # standard dqn logic
            for i in range(self.batch_size):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        # treino em batch único
        self.model.fit(states, target, batch_size=self.batch_size, verbose=0, epochs=1)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)