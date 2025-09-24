"""
Agent
The main Trading Agent implementation.
Copyright (c) 2018 Prabhsimran Singh
Licensed under the MIT License (see LICENSE for details)
Written by Prabhsimran Singh
"""
import random
from collections import deque
import heapq
import numpy as np
import tensorflow as tf
import keras.backend as K
from itertools import count
from keras.models import Sequential, Model
from keras.models import load_model, clone_model
from keras.layers import Dense, Lambda, Input, Add
from keras.optimizers import Adam
from .NoisyDense import NoisyDense
from keras.layers import Layer
from keras.utils import CustomObjectScope
import os


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """

    def __init__(self, state_size, strategy="t-dqn", reset_every=10000, pretrained=False, model_name=None, manual=False):
        self.strategy = strategy

        # agent config
        self.state_size = state_size    	# normalized previous days
        self.action_size = 3           		# [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.buffer = []
        self.first_iter = True
        self.nstep = 5
        self.n_step_buffer = deque(maxlen=self.nstep)
        self.cnt = count()
        self.alpha = 0.6
        
        # model config
        self.model_name = model_name
        self.gamma = 0.95  # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        
        # Fix optimizer compatibility
        try:
            self.optimizer = Adam(learning_rate=self.learning_rate)
        except:
            self.optimizer = Adam(lr=self.learning_rate)

        # Initialize model with error handling
        self.model = None
        if pretrained and self.model_name is not None:
            try:
                self.model = self.load(manual)
                print("Pre-trained model loaded successfully")
            except Exception as e:
                print(f"Failed to load pre-trained model: {e}")
                print("Creating new model instead...")
                self.model = self._model()
        else:
            self.model = self._model()

        # Ensure model was created successfully
        if self.model is None:
            print("Creating emergency fallback model...")
            self.model = self.create_fallback_model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            try:
                with CustomObjectScope({"NoisyDense": NoisyDense}):
                    # target network
                    self.target_model = clone_model(self.model)
                self.target_model.set_weights(self.model.get_weights())
            except Exception as e:
                print(f"Failed to create target model: {e}")
                print("Using main model as target model...")
                self.target_model = self.model

    def _model(self):
        """Creates the model with error handling"""
        try:
            X_input = Input((self.state_size,))
            X = Dense(units=128, activation="relu", input_dim=self.state_size)(X_input)
            X = Dense(units=256, activation="relu")(X)
            X = Dense(units=256, activation="relu")(X)
            
            # Try to use NoisyDense, fallback to regular Dense if it fails
            try:
                X = NoisyDense(units=128, activation="relu")(X)
                state_value = NoisyDense(1, activation="linear")(X)
                action_advantage = NoisyDense(self.action_size, activation="linear")(X)
            except Exception as e:
                print(f"NoisyDense failed, using regular Dense: {e}")
                X = Dense(units=128, activation="relu")(X)
                state_value = Dense(1, activation="linear")(X)
                action_advantage = Dense(self.action_size, activation="linear")(X)

            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.action_size,))(state_value)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(self.action_size,))(action_advantage)

            out = Add()([state_value, action_advantage])

            model = Model(inputs=X_input, outputs=out)
            model.compile(loss=self.loss, optimizer=self.optimizer)

            print("Dueling DQN model created successfully")
            return model

        except Exception as e:
            print(f"Failed to create dueling model: {e}")
            return self.create_simple_model()

    def create_simple_model(self):
        """Creates a simple sequential model as fallback"""
        try:
            model = Sequential()
            model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
            model.add(Dense(units=256, activation="relu"))
            model.add(Dense(units=256, activation="relu"))
            model.add(Dense(units=128, activation="relu"))
            model.add(Dense(units=self.action_size, activation="linear"))

            model.compile(loss=self.loss, optimizer=self.optimizer)
            print("Simple sequential model created successfully")
            return model
        
        except Exception as e:
            print(f"Failed to create simple model: {e}")
            return self.create_fallback_model()

    def create_fallback_model(self):
        """Creates a minimal model that should always work"""
        try:
            model = Sequential()
            model.add(Dense(self.action_size, input_dim=self.state_size, activation='linear'))
            
            # Use string optimizer name for maximum compatibility
            model.compile(loss='mse', optimizer='adam')
            print("Fallback model created successfully")
            return model
        
        except Exception as e:
            print(f"Even fallback model failed: {e}")
            return None

    def remember(self, state, action, reward, next_state, done, td_error):
        """Adds relevant data to memory"""
        # n-step queue for calculating return of n previous steps
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.nstep:
            return
        
        l_reward, l_next_state, l_done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            l_reward = r + self.gamma * l_reward * (1 - d)
            l_next_state, l_done = (n_s, d) if d else (l_next_state, l_done)
        
        l_state, l_action = self.n_step_buffer[0][:2]

        t = (l_state, l_action, l_reward, l_next_state, l_done)
        heapq.heappush(self.buffer, (-td_error, next(self.cnt), t))
        if len(self.buffer) > 100000:
            self.buffer = self.buffer[:-1]
        
        heapq.heapify(self.buffer)

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions"""
        try:
            # Handle case where model failed to load
            if self.model is None:
                print("Model is None, returning random action")
                return random.randrange(self.action_size)

            # take random action in order to diversify experience at the beginning
            if not is_eval and random.random() <= self.epsilon:
                return random.randrange(self.action_size)

            if self.first_iter:
                self.first_iter = False
                return 1  # make a definite buy on the first iter

            # Update n_iter to hard updating target model eventually
            self.n_iter += 1
            
            action_probs = self.model.predict(state, verbose=0)
            return np.argmax(action_probs[0])

        except Exception as e:
            print(f"Error in act method: {e}")
            return random.randrange(self.action_size)

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory"""
        try:
            if self.model is None:
                print("Cannot train: model is None")
                return 0.0

            if len(self.buffer) < batch_size:
                return 0.0

            # Semi Stochastic Prioritization
            prioritization = int(batch_size * self.alpha)
            batch_prioritized = heapq.nsmallest(prioritization, self.buffer)
            batch_uniform = random.sample(self.buffer, batch_size - prioritization)
            batch = batch_prioritized + batch_uniform
            batch = [e for (_, _, e) in batch]

            X_train, y_train = [], []
            
            # DQN with fixed targets
            if self.strategy == "t-dqn":
                if self.n_iter % self.reset_every == 0:
                    # reset target model weights
                    if self.target_model is not None:
                        self.target_model.set_weights(self.model.get_weights())

                for state, action, reward, next_state, done in batch:
                    if done:
                        target = reward
                    else:
                        # approximate deep q-learning equation with fixed targets
                        if self.target_model is not None:
                            target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
                        else:
                            target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

                    # estimate q-values based on current state
                    q_values = self.model.predict(state, verbose=0)
                    # update the target for current action based on discounted reward
                    q_values[0][action] = target

                    X_train.append(state[0])
                    y_train.append(q_values[0])

            else:
                # Default DQN behavior for other strategies
                for state, action, reward, next_state, done in batch:
                    if done:
                        target = reward
                    else:
                        target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

                    q_values = self.model.predict(state, verbose=0)
                    q_values[0][action] = target

                    X_train.append(state[0])
                    y_train.append(q_values[0])

            # update q-function parameters based on huber loss gradient
            loss = self.model.fit(
                np.array(X_train), np.array(y_train),
                epochs=1, verbose=0
            ).history["loss"][0]

            # as the training goes on we want the agent to
            # make less random and more optimal decisions
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            return loss

        except Exception as e:
            print(f"Error in training: {e}")
            return 0.0

    def calculate_td_error(self, state, action, reward, next_state, done):
        """Calculate TD error with error handling"""
        try:
            if self.model is None or self.target_model is None:
                return 0.0

            if not done:
                target = reward + self.gamma * self.target_model.predict(next_state, verbose=0)[0][np.argmax(self.model.predict(next_state, verbose=0)[0])]
            else:
                target = reward

            q_values = self.model.predict(state, verbose=0)[0][action]
            
            return q_values - target
        except Exception as e:
            print(f"Error calculating TD error: {e}")
            return 0.0

    def save(self, episode):
        """Save model with error handling"""
        try:
            if self.model is not None:
                self.model.save("models/{}_{}".format(self.model_name, episode))
                print(f"Model saved successfully: models/{self.model_name}_{episode}")
            else:
                print("Cannot save: model is None")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load(self, manual):
        """Load model with comprehensive error handling"""
        try:
            model_path = 'models/' + self.model_name if manual else 'rainbow/models/' + self.model_name
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Check file size to detect corruption
            file_size = os.path.getsize(model_path)
            if file_size < 1000:  # Model files should be larger than 1KB
                print(f"Model file appears to be corrupted (size: {file_size} bytes)")
                raise ValueError("Model file appears to be corrupted")
            
            print(f"Attempting to load model: {model_path}")
            
            # Try to load with custom objects
            try:
                model = load_model(model_path, custom_objects={'NoisyDense': NoisyDense, 'huber_loss': huber_loss})
                print("Model loaded with NoisyDense custom objects")
                return model
            except Exception as e1:
                print(f"Failed to load with NoisyDense: {e1}")
                
                # Try without custom objects
                try:
                    model = load_model(model_path)
                    print("Model loaded without custom objects")
                    return model
                except Exception as e2:
                    print(f"Failed to load without custom objects: {e2}")
                    raise e2
                    
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e