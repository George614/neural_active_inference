from collections import deque
import numpy as np
import random

class ReplayBuffer(object):
    '''
    Vanilla Experience Replay Buffer
    Ref: https://github.com/higgsfield/RL-Adventure
    '''
    def __init__(self, capacity, seed=None):
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)
        self.store_prior = False
        self.store_Rte = False

    def push(self, state, action, reward, next_state, done, R_te=None, prior=None):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        if R_te is not None:
            R_te = np.expand_dims(R_te, 0)
            if not self.store_Rte:
                self.store_Rte = True
        if prior is not None:
            if not self.store_prior:
                self.store_prior = True
            prior = np.expand_dims(prior, 0)
        if R_te is not None and prior is not None:
            self.buffer.append((state, action, reward, next_state, done, R_te, prior))
        elif R_te is None and prior is not None:
            self.buffer.append((state, action, reward, next_state, done, prior))
        elif R_te is not None and prior is None:
            self.buffer.append((state, action, reward, next_state, done, R_te))
        else:
            self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if self.store_Rte and self.store_prior:
            state, action, reward, next_state, done, R_te, prior = zip(*random.sample(self.buffer, batch_size))
        elif self.store_Rte and not self.store_prior:
            state, action, reward, next_state, done, R_te = zip(*random.sample(self.buffer, batch_size))
        elif not self.store_Rte and self.store_prior:
            state, action, reward, next_state, done, prior = zip(*random.sample(self.buffer, batch_size))
        else:
            state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        state      = np.asarray(np.concatenate(state), dtype=np.float32)
        next_state = np.asarray(np.concatenate(next_state), dtype=np.float32)
        action     = np.asarray(action, dtype=np.int32)
        reward     = np.asarray(reward, dtype=np.float32)
        done       = np.asarray(done, dtype=bool)
        if self.store_Rte and self.store_prior:
            R_te = np.asarray(R_te, dtype=np.float32)
            prior = np.asarray(np.concatenate(prior), dtype=np.float32)
            return (state, action, reward, next_state, done, R_te, prior)
        elif self.store_Rte and not self.store_prior:
            R_te = np.asarray(R_te, dtype=np.float32)
            return (state, action, reward, next_state, done, R_te)
        elif not self.store_Rte and self.store_prior:
            prior = np.asarray(np.concatenate(prior), dtype=np.float32)
            return (state, action, reward, next_state, done, prior)
        else:
            return (state, action, reward, next_state, done)

    def __len__(self):
        return len(self.buffer)


class NaivePrioritizedBuffer(object):
    '''
    Prioritized Experience Replay Buffer
    Ref: https://github.com/higgsfield/RL-Adventure
    '''
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.store_prior = False
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done, R_te, prior=None):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        if prior is not None:
            if not self.store_prior:
                self.store_prior = True
            prior = np.expand_dims(prior, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            if self.store_prior:
                self.buffer.append((state, action, reward, next_state, done, R_te, prior))
            else:
                self.buffer.append((state, action, reward, next_state, done, R_te))
        else:
            if self.store_prior:
                self.buffer[self.pos] = (state, action, reward, next_state, done, R_te, prior)
            else:
                self.buffer[self.pos] = (state, action, reward, next_state, done, R_te)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs  = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        batch       = list(zip(*samples))
        states      = np.asarray(np.concatenate(batch[0]), dtype=np.float32)
        actions     = np.asarray(batch[1], dtype=np.int32)
        rewards     = np.asarray(batch[2], dtype=np.float32)
        next_states = np.asarray(np.concatenate(batch[3]), dtype=np.float32)
        dones       = np.asarray(batch[4], dtype=bool)
        R_tes       = np.asarray(batch[5], dtype=np.float32)
        if self.store_prior:
            prior = np.asarray(np.concatenate(batch[6]), dtype=np.float32)
            return (states, actions, rewards, next_states, dones, R_tes, prior, indices, weights)

        return (states, actions, rewards, next_states, dones, R_tes, indices, weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class HindsightBuffer(object):
    '''
    Replay Buffer for storing hindsight error and
    offset and training the predictive component.
    '''
    def __init__(self, capacity, seed=None):
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, init_obv, hindsight_error):
        init_obv = np.expand_dims(init_obv, 0)
        self.buffer.append((init_obv, hindsight_error))

    def sample(self, batch_size):
        init_obv, hindsight_error = zip(*random.sample(self.buffer, batch_size))
        init_obv = np.asarray(np.concatenate(init_obv), dtype=np.float32)
        hindsight_error = np.asarray(hindsight_error, dtype=np.float32)
        return (init_obv, hindsight_error)

    def __len__(self):
        return len(self.buffer)