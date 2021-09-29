# -*- coding: utf-8 -*-
"""
Collect transitions from trained QAI agent.
Data can be used for training prior preference
model later on.

@author: Zhizhuo (George) Yang
"""
import os
import logging
import sys
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import numpy as np
sys.path.insert(0, 'utils/')
from utils import load_object
sys.path.insert(0, 'model/')
from interception_py_env import InterceptionEnv

approach_angle_idx = 3
env_prior = None # 'prior_error' or prior_obv or None
num_episodes = 100

# load trained QAI model
model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_posRte_mse_4D_obv_w_speedchg_allSpeeds_PriorError_perfect_DQNhyperP_512net_noL2Reg_relu_learnSche_3k/"
trial_num = 1
episode_num = 850
qaiModel = load_object(model_save_path + "trial_{}_epd_{}.agent".format(trial_num, episode_num))
print("Loaded QAI model from {}".format(model_save_path))
qaiModel.epsilon.assign(0.0)

total_rewards = 0
transition_list = []
for ep in range(num_episodes):
    target_speed_idx = np.random.randint(3)
    env = InterceptionEnv(target_speed_idx, approach_angle_idx, return_prior=env_prior)
    print("Interception environment with target_speed_idx {} and approach_angle_idx {}".format(target_speed_idx, approach_angle_idx))
    ep_reward = 0
    observation = env.reset()
    done = False
    while not done:
        obv = tf.convert_to_tensor(observation, dtype=tf.float32)
        obv = tf.expand_dims(obv, axis=0)
        action = qaiModel.act(obv)
        action = action.numpy().squeeze()
        if env_prior is not None:
            next_obv, reward, done, prior, _ = env.step(action)
            # next_obv, reward, done, prior, _ = env.step()  # uncomment to use prior action
        else:
            next_obv, reward, done, _ = env.step(action)
        transition_list.append((observation, action, reward, next_obv, done))
        observation = next_obv
        ep_reward += reward
    print("Episode number {} reward {}".format(ep+1, ep_reward))
    total_rewards += ep_reward

env.close()
np.save(os.getcwd() + "/transition_tuples.npy", transition_list)
print("Average reward is {}".format(total_rewards/num_episodes))
print("Transition data collection finished.")