# -*- coding: utf-8 -*-
"""
Collect transitions from trained QAI agent.
Data can be used for training prior preference
model later on.
"""

from utils import load_object
import numpy as np
import tensorflow as tf
import os
import logging
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)
sys.path.insert(0, "utils/")
sys.path.insert(0, "model/")
env_id = "MountainCar-v0"  # gym env names or 'interception'

if env_id == "interception":
    from interception_py_env import InterceptionEnv
else:
    import gym

    env = gym.make(env_id)

approach_angle_idx = 3
env_prior = None  # 'prior_error' or prior_obv or None
num_episodes = 100

# load trained QAI model
model_save_path = "D:/Projects/neural_active_inference/exp/mcar/qai/"
trial_num = 1
episode_num = 1950
qaiModel = load_object(
    model_save_path + "trial_{}_epd_{}.agent".format(trial_num, episode_num)
)
print("Loaded QAI model from {}".format(model_save_path))
qaiModel.epsilon.assign(0.0)

total_rewards = 0
transition_list = []
for ep in range(num_episodes):
    if env_id == "interception":
        target_speed_idx = np.random.randint(3)
        env = InterceptionEnv(
            target_speed_idx, approach_angle_idx, return_prior=env_prior
        )
        print(
            "Interception environment with target_speed_idx {} and approach_angle_idx {}".format(
                target_speed_idx, approach_angle_idx
            )
        )
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
    print("Episode number {} reward {}".format(ep + 1, ep_reward))
    total_rewards += ep_reward

env.close()
np.save(os.getcwd() + "/transition_tuples.npy", transition_list)
print("Average reward is {}".format(total_rewards / num_episodes))
print("Transition data collection finished.")
