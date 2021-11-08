# -*- coding: utf-8 -*-
"""
Test the designed prior function on the interception task directly (choose 
the action by prior error) and plot the rewards.

@author: Zhizhuo (George) Yang
"""
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from interception_py_env import InterceptionEnv

def calc_window_mean(window):
    """
        Calculates the mean/average over a finite window of values
    """
    mu = 0.0
    for i in range(len(window)):
        v_i = window[i]
        mu += v_i
    mu = mu / (len(window) * 1.0)
    return mu

approach_angle_idx = 3
env_prior = 'prior_error'
use_slope = True
number_trials = 5
number_episodes = 1000

all_win_mean = []

for tr in range(number_trials):
    global_reward = []
    reward_window = []
    trial_win_mean = []
    
    for ep in tqdm(range(number_episodes)):
        target_speed_idx = np.random.randint(3)
        env = InterceptionEnv(target_speed_idx, approach_angle_idx, return_prior=env_prior, use_slope=use_slope)
        # print("Interception environment with target_speed_idx {} and approach_angle_idx {}".format(target_speed_idx, approach_angle_idx))
        env.reset()
        done = False
        episode_reward = 0
        while not done:
            next_obv, reward, done, prior, _ = env.step()
            episode_reward += reward
            
        global_reward.append(episode_reward)
        reward_window.append(episode_reward)
        if len(reward_window) > 100:
            reward_window.pop(0)
        reward_window_mean = calc_window_mean(reward_window)
        trial_win_mean.append(reward_window_mean)
    
    env.close()
    all_win_mean.append(np.asarray(trial_win_mean))

all_win_mean = np.stack(all_win_mean)
mean_rewards = np.mean(all_win_mean, axis=0)
std_rewards = np.std(all_win_mean, axis=0)
fig, ax = plt.subplots()
ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='red', label='mean', linewidth=0.5)
ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='pink', alpha=0.4)
ax.legend(loc='upper right')
ax.set_ylabel("Rewards")
ax.set_xlabel("Number of episodes")
ax.set_title("Window-averaged rewards")
fig.savefig(os.getcwd() + "/prior_mean_win_rewards.png", dpi=200)