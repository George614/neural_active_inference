# -*- coding: utf-8 -*-
"""
Read episodic rewards from given folders and plot window-averaged
rewards with mean (solid line) and std (shade). For comparison
purpose, different conditions are color coded.

@author: Zhizhuo (George) Yang
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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

parent_dir = "D:/Projects/neural_active_inference/exp/interception/qai/"
exp_dir_list = ["DQN_noDelay_RW_2x512net_relu_learnSche_3k",
				"simpleNN_firstOrderPriorEnv_noDelay_InstOnly_relu_learnSche_3k",
				"recogNN_noDelay_InstEpst_discount0.99_pedal1.0_relu_learnSche_3k",
				"recogNN_noDelay_InstEpst_DynamicHdstBuffer_discount0.99_relu_learnSche_3k_tune2"]
input_dirs = [parent_dir + exp_dir for exp_dir in exp_dir_list]
win_reward_np_list = []
pass_front_pert_list = []
plot_front_pert = False

for input_dir in input_dirs:
	result_dir = Path(input_dir)
	reward_list = []
	for _, npystr in enumerate(list(result_dir.glob("*R.npy"))):
	    rewards = np.load(str(npystr))
	    reward_list.append(rewards)
	win_reward_list = []
	for ep_rewards in reward_list:
	    reward_win = []
	    trial_win_mean = []
	    for i in range(len(ep_rewards)):    
	        reward_win.append(ep_rewards[i])
	        if len(reward_win) > 100:
	            reward_win.pop(0)
	        reward_win_mean = calc_window_mean(reward_win)
	        trial_win_mean.append(reward_win_mean)
	    win_reward_list.append(trial_win_mean)
	    
	win_reward_np = np.asarray(win_reward_list)
	win_reward_np_list.append(win_reward_np)
	if plot_front_pert:
		with open(input_dir + "/who_passes_first.txt", 'r') as f:
			percent_data = f.readlines()
		sub_str_list = percent_data[1::2]
		sub_percent = [float(strv.split(' ')[1][:-2]) for strv in sub_str_list]
		pass_front_pert_list.append(np.asarray(sub_percent))


fig, ax = plt.subplots()
mean_rewards = np.mean(win_reward_np_list[0], axis=0)
std_rewards = np.std(win_reward_np_list[0], axis=0)
ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='red', label='DQN_Reward', linewidth=0.5)
ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='pink', alpha=0.25)
mean_rewards = np.mean(win_reward_np_list[1], axis=0)
std_rewards = np.std(win_reward_np_list[1], axis=0)
ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='blue', label='AIF_InstOnly', linewidth=0.5)
ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='cyan', alpha=0.25)
mean_rewards = np.mean(win_reward_np_list[2], axis=0)
std_rewards = np.std(win_reward_np_list[2], axis=0)
ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='green', label='AIF_InstEpst', linewidth=0.5)
ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='lime', alpha=0.25)
# mean_rewards = np.mean(win_reward_np_list[3], axis=0)
# std_rewards = np.std(win_reward_np_list[3], axis=0)
# ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='purple', label='AIF_InstEpst_Hdst', linewidth=0.5)
# ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='violet', alpha=0.25)
# mean_rewards = np.mean(win_reward_np_list[4], axis=0)
# std_rewards = np.std(win_reward_np_list[4], axis=0)
# ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='orange', label='pedalLag_0.6', linewidth=0.5)
# ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='gold', alpha=0.25)
# mean_rewards = np.mean(win_reward_np_list[5], axis=0)
# std_rewards = np.std(win_reward_np_list[5], axis=0)
# ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='royalblue', label='pedalLag_0.4', linewidth=0.5)
# ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='cornflowerblue', alpha=0.25)
leg = ax.legend(bbox_to_anchor=(0., 1.02, 1., .2), loc='lower left', fontsize='medium', ncol=2, mode='expand', borderaxespad=0.)
for line in leg.get_lines():
	line.set_linewidth(2.0)
ax.set_ylabel("Mean window-averaged rewards")
ax.set_xlabel("Number of episodes")
# ax.set_title("Window-averaged rewards")
fig.savefig(os.getcwd() + "/DQN_AIFInstOnly_InstEpst_compare.png", dpi=300, bbox_inches="tight")

if plot_front_pert:
	fig = plt.figure()
	ax = fig.add_axes([0, 0, 1, 1])
	boxplot = ax.boxplot(pass_front_pert_list, notch=True)
	ax.set_xticklabels(['2.0', '1.5','1.2', '0.8', '0.6', '0.4'])
	ax.set_xlabel("Pedal lag coefficient")
	ax.set_ylabel("Percentage")
	ax.set_title("Ratio of subject passing in front of target")
	fig.savefig(os.getcwd() + "/pass_front_compare.png", dpi=300, bbox_inches="tight")
