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
exp_dir_list = ["negRti_posRte_mse_4D_obv_w_speedchg_newPrior_noSlope_DQNhyperP_512net_noL2Reg_relu_learnSche_3k",
				"negRti_posRte_rho0.5_mse_4D_obv_w_speedchg_newPrior_DQNhyperP_512net_noL2Reg_relu_learnSche_3k",
				"negRti_posRte_rho0.25_mse_4D_obv_w_speedchg_newPrior_DQNhyperP_512net_noL2Reg_relu_learnSche_3k",
				"negRti_posRte_rho0.125_mse_4D_obv_w_speedchg_newPrior_DQNhyperP_512net_noL2Reg_relu_learnSche_3k",
				"negRti_noRte_mse_4D_obv_w_speedchg_newPrior_noSlope_DQNhyperP_512net_noL2Reg_relu_learnSche_3k"]
input_dirs = [parent_dir + exp_dir for exp_dir in exp_dir_list]
win_reward_np_list = []

for input_dir in input_dirs:
	result_dir = Path(input_dir)
	reward_list = []
	for _, npystr in enumerate(list(result_dir.glob("*.npy"))):
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


fig, ax = plt.subplots()
mean_rewards = np.mean(win_reward_np_list[0], axis=0)
std_rewards = np.std(win_reward_np_list[0], axis=0)
ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='red', label='AIF_rho_1.0', linewidth=0.5)
ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='pink', alpha=0.25)
mean_rewards = np.mean(win_reward_np_list[1], axis=0)
std_rewards = np.std(win_reward_np_list[1], axis=0)
ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='blue', label='AIF_rho_0.5', linewidth=0.5)
ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='cyan', alpha=0.25)
mean_rewards = np.mean(win_reward_np_list[2], axis=0)
std_rewards = np.std(win_reward_np_list[2], axis=0)
ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='green', label='AIF_rho_0.25', linewidth=0.5)
ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='lime', alpha=0.25)
mean_rewards = np.mean(win_reward_np_list[3], axis=0)
std_rewards = np.std(win_reward_np_list[3], axis=0)
ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='purple', label='AIF_rho_0.125', linewidth=0.5)
ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='violet', alpha=0.25)
mean_rewards = np.mean(win_reward_np_list[4], axis=0)
std_rewards = np.std(win_reward_np_list[4], axis=0)
ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='gold', label='AIF_rho_0.0', linewidth=0.5)
ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='yellow', alpha=0.25)
ax.legend(bbox_to_anchor=(0., 1.1, 1., .2), loc='lower left', fontsize='small', ncol=3, mode='expand', borderaxespad=0.)
ax.set_ylabel("Rewards")
ax.set_xlabel("Number of episodes")
ax.set_title("Window-averaged rewards")
fig.savefig(os.getcwd() + "/AIF_rho_compare.png", dpi=200, bbox_inches="tight")