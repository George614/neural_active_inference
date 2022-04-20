# -*- coding: utf-8 -*-
"""
Plot figures and generate videos after training.

@author: Zhizhuo (George) Yang
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.animation import FuncAnimation
from pathlib import Path
from collections import deque
import subprocess

out_dir = "D:/Projects/neural_active_inference/exp/interception/qai/recogNN_noDelay_InstEpst_HdstBuffer_discount0_relu_learnSche_3k_tune2/"
result_dir = Path(out_dir)
num_trials = 5
num_episodes = 3000


def plot_efe(efe_list, trial_num, episode_num):
    fig = plt.figure()
    ymax = np.max(np.asarray(efe_list))
    ymin = np.min(np.asarray(efe_list))
    ax = plt.axes(ylim=(ymin, ymax))
    ax.set_xlabel('Actions (pedal positions)')
    ax.set_ylabel('EFE values')
    ax.set_title('EFE values for each action (pedal position)')
    actions = ['2.0', '4.0', '8.0', '10.0', '12.0', '14.0']

    def init():
        colors = ['b','b','b','b','b','b']
        colors[np.argmax(efe_list[0])] = 'r'
        bars = ax.bar(actions, efe_list[0], color=colors)
        return bars

    def update(i):
        for bar in ax.containers:
            bar.remove()
        efe_vals = efe_list[i]
        colors = ['b','b','b','b','b','b']
        colors[np.argmax(efe_vals)] = 'r'
        bars = ax.bar(actions, efe_vals, color=colors)
        return bars
    
    print("Creating animation for EFE values...")
    ani = FuncAnimation(fig, update, init_func=init, frames=len(efe_list), blit=False)
    ani.save(os.path.join(out_dir, "EFE_animation_trial_{}_epd_{}.avi".format(trial_num, episode_num)), fps=30, dpi=200)
     

def combine_videos(trial_num, episode_num):
    vfile = list(result_dir.glob("trial_{}_epd_{}*.avi".format(trial_num, episode_num)))[0].name
    target_speed_idx = vfile.split('.')[0].split('_')[-1]
    cmd_traj = ["ffmpeg", "-i", "trial_{}_epd_{}_tsidx_{}.avi".format(trial_num, episode_num, target_speed_idx), "-vf",
                "scale=640:-1", "trajectory_trial_{}_epd_{}_tsidx_{}.avi".format(trial_num, episode_num, target_speed_idx)]
    cmd_efe = ["ffmpeg", "-i", "EFE_animation_trial_{}_epd_{}.avi".format(trial_num, episode_num),
                "-vf", "scale=640:-1", "EFE_scaled_trial_{}_epd_{}.avi".format(trial_num, episode_num)]
    cmd_combine = ["ffmpeg", "-i", "trajectory_trial_{}_epd_{}_tsidx_{}.avi".format(trial_num, episode_num, target_speed_idx),
                    "-i", "EFE_scaled_trial_{}_epd_{}.avi".format(trial_num, episode_num),
                    "-filter_complex", "vstack=inputs=2", "combined_trial_{}_epd_{}_tsidx_{}.avi".format(trial_num, episode_num, target_speed_idx)]
    
    print("Creating combined animation gifs for trail {} epd {}".format(trial_num, episode_num))
    subprocess.run(cmd_traj)
    subprocess.run(cmd_efe)
    subprocess.run(cmd_combine)
    print("Done")


def plot_TTC_trial_progress(out_dir, TTC_list=None):
    if TTC_list is None:
        out_path = Path(out_dir)
        TTC_dir_list = list(out_path.glob("*TTCs.npy"))
        TTC_list = []
        for ttc in TTC_dir_list:
            TTC_trial = np.load(str(ttc), allow_pickle=True)
            TTC_list.append(TTC_trial)    
    for i in range(len(TTC_list)):
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_ylim(0.0, 3.0)
        # ax.scatter(np.arange(0, num_episodes, 25), TTC_list[i][0,:], marker="*", label='target_1st_order')
        # ax.scatter(np.arange(0, num_episodes, 25), TTC_list[i][1,:], marker="_", label='target_actual')
        # ax.scatter(np.arange(0, num_episodes, 25), TTC_list[i][2,:], marker=".", label='subject')
        ax.scatter(np.arange(TTC_list[i].shape[1]), TTC_list[i][0,:], marker="*", label='target_1st_order')
        ax.scatter(np.arange(TTC_list[i].shape[1]), TTC_list[i][1,:], marker="_", label='target_actual')
        ax.scatter(np.arange(TTC_list[i].shape[1]), TTC_list[i][2,:], marker=".", label='agent')
        ax.set_xlabel("checkpoints") # checkpoints are taken every 25 episodes
        ax.set_ylabel("Time in seconds")
        ax.set_title("TTC during a trial")
        ax.legend(loc='lower left', fontsize='xx-small', ncol=3, mode=None, borderaxespad=0.)
        fig.savefig(out_dir + "trial_{}_TTC_progress.png".format(i), dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_TTC_boxplot(out_dir, TTC_list=None, plot_fname=None):
    if TTC_list is None:
        out_path = Path(out_dir)
        TTC_dir_list = list(out_path.glob("*TTCs.npy"))
        TTC_list = []
        for ttc in TTC_dir_list:
            TTC_trial = np.load(str(ttc), allow_pickle=True)
            TTC_list.append(TTC_trial)
    fspeed0list = []
    fspeed1list = []
    fspeed2list = []
    for trial_TTCs in TTC_list:
        for i in range(trial_TTCs.shape[1]):
            fspeed_idx = trial_TTCs[3, i]
            if fspeed_idx == 0:
                fspeed0list.append(trial_TTCs[:3, i])
            elif fspeed_idx == 1:
                fspeed1list.append(trial_TTCs[:3, i])
            else:
                fspeed2list.append(trial_TTCs[:3, i])
    fspeed0np = np.asarray(fspeed0list)
    fspeed1np = np.asarray(fspeed1list)
    fspeed2np = np.asarray(fspeed2list)
    df0 = pd.DataFrame(fspeed0np, columns=["target first order", "target actual", "agent"])
    df1 = pd.DataFrame(fspeed1np, columns=["target first order", "target actual", "agent"])
    df2 = pd.DataFrame(fspeed2np, columns=["target first order", "target actual", "agent"])
    df0['Initial speed'] = np.repeat([11.25], len(df0))
    df1['Initial speed'] = np.repeat([9.47], len(df1))
    df2['Initial speed'] = np.repeat([8.18], len(df2))
    df_combined = pd.concat([df0, df1, df2], axis=0)
    dd = pd.melt(df_combined, id_vars=['Initial speed'], value_vars=['target first order','target actual', 'agent'], var_name='TTC Type')
    fig = plt.figure()
    ax = sns.boxplot(x='Initial speed', y='value', data=dd, hue='TTC Type')
    ax.set_ylabel('Time (s)')
    if plot_fname is not None:
        fig.savefig(out_dir + plot_fname + ".png", dpi=200, bbox_inches="tight")
    else:
        fig.savefig(out_dir + "TTC_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_EFE():
    EFE_list = list(result_dir.glob("*EFE_values.npy"))
    efe_value_list = []
    for efe in EFE_list:
        efe_values = np.load(str(efe), allow_pickle=True)
        efe_value_list.append(efe_values)
        
    for tr in range(len(efe_value_list)):
        for ep in range(num_episodes//25):
            plot_efe(efe_value_list[tr][ep], tr, ep*25)


def plot_hindsight_error():
    hindsight_dir_list = list(result_dir.glob("*hindsight_errors.npy"))
    hindsight_list = []
    for hindsights in hindsight_dir_list:
        hindsight_trial = np.load(str(hindsights), allow_pickle=True)
        hindsight_list.append(hindsight_trial)

    for i in range(len(hindsight_list)):
        fig, ax = plt.subplots(constrained_layout=True)
        # ax.plot(np.arange(0, num_episodes, 25), hindsight_list[i][:], marker="*")
        ax.plot(np.arange(0, num_episodes), hindsight_list[i][:], marker=".", markersize=3, linewidth=0.5)
        ax.axhline(y=0.0, color = 'red', linestyle = '--', linewidth=0.5)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Time in seconds")
        ax.set_title("Hindsight errors throughout a trial")
        fig.savefig(out_dir + "trial_{}_hindsight_errors.png".format(i), dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_alpha_beta():
    alpha_dir_list = list(result_dir.glob("*alpha.npy"))
    beta_dir_list = list(result_dir.glob("*beta.npy"))
    alpha_list = []
    beta_list = []
    for alpha_dir in alpha_dir_list:
        alpha_trial = np.load(str(alpha_dir), allow_pickle=True)
        alpha_list.append(alpha_trial)
    for beta_dir in beta_dir_list:
        beta_trial = np.load(str(beta_dir), allow_pickle=True)
        beta_list.append(beta_trial)
    for i in range(len(alpha_list)):
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(np.arange(len(alpha_list[0])), alpha_list[i][:], marker=".", label="alpha", color="red", markersize=0.5, linewidth=0.5)
        ax.plot(np.arange(len(beta_list[0])), beta_list[i][:], marker=".", label="beta", color="blue", markersize=0.5, linewidth=0.5)
        ax.set_xlabel("Episodes")
        ax.set_title("alpha and beta throughout a trial")
        ax.legend(loc='upper right', fontsize='x-small')
        fig.savefig(out_dir + "trial_{}_alpha_beta.png".format(i), dpi=200, bbox_inches="tight")
        plt.close(fig)
    alpha_np = np.asarray(alpha_list)
    beta_np = np.asarray(beta_list)
    fig, ax = plt.subplots()
    mean_alpha = np.mean(alpha_np, axis=0)
    std_alpha = np.std(alpha_np, axis=0)
    mean_beta = np.mean(beta_np, axis=0)
    std_beta = np.std(beta_np, axis=0)
    ax.plot(np.arange(len(mean_alpha)), mean_alpha, alpha=1.0, color='red', label='mean_alpha', linewidth=0.5)
    ax.fill_between(np.arange(len(mean_alpha)), mean_alpha - std_alpha, mean_alpha + std_alpha, color='pink', alpha=0.25)
    ax.plot(np.arange(len(mean_beta)), mean_beta, alpha=1.0, color='blue', label='mean_beta', linewidth=0.5)
    ax.fill_between(np.arange(len(mean_beta)), mean_beta - std_beta, mean_beta + std_beta, color='cyan', alpha=0.25)
    ax.legend(bbox_to_anchor=(0., 1.1, 1., .2), loc='lower left', fontsize='small', ncol=2, mode='expand', borderaxespad=0.)
    ax.set_xlabel("Number of episodes")
    ax.set_title("Averaged alpha and beta")
    fig.savefig(out_dir + "mean_alpha_beta.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_TTC_diff():
    TTC_diffs_dir_list = list(result_dir.glob("*TTC_diffs.npy"))
    TTC_diffs_list = []
    for TTC_diffs in TTC_diffs_dir_list:
        TTC_diffs_trial = np.load(str(TTC_diffs), allow_pickle=True)
        TTC_diffs_list.append(TTC_diffs_trial)

    offsets_dir_list = list(result_dir.glob("*failed_offsets.npy"))
    offsets_list = []
    for offsets in offsets_dir_list:
        offsets_trial = np.load(str(offsets), allow_pickle=True)
        offsets_list.append(offsets_trial)

    for i in range(len(TTC_diffs_list)):
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(TTC_diffs_list[i][:, 0], TTC_diffs_list[i][:, 1], label='TTC_diff', color='blue', marker=".", markersize=3, linewidth=0.5)
        if len(offsets_list) > 0 and len(offsets_list[i]) > 0:
            ax.plot(offsets_list[i][:, 0], offsets_list[i][:, 1], label='offset', color='green', marker=".", markersize=3, linewidth=0.5)
        ax.axhline(y=0.0, color = 'red', linestyle = '--', linewidth=0.5)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Time in seconds")
        ax.set_title("TTC difference at the end of each failed episode & offset")
        ax.legend(loc='upper right', fontsize='x-small')
        fig.savefig(out_dir + "trial_{}_TTC_diffs.png".format(i), dpi=200, bbox_inches="tight")
        plt.close(fig)


def calc_window_mean(window):
    mu = sum(window) * 1.0 / len(window)
    return mu


def plot_rewards():
    reward_list = []
    for _, npystr in enumerate(list(result_dir.glob("*R.npy"))):
        rewards = np.load(str(npystr))
        reward_list.append(rewards)
    win_reward_list = []
    for ep_rewards in reward_list:
        reward_window = deque(maxlen=100)
        trial_win_mean = []
        for i in range(len(ep_rewards)):    
            reward_window.append(ep_rewards[i])
            reward_win_mean = calc_window_mean(reward_window)
            trial_win_mean.append(reward_win_mean)
        win_reward_list.append(trial_win_mean)
    for tr in range(len(win_reward_list)):
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_ylim(0.0, 1.0)
        ax.plot(np.arange(len(win_reward_list[tr])), win_reward_list[tr], linewidth=0.5)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Rewards")
        fig.savefig(out_dir + "trial_{}_win_avg.png".format(tr), dpi=200)
        plt.close(fig)
    win_reward_np = np.asarray(win_reward_list)
    fig, ax = plt.subplots()
    mean_rewards = np.mean(win_reward_np, axis=0)
    std_rewards = np.std(win_reward_np, axis=0)
    ax.set_ylim(0.0, 1.0)
    ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='red', label='mean_rewards', linewidth=0.5)
    ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='pink', alpha=0.25)
    ax.legend(bbox_to_anchor=(0., 1.1, 1., .2), loc='lower left', fontsize='small', ncol=3, mode='expand', borderaxespad=0.)
    ax.set_ylabel("Rewards")
    ax.set_xlabel("Number of episodes")
    ax.set_title("Window-averaged rewards")
    fig.savefig(out_dir + "mean_win_rewards.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == '__main__':
    plot_hindsight_error()
    plot_alpha_beta()
    plot_TTC_boxplot(out_dir)
    plot_TTC_trial_progress(out_dir)
    plot_TTC_diff()
    plot_rewards()
    # plot_all_EFE()

    # os.chdir(out_dir)
    # for tr in range(num_trials):
    #     for ep in range(num_episodes//25):
    #         combine_videos(tr, ep*25)
            
    # print("All gifs are created.")
