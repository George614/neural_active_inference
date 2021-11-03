# -*- coding: utf-8 -*-
"""
Plot figures and generate videos after training.

@author: Zhizhuo (George) Yang
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import subprocess

out_dir = "D:/Projects/neural_active_inference/exp/interception/qai/delayed_action_noEpst_noEpsGreedy_hindsightError_DQNhyperP_512net_relu_learnSche_50/"
result_dir = Path(out_dir)
num_trials = 3
num_episodes = 50


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


def plot_TTC():
    TTC_dir_list = list(result_dir.glob("*TTCs.npy"))
    TTC_list = []
    for ttc in TTC_dir_list:
        TTC_trial = np.load(str(ttc), allow_pickle=True)
        TTC_list.append(TTC_trial)

    for i in range(len(TTC_list)):
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_ylim(0.0, 3.0)
        ax.scatter(np.arange(0, num_episodes, 25), TTC_list[i][0,:], marker="*", label='target_1st_order')
        ax.scatter(np.arange(0, num_episodes, 25), TTC_list[i][1,:], marker="_", label='target_actual')
        ax.scatter(np.arange(0, num_episodes, 25), TTC_list[i][2,:], marker=".", label='subject')
        ax.set_xlabel("episodes")
        ax.set_ylabel("Time in seconds")
        ax.set_title("TTC during a trial")
        ax.legend(loc='lower left', fontsize='xx-small', ncol=3, mode=None, borderaxespad=0.)
        fig.savefig(out_dir + "/trial_{}_TTC_compare.png".format(i), dpi=200, bbox_inches="tight")
    

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
        ax.plot(np.arange(0, num_episodes), hindsight_list[i][:], marker="*")
        ax.set_xlabel("episodes")
        ax.set_ylabel("Time in seconds")
        ax.set_title("Hindsight errors throughout a trial")
        fig.savefig(out_dir + "/trial_{}_hindsight_errors.png".format(i), dpi=200, bbox_inches="tight")


plot_TTC()
plot_hindsight_error()
# plot_all_EFE()

# os.chdir(out_dir)
# for tr in range(num_trials):
#     for ep in range(num_episodes//25):
#         combine_videos(tr, ep*25)
        
# print("All gifs are created.")
