# -*- coding: utf-8 -*-
"""
Loaded a trained agent and test it under given conditions.
Options include:
1. Generating a video of the trajectory for the test episode.
2. Generating a video of the EFE values for each action throughout the episode.
3. Combining 2 videos generated previously into a synced gif.

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
import cv2
import time
sys.path.insert(0, 'utils/')
from utils import load_object
sys.path.insert(0, 'model/')
from interception_py_env import InterceptionEnv

model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_posRte_mse_4D_obv_w_speedchg_newPrior_noSlope_DQNhyperP_512net_noL2Reg_relu_learnSche_3k/"
env_id = 'interception' #'mcar' or 'interception'
target_speed_idx = 2
approach_angle_idx = 3
env_prior = 'prior_error' # or prior_obv or None
create_video = True
plot_efe = True
combine_videos = True

if env_id == 'interception':
    env = InterceptionEnv(target_speed_idx, approach_angle_idx, return_prior=env_prior)
    print("Interception environment with target_speed_idx {} and approach_angle_idx {}".format(target_speed_idx, approach_angle_idx))
elif env_id == 'mcar':
    import gym
    env = gym.make("MountainCar-v0")

FPS = 30
frame_duration = 1 / FPS

trial_num = 0
episode_num = 2650
qaiModel = load_object(model_save_path + "trial_{}_epd_{}.agent".format(trial_num, episode_num))
print("Loaded QAI model from {}".format(model_save_path))
qaiModel.epsilon.assign(0.0)

if create_video:
    _ = env.reset()
    env.step(0)
    img = env.render(mode='rgb_array')
    width = img.shape[1]
    height = img.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # 'XVID' with '.avi' or 'mp4v' with '.mp4' suffix

if plot_efe:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

observation = env.reset()
if create_video:
    video = cv2.VideoWriter(model_save_path+"trial_{}_epd_{}.avi".format(trial_num, episode_num), fourcc, float(FPS), (width, height))
if plot_efe:
    efe_list = []
prev_time = time.time()
done = False

while not done:
    obv = tf.convert_to_tensor(observation, dtype=tf.float32)
    obv = tf.expand_dims(obv, axis=0)
    if plot_efe:
        action, efe_values = qaiModel.act(obv, return_efe=True)
        efe_values = efe_values.numpy().squeeze()
        efe_list.append(efe_values)
    else:
        action = qaiModel.act(obv)
    action = action.numpy().squeeze()
    if env_prior is not None:
        next_obv, reward, done, prior, _ = env.step(action)
        # next_obv, reward, done, prior, _ = env.step()  # uncomment to use prior action
    else:
        next_obv, reward, done, _ = env.step(action)
    observation = next_obv    
    time.sleep(max(frame_duration - (time.time() - prev_time), 0))
    prev_time = time.time()
    if create_video:
        img = env.render(mode='rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)
    else:
        env.render()

if create_video:
    video.release()

if plot_efe:
    fig = plt.figure()
    ymax = np.max(np.asarray(efe_list))
    ymin = np.min(np.asarray(efe_list))
    ax = plt.axes(ylim=(ymin, ymax))
    if env_id == 'interception':
        ax.set_xlabel('Actions (pedal positions)')
        ax.set_ylabel('EFE values')
        ax.set_title('EFE values for each action (pedal position)')
        actions = ['2.0', '4.0', '8.0', '10.0', '12.0', '14.0']
    elif env_id == 'mcar':
        ax.set_xlabel('Actions')
        ax.set_ylabel('EFE values')
        actions = ['left', 'none', 'right']

    def init():
        if env_id == 'interception':
            colors = ['b','b','b','b','b','b']
        elif env_id == 'mcar':
            colors = ['b','b','b']
        colors[np.argmax(efe_list[0])] = 'r'
        bars = ax.bar(actions, efe_list[0], color=colors)
        return bars

    def update(i):
        for bar in ax.containers:
            bar.remove()
        efe_vals = efe_list[i]
        if env_id == 'interception':
            colors = ['b','b','b','b','b','b']
        elif env_id == 'mcar':
            colors = ['b','b','b']
        colors[np.argmax(efe_vals)] = 'r'
        bars = ax.bar(actions, efe_vals, color=colors)
        return bars
    
    print("Creating animation for EFE values...")
    ani = FuncAnimation(fig, update, init_func=init, frames=len(efe_list), blit=False)
    ani.save(os.path.join(model_save_path, "EFE_animation_trial_{}_epd_{}.avi".format(trial_num, episode_num)), fps=FPS, dpi=200)

env.close()
print("Test simulation finished.")

if combine_videos:
    import subprocess
    cmd_traj = ["ffmpeg", "-i", "trial_{}_epd_{}.avi".format(trial_num, episode_num), "-vf",
                "scale=640:-1", "trajectory_trial_{}_epd_{}.gif".format(trial_num, episode_num)]
    cmd_efe = ["ffmpeg", "-i", "EFE_animation_trial_{}_epd_{}.avi".format(trial_num, episode_num),
                "-vf", "scale=640:-1", "EFE_animation_trial_{}_epd_{}.gif".format(trial_num, episode_num)]
    cmd_combine = ["ffmpeg", "-i", "trajectory_trial_{}_epd_{}.gif".format(trial_num, episode_num),
                    "-i", "EFE_animation_trial_{}_epd_{}.gif".format(trial_num, episode_num),
                    "-filter_complex", "vstack=inputs=2", "combined_trial_{}_epd_{}.gif".format(trial_num, episode_num)]
    print("Creating combined animation gifs...")
    os.chdir(model_save_path)
    subprocess.run(cmd_traj)
    subprocess.run(cmd_efe)
    subprocess.run(cmd_combine)
    print("All gifs are created.")