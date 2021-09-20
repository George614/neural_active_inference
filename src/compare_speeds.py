# -*- coding: utf-8 -*-
"""
Loaded trained agents trained using different prior functions
then record their speeds during an episode on the interception
task under different target initial speeds.

@author: Zhizhuo (George) Yang
"""
import os
import logging
import sys
import pickle
import copy
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import numpy as np
sys.path.insert(0, 'utils/')
from utils import load_object
import matplotlib.pyplot as plt
sys.path.insert(0, 'model/')
from interception_py_env import InterceptionEnv


def load_AIF_agent(trial_num, episode_num, model_save_path):
    qaiModel = load_object(model_save_path + "trial_{}_epd_{}.agent".format(trial_num, episode_num))
    print("Loaded QAI model from {}".format(model_save_path))
    qaiModel.epsilon.assign(0.0)
    return qaiModel


def record_speeds(target_speed_idx, env, qaiModel=None):
    agent_speed_list = []  # with perfect prior
    required_speed_list = []
    first_order_speed_list = []
    distances_list = []  # tuple of (target distance, subject distance)
    observation = env.reset()
    done = False
    frame = 0
    interval = 25

    while not done:
        if frame % interval == 0:
            distances_list.append((env.state[0], env.state[2]))
        frame += 1
        obv = tf.convert_to_tensor(observation, dtype=tf.float32)
        obv = tf.expand_dims(obv, axis=0)
        action = qaiModel.act(obv)
        action = action.numpy().squeeze()
        next_obv, reward, done, prior, info = env.step(action)
        required_speed = env.info['required_speed']
        first_order_speed = env.info['1st-order_speed']
        agent_speed = env.state[3]
        observation = next_obv
        agent_speed_list.append(agent_speed)
        required_speed_list.append(required_speed)
        first_order_speed_list.append(first_order_speed)

    return np.asarray(agent_speed_list), np.asarray(required_speed_list), np.asarray(first_order_speed_list), reward, distances_list


if __name__ == "__main__":
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    # set testing conditions
    approach_angle_idx = 3
    env_prior = 'prior_error' # or prior_obv or None
    num_trials = 10
    
    # load the AIF agents trained with all target initial speeds
    model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_posRte_mse_4D_obv_w_speedchg_allSpeeds_PriorError_DQNhyperP_512net_noL2Reg_relu_learnSche_3k/"
    qaiModel1 = load_AIF_agent(0, 2950, model_save_path)
    model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_posRte_mse_4D_obv_w_speedchg_allSpeeds_PriorError_realtimePrior_DQNhyperP_512net_noL2Reg_relu_learnSche_3k/"
    qaiModel2 = load_AIF_agent(0, 2950, model_save_path)
    
    for target_speed_idx in [0, 1, 2]:
        env = InterceptionEnv(target_speed_idx, approach_angle_idx, return_prior=env_prior, perfect_prior=True)
        print("Interception environment with target_speed_idx {} and approach_angle_idx {}".format(target_speed_idx, approach_angle_idx))
        env2 = copy.deepcopy(env)

        for trial in range(num_trials):
            agent1_speeds, required_speeds1, first_order_speeds1, succeed1, dist_list1 = record_speeds(target_speed_idx, env, qaiModel=qaiModel1)
            agent2_speeds, required_speeds2, first_order_speeds2, succeed2, dist_list2 = record_speeds(target_speed_idx, env2, qaiModel=qaiModel2)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
            # plot speeds for different agents and different priors    
            ax1.plot(np.arange(len(agent1_speeds)), agent1_speeds, color='red', label='perfect_agent', linewidth=0.5)
            ax1.plot(np.arange(len(required_speeds1)), required_speeds1, color='orange', label='perfect_speed1', linewidth=0.5)
            ax1.plot(np.arange(len(first_order_speeds1)), first_order_speeds1, linestyle = '--', color='orange', label='1st-order_speed1', linewidth=0.5)
            ax1.plot(np.arange(len(agent2_speeds)), agent2_speeds, color='blue', label='1st-order_agent', linewidth=0.5)
            ax1.plot(np.arange(len(required_speeds2)), required_speeds2, color='cyan', label='perfect_speed2', linewidth=0.5)
            ax1.plot(np.arange(len(first_order_speeds2)), first_order_speeds2, linestyle = '--', color='cyan', label='1st-order_speed2', linewidth=0.5)
            ax1.axhline(y=max(env.action_speed_mappings), color = 'black', linestyle = '-', label='max_speed', linewidth=0.5)
            ax1.legend(bbox_to_anchor=(0., 1.1, 1., .2), loc='lower left', fontsize='xx-small', ncol=2, mode='expand', borderaxespad=0.)
            ax1.set_ylim([0, 15])
            ax1.set_ylabel("Speed")
            ax1.set_xlabel("Time steps")
            ax1.text(0, 13, "perfect_agent {}".format("succeed" if succeed1 else "failed", fontsize=5))
            ax1.text(0, 12, "1st-order_agent {}".format("succeed" if succeed2 else "failed", fontsize=5))
            ax1.set_title("Speed comparison")
            # plot connection lines between subject and target throughout an episode
            for dist_pair in dist_list1:
                ax2.plot([dist_pair[1], 0], [0, dist_pair[0]], 'go-', linewidth=2)
            ax2.set_xlim([0, env.subject_init_distance_max])
            ax2.set_ylim([0, env.target_init_distance])
            ax2.set_xlabel("Subject distance")
            ax2.set_ylabel("Target distance")
            ax2.set_title("Perfect agent")
            for dist_pair in dist_list2:
                ax3.plot([dist_pair[1], 0], [0, dist_pair[0]], 'bo-', linewidth=2)
            ax3.set_xlim([0, env.subject_init_distance_max])
            ax3.set_ylim([0, env.target_init_distance])
            ax3.set_xlabel("Subject distance")
            ax3.set_ylabel("Target distance")
            ax3.set_title("1st-order_agent")
            fig.savefig(os.getcwd() + "/plotting/speeds_compare_speedIdx_{}_trial_{}.png".format(target_speed_idx, trial+1), dpi=200, bbox_inches="tight")