# -*- coding: utf-8 -*-
"""
Test trained agents and record episodic (accumulated) epistemic value given 
different target initial speeds periodicly (based on the training progress).

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
import matplotlib.pyplot as plt
from pathlib import Path

approach_angle_idx = 3
num_runs = 5  # number of episodes for testing each specific config
dim_a = 6  # number of actions


def load_AIF_agent(trial_num, episode_num, model_save_path):
    qaiModel = load_object(model_save_path + "trial_{}_epd_{}.agent".format(trial_num, episode_num))
    print("Loaded QAI model from {}".format(model_save_path))
    qaiModel.epsilon.assign(0.0)
    return qaiModel


def record_Rte(target_speed_idx, qaiModel=None):
    env = InterceptionEnv(target_speed_idx, approach_angle_idx, return_prior=None)
    print("Interception environment with target_speed_idx {} and approach_angle_idx {}".format(target_speed_idx, approach_angle_idx))

    Rte_list = []

    for i in range(num_runs):
        observation = env.reset()
        done = False
        Rte_acum = []

        while not done:
            obv = tf.convert_to_tensor(observation, dtype=tf.float32)
            obv = tf.expand_dims(obv, axis=0)
            action = qaiModel.act(obv)
            a_t = tf.one_hot(action, depth=dim_a)
            o_next_tran_mu, _, _ = qaiModel.transition.predict(tf.concat([obv, a_t], axis=-1))
            action = action.numpy().squeeze()
            next_obv, reward, done, _ = env.step(action)
            delta = next_obv - o_next_tran_mu.numpy().squeeze()
            R_te = np.sum(delta * delta)
            Rte_acum.append(R_te)
            observation = next_obv

        Rte_list.append(np.sum(Rte_acum))

    Rte_avg = np.mean(Rte_list)
    return Rte_avg


def record_Rte_epd_means(model_save_path, speed_idx):
    """ 
    Averaged episodic epistemic value across all trials at interval
    of 50 episodes during the training progress for a given target 
    initial speed index.
    """
    model_dir = Path(model_save_path)
    num_trials = len(list(model_dir.glob("*.npy")))
    Rte_epd_means = []
    for epd_idx in range(0, 2001, 50):
        Rte_all_trials = []
        for trial in range(num_trials):
            qaiModel = load_AIF_agent(trial, epd_idx, model_save_path)
            Rte_snapshot = record_Rte(speed_idx, qaiModel)
            Rte_all_trials.append(Rte_snapshot)
        epd_Rte_mean = np.mean(Rte_all_trials)
        Rte_epd_means.append(epd_Rte_mean)
    return Rte_epd_means


# run the AIF agents trained separately with different target initial speeds
model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_posRte_mse_4D_obv_w_speedchg_fspeedIdx0_PriorError_DQNhyperP_512net_noL2Reg_relu_learnSche_3k/"
Rte_epd_means_speed_0 = record_Rte_epd_means(model_save_path, 0)
model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_posRte_mse_4D_obv_w_speedchg_fspeedIdx1_PriorError_DQNhyperP_512net_noL2Reg_relu_learnSche_3k/"
Rte_epd_means_speed_1 = record_Rte_epd_means(model_save_path, 1)
model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_posRte_mse_4D_obv_w_speedchg_fspeedIdx2_PriorError_DQNhyperP_512net_noL2Reg_relu_learnSche_3k/"
Rte_epd_means_speed_2 = record_Rte_epd_means(model_save_path, 2)
# run the AIF agent trained with different target initial speeds all together
model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_posRte_mse_4D_obv_w_speedchg_allSpeeds_PriorError_DQNhyperP_512net_noL2Reg_relu_learnSche_3k/"
Rte_epd_means_allSpeed_0 = record_Rte_epd_means(model_save_path, 0)
Rte_epd_means_allSpeed_1 = record_Rte_epd_means(model_save_path, 1)
Rte_epd_means_allSpeed_2 = record_Rte_epd_means(model_save_path, 2)
#%% plot the averaged episodic epistemic value as training progresses
fig, ax = plt.subplots()
ax.plot(np.arange(50, 2001, 50), Rte_epd_means_speed_0[1:], alpha=1.0, color='red', label='separate_speedIdx_0', linewidth=0.5)
ax.plot(np.arange(50, 2001, 50), Rte_epd_means_speed_1[1:], alpha=1.0, color='orange', label='separate_speedIdx_1', linewidth=0.5)
ax.plot(np.arange(50, 2001, 50), Rte_epd_means_speed_2[1:], alpha=1.0, color='gold', label='separate_speedIdx_2', linewidth=0.5)
ax.plot(np.arange(50, 2001, 50), Rte_epd_means_allSpeed_0[1:], alpha=1.0, color='green', label='together_speedIdx_0', linewidth=0.5)
ax.plot(np.arange(50, 2001, 50), Rte_epd_means_allSpeed_1[1:], alpha=1.0, color='blue', label='together_speedIdx_1', linewidth=0.5)
ax.plot(np.arange(50, 2001, 50), Rte_epd_means_allSpeed_2[1:], alpha=1.0, color='purple', label='together_speedIdx_2', linewidth=0.5)
ax.legend(bbox_to_anchor=(0., 1.1, 1., .2), loc='lower left', fontsize='x-small', ncol=3, mode='expand', borderaxespad=0.)
ax.set_ylabel("Epistemic value")
ax.set_xlabel("Number of episodes trained")
ax.set_title("Averaged episodic epistemic value for all initial target speeds")
fig.savefig(os.getcwd() + "/epistemic_comparison.png", dpi=200, bbox_inches="tight")