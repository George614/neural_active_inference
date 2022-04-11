# -*- coding: utf-8 -*-
"""
Loaded a trained agent, test it and record 1) mean rewards,
2) percetange of subject pasing in front of target in failed
episodes, 3)  and TTC plots.

@author: Zhizhuo (George) Yang
"""
import os
import logging
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import numpy as np
import time
sys.path.insert(0, 'utils/')
from utils import load_object
sys.path.insert(0, 'model/')
from interception_py_env import InterceptionEnv
sys.path.insert(0, 'plotting/')
from plot_utils import plot_TTC_boxplot
from pathlib import Path

folder_names = ["recogNN_noDelay_InstEpst0.25_hdst3x_512net_relu_learnSche_3k_bonusRW",
                "recogNN_noDelay_InstEpst0.25_Nohdst_512net_relu_learnSche_3k_bonusRW",
                "recogNN_noDelay_InstOnly_hdst3x_512net_relu_learnSche_3k_bonusRW",
                "DQN_noDelay_RW_2x512net_relu_learnSche_3k",
                "recogNN_noDelay_pedal0.6_InstEpst0.25_hdst3x_512net_relu_learnSche_3k_bonusRW",
                "recogNN_noDelay_pedal0.6_InstOnly_hdst3x_512net_relu_learnSche_3k_bonusRW",
                "recogNN_noDelay_pedal0.6_InstEpst0.25_noHdst_512net_relu_learnSche_3k_bonusRW",
                "recogNN_noDelay_pedal0.5_InstEpst0.25_noHdst_512net_relu_learnSche_3k_bonusRW",
                "recogNN_noDelay_pedal0.5_InstOnly_hdst3x_512net_relu_learnSche_3k_bonusRW",
                "recogNN_noDelay_pedal0.5_InstEpst0.25_hdst3x_512net_relu_learnSche_3k_bonusRW",
                "recogNN_noDelay_InstOnly_noHdst_pedal0.6_relu_learnSche_3k",
                "recogNN_noDelay_InstEpst0.25_Outside_Hdst_pedal0.6_relu_learnSche_3k",
                "recogNN_noDelay_pedal1.0_InstOnly_noHdst_512net_relu_learnSche_3k",
                "recogNN_noDelay_pedal1.0_InstOnly_noHdst_noTransGrads_512net_relu_learnSche_3k"]
model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/" + folder_names[12] + "/"
test_episodes = 30
env_prior = 'prior_error' # or prior_obv or None
perfect_prior = False
# trial_num = 0
# episode_num = 2900
# qaiModel = load_object(model_save_path + "trial_{}_epd_{}.agent".format(trial_num, episode_num))

final_model_path = Path(model_save_path + "final_agents/")
agent_list = list(final_model_path.glob("*.agent"))
# model_every500_path = Path(model_save_path)
# agent_list = list(model_every500_path.glob("*_500.agent"))
agent_name = "_final_agent"
reward_list = []
TTC_list = []
target_front_count = 0
subject_front_count = 0

for trial_agent in agent_list:
    qaiModel = load_object(str(trial_agent))
    print("Loaded QAI model from {}".format(str(trial_agent)))
    qaiModel.epsilon.assign(0.0)

    target_1st_order_TTC_list = []
    target_actual_mean_TTC_list = []
    agent_TTC_list = []
    f_speed_idx_list = []
    TTC_diff_list = []

    for _ in range(test_episodes):
        f_speed_idx = np.random.randint(3)
        env = InterceptionEnv(target_speed_idx=f_speed_idx, approach_angle_idx=3, return_prior=env_prior, use_slope=False, perfect_prior=perfect_prior)
        observation = env.reset()
        TTC_calculated = False
        episode_reward = 0
        done = False
        while not done:
            obv = tf.convert_to_tensor(observation, dtype=tf.float32)
            obv = tf.expand_dims(obv, axis=0)
            action = qaiModel.act(obv)
            action = action.numpy().squeeze()
            next_obv, reward, done, obv_prior, info = env.step(action)
            episode_reward += reward
            # calculate and record TTC
            speed_phase = info['speed_phase']
            if speed_phase == 1 and not TTC_calculated:
                target_1st_order_TTC = env.state[0] / env.state[1]
                target_actual_mean_TTC = info['target_TTC']
                agent_TTC = env.state[2] / env.state[3]
                TTC_calculated = True
                target_1st_order_TTC_list.append(target_1st_order_TTC)
                target_actual_mean_TTC_list.append(target_actual_mean_TTC)
                agent_TTC_list.append(agent_TTC)
                f_speed_idx_list.append(f_speed_idx)
            observation = next_obv
        
        reward_list.append(episode_reward)
        if episode_reward == 0:
            # record who passes the interception point first
            if env.state[0] < env.state[2]:
                target_front_count += 1
            else:
                subject_front_count += 1
            # record TTC difference
            TTC_diff = env.state[2] / env.state[3] - env.state[0] / env.state[1]
            TTC_diff_list.append((f_speed_idx, TTC_diff))

    trial_agent_TTCs = np.vstack((target_1st_order_TTC_list, target_actual_mean_TTC_list, agent_TTC_list, f_speed_idx_list))
    TTC_list.append(trial_agent_TTCs)

total_fail_cases = target_front_count + subject_front_count
if total_fail_cases != 0:
    print("target_passes_first: {:.2f}%".format(100 * target_front_count/total_fail_cases))
    print("subject_passes_first: {:.2f}%".format(100 * subject_front_count/total_fail_cases))
mean_reward = np.mean(reward_list)
print("mean rewards: {:.2f}".format(mean_reward))
with open("{}test_stats{}.txt".format(model_save_path, agent_name), 'w+') as f:
    f.write("target_passes_first: {:.2f}%\n".format(100 * target_front_count/total_fail_cases))
    f.write("subject_passes_first: {:.2f}%\n".format(100 * subject_front_count/total_fail_cases))
    f.write("mean rewards: {:.2f}\n".format(mean_reward))
print("plotting TTC boxplot...")
plot_TTC_boxplot(model_save_path, TTC_list, plot_fname="TTC_boxplot_InstOnly_noHdst_noBonus_pedal1.0" + agent_name)
print("all done.")
