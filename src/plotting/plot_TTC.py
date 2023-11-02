# -*- coding: utf-8 -*-
"""
Read and plot the results of Time-To-Contacts for different target initial 
speeds for the interception task.

@author: Zhizhuo (George) Yang
"""
import os
import numpy as np
import matplotlib.pyplot as plt

results_dir = "D:/Projects/neural_active_inference/src/"


def read_results(subject_type, target_speed_idx):
    with open(results_dir+'TTC_{}_fspeed_idx_{}.txt'.format(subject_type, target_speed_idx), 'r') as f:
        all_data = f.readlines()
    target_1st_order_TTC = all_data[2].split(' ')[:-1]
    target_actual_mean_TTC = all_data[4].split(' ')[:-1]
    agent_TTC = all_data[6].split(' ')[:-1]
    target_1st_order_TTC = [float(ttc) for ttc in target_1st_order_TTC]
    target_actual_mean_TTC = [float(ttc) for ttc in target_actual_mean_TTC]
    agent_TTC = [float(ttc) for ttc in agent_TTC]
    mean_target_1st_order_TTC = np.mean(target_1st_order_TTC)
    mean_target_actual_mean_TTC = np.mean(target_actual_mean_TTC)
    mean_agent_TTC = np.mean(agent_TTC)
    return mean_target_1st_order_TTC, mean_target_actual_mean_TTC, mean_agent_TTC


prior_0_target_1st, prior_0_target_act, prior_0_agent = read_results(
    'prior_function', 0)
prior_1_target_1st, prior_1_target_act, prior_1_agent = read_results(
    'prior_function', 1)
prior_2_target_1st, prior_2_target_act, prior_2_agent = read_results(
    'prior_function', 2)
agent_0_target_1st, agent_0_target_act, agent_0_agent = read_results(
    'agent', 0)
agent_1_target_1st, agent_1_target_act, agent_1_agent = read_results(
    'agent', 1)
agent_2_target_1st, agent_2_target_act, agent_2_agent = read_results(
    'agent', 2)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('TTC for different initial target speeds')
fig.supxlabel('Inital target speeds', y=-0.02)
ax1.set_xlim(0.0, 3.5)
ax1.set_ylim(0.0, 3.0)
ax2.set_xlim(0.0, 3.5)
ax2.set_ylim(0.0, 3.0)
ax1.scatter([1, 2, 3], [prior_2_target_1st, prior_1_target_1st,
                        prior_0_target_1st], marker="*", label='target_1st_order')
ax1.scatter([1, 2, 3], [prior_2_target_act, prior_1_target_act,
                        prior_0_target_act], marker="_", label='target_actual_mean')
ax1.scatter([1, 2, 3], [prior_2_agent, prior_1_agent,
                        prior_0_agent], marker=".", label='subject mean')
ax1.set_ylabel('Time (s)')
ax1.set_xticklabels(['', '8.18', '9.47', '11.25'])
ax1.set_xlabel('Prior function')
ax2.scatter([1, 2, 3], [agent_2_target_1st, agent_1_target_1st,
                        agent_0_target_1st], marker="*")
ax2.scatter([1, 2, 3], [agent_2_target_act, agent_1_target_act,
                        agent_0_target_act], marker="_")
ax2.scatter([1, 2, 3], [agent_2_agent, agent_1_agent,
                        agent_0_agent], marker=".")
ax2.set_xticklabels(['', '8.18', '9.47', '11.25'])
ax2.set_xlabel('AIF agent')
ax1.legend(loc='lower left', fontsize='small',
           ncol=1, mode='expand', borderaxespad=0.)
fig.savefig(os.getcwd() + "/TTC_compare.png", dpi=200, bbox_inches="tight")
