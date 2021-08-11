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

approach_angle_idx = 3
num_trials = 50
env_prior = 'prior_error' # or prior_obv or None


def load_AIF_agent(trial_num, episode_num, model_save_path):
    qaiModel = load_object(model_save_path + "trial_{}_epd_{}.agent".format(trial_num, episode_num))
    print("Loaded QAI model from {}".format(model_save_path))
    qaiModel.epsilon.assign(0.0)
    return qaiModel


def record_TTC(target_speed_idx, subject_type, qaiModel=None):
    env = InterceptionEnv(target_speed_idx, approach_angle_idx, return_prior=env_prior)
    print("Interception environment with target_speed_idx {} and approach_angle_idx {}".format(target_speed_idx, approach_angle_idx))

    target_1st_order_TTC_list = []
    target_actual_mean_TTC_list = []
    agent_TTC_list = []

    for i in range(num_trials):
        observation = env.reset()
        done = False
        TTC_calculated = False

        while not done:
            if subject_type == 'agent':
                obv = tf.convert_to_tensor(observation, dtype=tf.float32)
                obv = tf.expand_dims(obv, axis=0)
                action = qaiModel.act(obv)
                action = action.numpy().squeeze()
            if env_prior is not None:
                if subject_type == 'agent':
                    next_obv, reward, done, prior, info = env.step(action)
                elif subject_type == 'prior_function':
                    next_obv, reward, done, prior, info = env.step()
                speed_phase = info['speed_phase']
                if speed_phase == 1 and not TTC_calculated:
                    target_1st_order_TTC = env.state[0] / env.target_init_speed
                    target_actual_mean_TTC = info['target_TTC']
                    agent_TTC = env.state[2] / env.state[3]
                    TTC_calculated = True
                    target_1st_order_TTC_list.append(target_1st_order_TTC)
                    target_actual_mean_TTC_list.append(target_actual_mean_TTC)
                    agent_TTC_list.append(agent_TTC)
            else:
                next_obv, reward, done, _ = env.step(action)
            observation = next_obv    

    target_1st_order_TTC_list = [str(round(num, 3)) + " " for num in target_1st_order_TTC_list]
    target_actual_mean_TTC_list = [str(round(num, 3)) + " " for num in target_actual_mean_TTC_list]
    agent_TTC_list = [str(round(num, 3)) + " " for num in agent_TTC_list]

    with open('TTC_{}_fspeed_idx_{}.txt'.format(subject_type, target_speed_idx), 'w') as f:
        f.write("Initial target speed index {}\n".format(target_speed_idx))
        f.write("target_1st_order_TTC\n")
        f.writelines(target_1st_order_TTC_list)
        f.write('\n')
        f.write("target_actual_mean_TTC\n")
        f.writelines(target_actual_mean_TTC_list)
        f.write('\n')
        f.write("agent_TTC\n")
        f.writelines(agent_TTC_list)

# run the prior function on the task with different target initial speeds
record_TTC(0, 'prior_function')
record_TTC(1, 'prior_function')
record_TTC(2, 'prior_function')

# run the AIF agent on the task with different target initial speeds
model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_posRte_mse_4D_obv_w_speedchg_fspeedIdx0_PriorError_DQNhyperP_512net_noL2Reg_relu_learnSche_3k/"
qaiModel = load_AIF_agent(2, 2950, model_save_path)
record_TTC(0, 'agent', qaiModel=qaiModel)
model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_posRte_mse_4D_obv_w_speedchg_fspeedIdx1_PriorError_DQNhyperP_512net_noL2Reg_relu_learnSche_3k/"
qaiModel = load_AIF_agent(2, 2950, model_save_path)
record_TTC(1, 'agent', qaiModel=qaiModel)
model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_posRte_mse_4D_obv_w_speedchg_fspeedIdx2_PriorError_DQNhyperP_512net_noL2Reg_relu_learnSche_3k/"
qaiModel = load_AIF_agent(0, 2950, model_save_path)
record_TTC(2, 'agent', qaiModel=qaiModel)