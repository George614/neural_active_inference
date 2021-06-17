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

model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/negRti_pred_obv_anneal_Rte_mse_4D_obv_R35/"
create_video = True
n_episodes = 1
target_speed_idx = 2
approach_angle_idx = 3
use_env_prior = False
scale_obv = True
env = InterceptionEnv(target_speed_idx, approach_angle_idx, return_prior=use_env_prior)

print("Interception environment with target_speed_idx {} and approach_angle_idx {}".format(target_speed_idx, approach_angle_idx))
frame_duration = 1 / env.FPS

trial_num = 2
episode_num = 50
qaiModel = load_object(model_save_path + "trial_{}_epd_{}.agent".format(trial_num, episode_num))
print("Loaded QAI model from {}".format(model_save_path))

if create_video:
    _ = env.reset()
    env.step(0)
    img = env.render(mode='rgb_array')
    width = img.shape[1]
    height = img.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # 'XVID' with '.avi' or 'mp4v' with '.mp4' suffix

for i in range(n_episodes):
    observation = env.reset()
    if scale_obv:
        # scale obv to range (0, 1)
        observation[0] /= env.target_init_distance
        observation[1] /= env.target_max_speed
        observation[2] /= env.subject_max_position
        observation[3] /= env.subject_speed_max
    if create_video:
        video = cv2.VideoWriter(model_save_path+"trial_{}_epd_{}.avi".format(trial_num, episode_num), fourcc, float(env.FPS), (width, height))
    prev_time = time.time()
    done = False
    while not done:
        obv = tf.convert_to_tensor(observation, dtype=tf.float32)
        obv = tf.expand_dims(obv, axis=0)
        action = qaiModel.act(obv)
        action = action.numpy().squeeze()
        if use_env_prior:
            next_obv, reward, done, prior, _ = env.step(action)
        else:
            next_obv, reward, done, _ = env.step(action)
            if scale_obv:
                # scale obv to range (0, 1)
                next_obv[0] /= env.target_init_distance
                next_obv[1] /= env.target_max_speed
                next_obv[2] /= env.subject_max_position
                next_obv[3] /= env.subject_speed_max
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

env.close()
print("Test simulation finished.")
