import os
import logging
import sys
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import numpy as np
import time
sys.path.insert(0, 'utils/')
from utils import load_object
sys.path.insert(0, 'model/')
from interception_py_env import InterceptionEnv


model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/correct_Rti_mse_normalize_obv/"
n_episodes = 10
target_speed_idx = 2
approach_angle_idx = 0
env = InterceptionEnv(target_speed_idx, approach_angle_idx)
print("Interception environment with target_speed_idx {} and approach_angle_idx {}".format(target_speed_idx, approach_angle_idx))
frame_duration = 1 / env.FPS

qaiModel = load_object(model_save_path + "qai.agent")
print("Loaded QAI model from {}".format(model_save_path))

for _ in range(n_episodes):
	observation = env.reset()
	env.render()
	prev_time = time.time()
	done = False
	while not done:
	    obv = tf.convert_to_tensor(observation, dtype=tf.float32)
	    obv = tf.expand_dims(obv, axis=0)
	    action = qaiModel.act(obv)
	    action = action.numpy().squeeze()
	    next_obv, reward, done, _ = env.step(action)
	    observation = next_obv    
	    time.sleep(max(frame_duration - (time.time() - prev_time), 0))
	    prev_time = time.time()
	    env.render()

env.close()
print("Test simulation finished.")
