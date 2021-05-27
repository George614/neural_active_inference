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

model_save_path = "D:/Projects/neural_active_inference/exp/interception/qai/correct_Rti_mse_2/"
n_episodes = 1
target_speed_idx = 2
approach_angle_idx = 0
env = InterceptionEnv(target_speed_idx, approach_angle_idx)

print("Interception environment with target_speed_idx {} and approach_angle_idx {}".format(target_speed_idx, approach_angle_idx))
frame_duration = 1 / env.FPS

qaiModel = load_object(model_save_path + "trial4.agent")
print("Loaded QAI model from {}".format(model_save_path))

width = 1000
height = 407
fourcc = cv2.VideoWriter_fourcc(*'XVID')

for i in range(n_episodes):
	observation = env.reset()
	img = env.render(mode='rgb_array')
	video = cv2.VideoWriter(model_save_path+"test_{}.avi".format(i), fourcc, float(env.FPS), (width, height))
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
	    img = env.render(mode='rgb_array')
	    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	    video.write(img)
	video.release()

env.close()
print("Test simulation finished.")
