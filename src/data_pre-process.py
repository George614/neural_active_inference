# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:48:42 2021

@author: Zhizhuo (George) Yang
"""
import numpy as np
import pandas as pd

data_path = "D:/Datasets/LocomotorInterception_2021/"
file_name = "human_interception.csv"
data = pd.read_csv(data_path + file_name)
subjectTTC = data['subject_z'] / data['subject_speed']
subjectTTC = subjectTTC.replace(np.inf, np.nan)
targetDis = np.sqrt(data['target_x']**2 + data['target_z']**2)
targetTTC = targetDis / data['target_speed']
deltaTTC = subjectTTC - targetTTC
estimated_speed = data["subject_z"] / targetTTC
speed_diff = data["subject_speed"] - estimated_speed
done = data['num_frames'] == data['frame']
diff_2_steps = np.stack([speed_diff[:-1].values, speed_diff[1:].values], axis=-1)

done_t = done[:-1]
done_tp1 = done[1:]
mask_done = np.logical_or(done_t.values, done_tp1.values)
mask_not_done = np.logical_not(mask_done)
mask_success = data['success'] == True
mask_success = mask_success[:-1]
speed_diff_2_steps = diff_2_steps[np.logical_and(mask_not_done, mask_success.values)]

mask1 = np.logical_and(speed_diff_2_steps[:, 0] > -15, speed_diff_2_steps[:, 0] < 15)
speed_diff_2_steps = speed_diff_2_steps[mask1]
np.save(data_path + "human_interception.npy", speed_diff_2_steps)