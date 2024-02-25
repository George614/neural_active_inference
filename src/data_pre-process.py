# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

data_path = "D:/Datasets/LocomotorInterception_2021/"
file_name = "human_interception.csv"
data = pd.read_csv(data_path + file_name)
subjectTTC = data['subject_z'] / data['subject_speed']
subjectTTC = subjectTTC.replace(np.inf, np.nan)
subject_dis = data['subject_z']
subjec_speed = data['subject_speed']
targetDis = np.sqrt(data['target_x']**2 + data['target_z']**2)
target_speed = data['target_speed']
targetTTC = targetDis / data['target_speed']
deltaTTC = subjectTTC - targetTTC
estimated_speed = data["subject_z"] / targetTTC
speed_diff = data["subject_speed"] - estimated_speed
done = data['num_frames'] == data['frame']
diff_2_steps = np.stack([speed_diff[:-1].values, speed_diff[1:].values], axis=-1)
all4 = pd.concat([targetDis, target_speed, subject_dis, subjec_speed], axis=1)

done_t = done[:-1]
done_tp1 = done[1:]
mask_done = np.logical_or(done_t.values, done_tp1.values)
mask_not_done = np.logical_not(mask_done)
mask_success = data['success'] == True
mask_success = mask_success[:-1]
speed_diff_2_steps = diff_2_steps[np.logical_and(mask_not_done, mask_success.values)]
all8 = np.hstack([all4.values[:-1, :], all4.values[1:, :]])
all8_masked = all8[np.logical_and(mask_not_done, mask_success.values)]

mask1 = np.logical_and(speed_diff_2_steps[:, 0] > -15, speed_diff_2_steps[:, 0] < 15)
speed_diff_2_steps = speed_diff_2_steps[mask1]
np.save(data_path + "human_interception.npy", speed_diff_2_steps)
np.save(data_path + "human_intercept_prior_4d.npy", all8_masked)