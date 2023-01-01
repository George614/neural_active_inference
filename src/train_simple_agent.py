import os
import logging
import sys, getopt, optparse
import pickle
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import numpy as np
import gym
sys.path.insert(0, 'utils/')
from collections import deque
from utils import parse_int_list, save_object, load_object
from config import Config
sys.path.insert(0, 'model/')
from simple_qai_model import QAIModel
from interception_py_env import InterceptionEnv
from buffers import ReplayBuffer
from scheduler import Linear_schedule

"""
Simulates the training of an active inference (AI) agent implemented by simple
artificial neural networks trainable by backpropagation of errors (backprop).

The particular agent this code trains is called the "QAIModel" which is a
variant of active inference that just focuses on learning a transition model
and expected free energy (EFE) network jointly using simple Q-learning.
Note that this agent uses the dynamic scalar normalization proposed in
Ororbia & Mali (2021) "Adapting to Dynamic Environments with Active Neural Generative Coding".

@author Alexander G. Ororbia, Zhizhuo (George) Yang
"""

def calc_window_mean(window):
    """
        Calculates the mean/average over a finite window of values
    """
    mu = sum(window) * 1.0 / len(window)
    return mu

def create_optimizer(opt_type, eta, momentum=0.9, epsilon=1e-08):
    """
        Inits an update rule for the AI agent.
    """
    print(" Opt.type = {0}  eta = {1}  eps = {2}".format(opt_type, eta, epsilon))
    moment_v = tf.Variable( momentum )
    eta_v  = tf.Variable( eta )
    if opt_type == "nag":
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=eta_v,momentum=moment_v,use_nesterov=True)
    elif opt_type == "momentum":
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=eta_v,momentum=moment_v,use_nesterov=False)
    elif opt_type == "adam":
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=eta_v,beta1=0.9, beta2=0.999) #1e-08)
    elif opt_type == "rmsprop":
        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=eta_v,decay=0.9, momentum=moment_v, epsilon=epsilon) #epsilon=1e-10)
    else:
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=eta_v)
    return optimizer

################################################################################
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["cfg_fname=","gpu_id="])
# Collect arguments from argv
cfg_fname = "run_interception_ai.cfg"
use_gpu = True
# gpu_id = 0
for opt, arg in options:
    if opt in ("--cfg_fname"):
        cfg_fname = arg.strip()
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())
        use_gpu = True
args = Config(cfg_fname)
# GPU arguments
mid = gpu_id
if use_gpu:
    print(" > Using GPU ID {0}".format(mid))
    os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(mid)
    gpu_tag = '/GPU:0'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'
################################################################################

n_trials = int(args.getArg("n_trials"))
out_dir = args.getArg("out_dir") #"/home/agoroot/IdeaProjects/playful_learning/exp/mcar/act_inf1/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
shutil.copy(cfg_fname, out_dir)

num_frames = 240000  # total number of training steps
num_episodes = int(args.getArg("num_episodes")) #500 # 2000  # total number of training episodes

target_update_freq = int(args.getArg("target_update_step"))  # in terms of steps
target_update_ep = int(args.getArg("target_update_ep")) #2  # in terms of episodes
buffer_size = int(args.getArg("buffer_size")) #200000 # 100000
learning_start = int(args.getArg("learning_start"))
prob_alpha = 0.6
batch_size = int(args.getArg("batch_size")) #256
dim_a = int(args.getArg("dim_a"))
dim_o = int(args.getArg("dim_o"))
grad_norm_clip = float(args.getArg("grad_norm_clip")) #1.0 #10.0
clip_type = args.getArg("clip_type")
log_interval = 4
epsilon_greedy = args.getArg("epsilon_greedy").strip().lower() == 'true'
epistemic_off = args.getArg("epistemic_off").strip().lower() == 'true'
keep_expert_batch = args.getArg("keep_expert_batch").strip().lower() == 'true'
use_per_buffer = args.getArg("use_per_buffer").strip().lower() == 'true'
equal_replay_batches = args.getArg("equal_replay_batches").strip().lower() == 'true'
vae_reg = False
epistemic_anneal = args.getArg("epistemic_anneal").strip().lower() == 'true'
perfect_prior = False
use_env_prior = False if args.getArg("env_prior").strip().lower() == 'none' else True
if use_env_prior:
    env_prior = args.getArg("env_prior")
else:
    env_prior = None
record_stats = True
record_interval = 25
record_video = args.getArg("record_video").strip().lower() == 'true'
seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()
# epsilon exponential decay schedule
epsilon_start = float(args.getArg("epsilon_start"))
epsilon_final = float(args.getArg("epsilon_final"))
# epsilon_decay = num_frames / 20
epsilon_by_frame = Linear_schedule(epsilon_start, epsilon_final, num_frames * 0.2) # linear schedule gives more exploration
# training frenquency, update model weights after collecting every n transitions from env
train_freq = int(args.getArg("train_freq"))
# apply n gradient steps in each training cycle
gradient_steps = int(args.getArg("gradient_steps"))

### initialize optimizer and environment ###
opt_type = args.getArg("optimizer").strip().lower()
lr  = tf.Variable(float(args.getArg("learning_rate")))
opt = create_optimizer(opt_type, eta=lr, epsilon=1e-5)

if args.getArg("env_name") == "InterceptionEnv":
    f_speed_idx = int(args.getArg("f_speed_idx"))
    env = InterceptionEnv(target_speed_idx=f_speed_idx, approach_angle_idx=3, return_prior=env_prior, use_slope=False, perfect_prior=perfect_prior)
else:
    env = gym.make(args.getArg("env_name"))
# set seeds
tf.random.set_seed(seed)
np.random.seed(seed)
env.seed(seed=seed)
args.variables['seed'] = seed

# get dimensions of the rendering if needed
if record_video:
    import cv2
    _ = env.reset()
    env.step(0)
    img = env.render(mode='rgb_array')
    env.close()
    width = img.shape[1]
    height = img.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # 'XVID' with '.avi' or 'mp4v' with '.mp4' suffix
    FPS = 30
################################################################################
all_win_mean = []

if args.hasArg("start_trial"):
    start_trial = int(args.getArg("start_trial"))
else:
    start_trial = 0
for trial in range(start_trial, n_trials):
    print(" >> Setting up experience replay buffers...")
    replay_buffer = ReplayBuffer(buffer_size, seed=seed)
    ################################################################################
    print(" >> Setting up prior and posterior models...")
    # initial our model using parameters in the config file
    pplModel = QAIModel(args=args)

    global_reward = []
    reward_window = deque(maxlen=100)
    trial_win_mean = []
    mean_ep_reward = []
    std_ep_reward = []
    if record_stats:
        target_1st_order_TTC_list = []
        target_actual_mean_TTC_list = []
        agent_TTC_list = []
        EFE_values_trial_list = []
        TTC_diff_list = []
        f_speed_idx_list = []
        target_front_count = 0
        subject_front_count = 0

    frame_idx = 0
    crash = False
    loss_efe = None

    print(" >> Starting simulation...")
    for ep_idx in range(num_episodes):  # training using episode as cycle
        ### training the PPL model ###
        if args.getArg("env_name") == "InterceptionEnv":
            f_speed_idx = np.random.randint(3)
            env = InterceptionEnv(target_speed_idx=f_speed_idx, approach_angle_idx=3, return_prior=env_prior, use_slope=False, perfect_prior=perfect_prior)
        observation = env.reset()
        init_condition = tf.expand_dims(observation[:2], axis=0)
        
        done = False
        if record_stats and ep_idx % record_interval == 0:
            TTC_calculated = False
            efe_list = []
            if record_video:
                video = cv2.VideoWriter(out_dir+"trial_{}_epd_{}_tsidx_{}.avi".format(trial, ep_idx, f_speed_idx), fourcc, float(FPS), (width, height))

        episode_reward = 0
        while not done:
            frame_idx += 1
            if epsilon_greedy:
                epsilon = epsilon_by_frame(frame_idx)
            else:
                epsilon = 0.0
            pplModel.epsilon.assign(epsilon)

            obv = tf.convert_to_tensor(observation, dtype=tf.float32)
            obv = tf.expand_dims(obv, axis=0)

            ## infer action given current observation ##
            if record_stats and ep_idx % record_interval == 0:
                action, efe_values, isRandom = pplModel.act(obv, return_efe=True)
                efe_values = efe_values.numpy().squeeze()
                efe_list.append(efe_values)
            else:
                action = pplModel.act(obv)
            action = action.numpy().squeeze()

            ## take a step in the environment ## 
            if use_env_prior:
                next_obv, reward, done, obv_prior, info = env.step(action)
            else:
                next_obv, reward, done, info = env.step(action)

            if record_stats and ep_idx % record_interval == 0:
                # calculate and record TTC and write 1 frame to the video
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
                if record_video:
                    img = env.render(mode='rgb_array', isRandom=isRandom)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    video.write(img)
            
            episode_reward += reward

            ## save transition tuple to the replay buffer, then train on batch w/ or w/o schedule ##
            if use_env_prior:
                replay_buffer.push(observation, action, reward, next_obv, done, prior=obv_prior)
            else:
                replay_buffer.push(observation, action, reward, next_obv, done)
            
            observation = next_obv

            batch_data = None
            if len(replay_buffer) > learning_start and frame_idx % train_freq == 0:
                for _ in range(gradient_steps):
                    batch_data = replay_buffer.sample(batch_size)
                    if use_env_prior:
                        [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done, batch_prior] = batch_data
                        batch_action = tf.one_hot(batch_action, depth=dim_a)
                        grads_efe, loss_efe, R_ti, efe_t, efe_target = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, obv_prior=batch_prior, reward=batch_reward)
                    else:
                        [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done] = batch_data
                        batch_action = tf.one_hot(batch_action, depth=dim_a)
                        grads_efe, loss_efe, R_ti, efe_t, efe_target = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, reward=batch_reward)

                    if tf.math.is_nan(loss_efe):
                        print("loss_efe nan at frame #", frame_idx)
                        break

                    ### clip gradients  ###
                    crash = False
                    grads_efe_clipped = []
                    for grad in grads_efe:
                        if grad is not None:
                            if clip_type == "hard_clip":
                                grad = tf.clip_by_value(grad, -grad_norm_clip, grad_norm_clip)
                            else:
                                grad = tf.clip_by_norm(grad, clip_norm=grad_norm_clip)
                            if tf.math.reduce_any(tf.math.is_nan(grad)):
                                print("grad_efe nan at frame # ", frame_idx)
                                crash = True
                        grads_efe_clipped.append(grad)

                    if crash:
                        break

                    ### Gradient descend by Adam optimizer excluding variables with no gradients ###
                    opt.apply_gradients(zip(grads_efe_clipped, pplModel.param_var))

            if frame_idx % target_update_freq == 0:
                pplModel.update_target()
        
        if reward == 0:
            # record who passes the interception point first
            if env.state[0] < env.state[2]:
                target_front_count += 1
            else:
                subject_front_count += 1
            # record TTC difference
            TTC_diff = env.state[2] / env.state[3] - env.state[0] / env.state[1]
            TTC_diff_list.append((ep_idx, TTC_diff))

        env.close()
        ### after each training episode is done ###

        if loss_efe is not None:
            print("-----------------------------------------------------------------")
            print("frame {}, L.efe = {}  eps = {}, rho = {}".format(frame_idx,
                  loss_efe.numpy(), pplModel.epsilon.numpy(), pplModel.rho.numpy()))
        
        global_reward.append(episode_reward)
        reward_window.append(episode_reward)
        reward_window_mean = calc_window_mean(reward_window)
        trial_win_mean.append(reward_window_mean)

        print("episode {}, r.mu = {:.3f}  win.mu = {:.3f}".format(ep_idx+1, episode_reward, reward_window_mean))
        print("-----------------------------------------------------------------")
        if ep_idx % 50 == 0:
            rewards_fname = "{0}trial{1}".format(out_dir, trial)
            print(" => Saving reward sequence to ", rewards_fname)
            np.save("{0}_R".format(rewards_fname), np.array(global_reward))
            agent_fname = "{0}trial_{1}_epd_{2}.agent".format(out_dir, trial, ep_idx)
            save_object(pplModel, fname=agent_fname)

        if record_stats and ep_idx % record_interval == 0:
            EFE_values_trial_list.append(np.asarray(efe_list))
            if record_video:
                video.release()
        
    env.close()
    all_win_mean.append(np.asarray(trial_win_mean))
    agent_fname = "{0}trial{1}".format(out_dir, trial)
    print("==> Saving reward sequence to ", agent_fname)
    np.save("{0}_R".format(agent_fname), np.array(global_reward))
    print("==> Saving QAI model: {0}".format(agent_fname))
    save_object(pplModel, fname="{0}.agent".format(agent_fname))

    if record_stats:
        trial_TTCs = np.vstack((target_1st_order_TTC_list, target_actual_mean_TTC_list, agent_TTC_list, f_speed_idx_list))
        print("==> Saving TTC sequence...")
        np.save("{0}trial_{1}_TTCs.npy".format(out_dir, trial), trial_TTCs)
        print("==> Saving EFE sequence...")
        np.save("{0}trial_{1}_EFE_values.npy".format(out_dir, trial), EFE_values_trial_list)
        print("==> Saving TTC_diff sequence...")
        np.save("{0}trial_{1}_TTC_diffs.npy".format(out_dir, trial), np.asarray(TTC_diff_list))
        with open("{}who_passes_first.txt".format(out_dir), 'a+') as f:
            total_fail_cases = target_front_count + subject_front_count
            f.write("trial_{}_target_passes_first: {:.2f}%\n".format(trial, 100 * target_front_count/total_fail_cases))
            f.write("trial_{}_subject_passes_first: {:.2f}%\n".format(trial, 100 * subject_front_count/total_fail_cases))

### plotting for window-averaged rewards ###
plot_rewards = args.getArg("plot_rewards").strip().lower() == 'true'
if plot_rewards:
    from pathlib import Path
    import matplotlib.pyplot as plt
    result_dir = Path(out_dir)
    reward_list = []
    for i, stuff in enumerate(list(result_dir.glob("*R.npy"))):
        rewards = np.load(str(stuff))
        reward_list.append(rewards)
        fig = plt.figure()
        line = plt.plot(np.arange(len(rewards)), rewards, linewidth=0.5)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        fig.savefig(str(result_dir)+"/trial_{}.png".format(i), dpi=200)
        plt.close(fig)
    reward_list = np.asarray(reward_list)
    mean_rewards = np.mean(reward_list, axis=0)
    std_rewards = np.std(reward_list, axis=0)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=0.7, color='red', label='mean', linewidth=0.5)
    ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='#888888', alpha=0.4)
    ax.legend(loc='upper right')
    ax.set_ylabel("Rewards")
    ax.set_xlabel("Number of episodes")
    ax.set_title("Episode rewards")
    fig.savefig(str(result_dir) + "/mean_rewards.png", dpi=200)
    plt.close(fig)
    
    for tr in range(len(all_win_mean)):
        fig = plt.figure()
        line = plt.plot(np.arange(len(all_win_mean[tr])), all_win_mean[tr], linewidth=0.5)
        plt.xlabel("Episodes")
        plt.ylabel("Window-averaged Rewards")
        fig.savefig(str(result_dir)+"/trial_{}_win_avg.png".format(tr), dpi=200)
        plt.close(fig)
    all_win_mean = np.stack(all_win_mean)
    mean_rewards = np.mean(all_win_mean, axis=0)
    std_rewards = np.std(all_win_mean, axis=0)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(mean_rewards)), mean_rewards, alpha=1.0, color='red', label='mean', linewidth=0.5)
    ax.fill_between(np.arange(len(mean_rewards)), np.clip(mean_rewards - std_rewards, 0, 1), np.clip(mean_rewards + std_rewards, 0, 1), color='pink', alpha=0.4)
    ax.legend(loc='upper right')
    ax.set_ylabel("Rewards")
    ax.set_xlabel("Number of episodes")
    ax.set_title("Window-averaged rewards")
    fig.savefig(str(result_dir) + "/mean_win_rewards.png", dpi=200)
    plt.close(fig)