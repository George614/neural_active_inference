import os
import logging
import random
import sys, getopt, optparse
import pickle
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import numpy as np
from gym_unity.envs import UnityToGymWrapper
from animalai.envs.environment import AnimalAIEnvironment
sys.path.insert(0, 'utils/')
from collections import deque
from utils import parse_int_list, save_object, load_object
from config import Config
sys.path.insert(0, 'model/')
from qai_model import QAIModel
from buffers import ReplayBuffer, NaivePrioritizedBuffer
from scheduler import Linear_schedule, Exponential_schedule

"""
Simulates the training of an active inference (AI) agent implemented by simple
artificial neural networks trainable by backpropagation of errors (backprop).

Author: Zhizhuo (George) Yang
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

def create_env_single_config(configuration_file):

    aai_env = AnimalAIEnvironment(
        seed = 123,
        file_name="D:/Projects/Animal_AI/animal-ai-main/env/AnimalAI",
        arenas_configurations=configuration_file,
        play=False,
        base_port=5000,
        inference=False,
        useCamera=True,
        resolution=36,
        useRayCasts=True,
        raysPerSide=2,
        rayMaxDegrees=60,
    )

    # env = UnityToGymWrapper(aai_env, uint8_visual=False, allow_multiple_obs=True, flatten_branched=False)
    # def make_env():
    #     def _thunk():
    #         env = UnityToGymWrapper(aai_env, uint8_visual=False, allow_multiple_obs=True, flatten_branched=True)
    #         return env
    #     return _thunk
    # env = DummyVecEnv([make_env()])
    gym_env = UnityToGymWrapper(aai_env, uint8_visual=False, allow_multiple_obs=True, flatten_branched=True)
    return gym_env
################################################################################
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["cfg_fname=","gpu_id="])
# Collect arguments from argv
cfg_fname = "run_animal_ai.cfg"
use_gpu = True
gpu_id = 0
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
out_dir = args.getArg("out_dir")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
shutil.copy(cfg_fname, out_dir)
eval_model = (args.getArg("eval_model").strip().lower() == 'true')
num_frames = 240000  # total number of training steps
num_episodes = int(args.getArg("num_episodes")) #500 # 2000  # total number of training episodes
test_episodes = 5  # number of episodes for testing
target_update_freq = int(args.getArg("target_update_step"))  # in terms of steps
target_update_ep = int(args.getArg("target_update_ep")) #2  # in terms of episodes
buffer_size = int(args.getArg("buffer_size")) #200000 # 100000
learning_start = int(args.getArg("learning_start"))
prob_alpha = 0.6
batch_size = int(args.getArg("batch_size"))
dim_a = int(args.getArg("dim_a"))
dim_o = int(args.getArg("dim_o"))
grad_norm_clip = float(args.getArg("grad_norm_clip"))
clip_type = args.getArg("clip_type")
log_interval = 4
epsilon_greedy = args.getArg("epsilon_greedy").strip().lower() == 'true'
epistemic_off = args.getArg("epistemic_off").strip().lower() == 'true'
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
# epsilon_by_frame = Exponential_schedule(epsilon_start, epsilon_final, epsilon_decay)
epsilon_by_frame = Linear_schedule(epsilon_start, epsilon_final, num_frames * 0.2) # linear schedule gives more exploration
# rho linear schedule for annealing epistemic term
anneal_start_reward = 0.5
rho_start = 1.0
rho_final = 0.1
rho_ep_duration = 100
rho_by_episode = Linear_schedule(rho_start, rho_final, rho_ep_duration)
# beta linear schedule for prioritized experience replay
beta_start = 0.4
beta_final = 1.0
beta_ep_duration = 600
beta_by_episode = Linear_schedule(beta_start, beta_final, beta_ep_duration)
# training frenquency, update model weights after collecting every n transitions from env
train_freq = int(args.getArg("train_freq"))
# apply n gradient steps in each training cycle
gradient_steps = int(args.getArg("gradient_steps"))

### initialize optimizer and environment ###
opt_type = args.getArg("optimizer").strip().lower()
lr  = tf.Variable(float(args.getArg("learning_rate")))
learning_rate_decay = float(args.getArg("learning_rate_decay"))
opt = create_optimizer(opt_type, eta=lr, epsilon=1e-5)

### initialize environment ###
if len(sys.argv) > 1:
    configuration_file = sys.argv[1]
else:   
    competition_folder = "../configs/competition/"
    configuration_files = os.listdir(competition_folder)
    configuration_random = random.randint(0, len(configuration_files))
    configuration_file = (
        competition_folder + configuration_files[configuration_random]
    )
env = create_env_single_config(configuration_file=configuration_file)
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
    if use_per_buffer:
        per_buffer = NaivePrioritizedBuffer(buffer_size, prob_alpha=prob_alpha)
    else:
        expert_buffer = ReplayBuffer(buffer_size, seed=seed)
        replay_buffer = ReplayBuffer(buffer_size, seed=seed)
    ################################################################################

    print(" >> Setting up prior and posterior models...")
    # initial our model using parameters in the config file
    pplModel = QAIModel(None, args=args)
    if epistemic_off:
        # turn off epistemic term
        pplModel.rho.assign(0.0)

    global_reward = []
    reward_window = deque(maxlen=100)
    trial_win_mean = []
    mean_ep_reward = []
    std_ep_reward = []
    frame_idx = 0
    crash = False
    rho_anneal_start = False
    loss_efe = None

    print(" >> Starting simulation...")

    # for frame_idx in range(1, num_frames + 1): # deprecated
    for ep_idx in range(num_episodes):  # training using episode as cycle
        ### training the PPL model ###
        obv_visual, obv_raycast = env.reset()
        done = False
        if record_video and ep_idx % record_interval == 0:
            video = cv2.VideoWriter(out_dir+"trial_{}_epd_{}.avi".format(trial, ep_idx), fourcc, float(FPS), (width, height))
        # linear schedule for VAE model regularization
        if vae_reg:
            gamma = gamma_by_episode(ep_idx)
            pplModel.gamma.assign(gamma)
        if use_per_buffer:
            beta = beta_by_episode(ep_idx)
        episode_reward = 0
        while not done:
            frame_idx += 1
            if epsilon_greedy:
                epsilon = epsilon_by_frame(frame_idx)
            else:
                epsilon = 0.0
            pplModel.epsilon.assign(epsilon)

            obv = tf.convert_to_tensor(obv_raycast, dtype=tf.float32)
            obv = tf.expand_dims(obv, axis=0)

            ## infer action given current observation ##
            action, _, _ = pplModel.act(obv)
            action = int(action.numpy().squeeze())

            ## take a step in the environment ## 
            next_obv, reward, done, info = env.step(action)
            next_visual_o, next_vec_o = next_obv

            ## infer epistemic signal using generative model ##
            obv_tp1 = tf.convert_to_tensor(next_vec_o, dtype=tf.float32)
            obv_tp1 = tf.expand_dims(obv_tp1, axis=0)
            a_t = tf.expand_dims(action, axis=0)
            a_t = tf.one_hot(a_t, depth=dim_a)
            R_te = pplModel.infer_epistemic(obv, obv_tp1, action=a_t)
            R_te = R_te.numpy().squeeze()

            if record_video and ep_idx % record_interval == 0:
                # calculate and record TTC and write 1 frame to the video
                img = env.render(mode='rgb_array', offset=offset, isRandom=isRandom)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)
            
            episode_reward += reward

            efe_N = 1.0
            ## save transition tuple to the replay buffer, then train on batch w/ or w/o schedule ##
            if use_per_buffer is True:
                if use_env_prior:
                    per_buffer.push(observation, action, reward, next_obv, done, R_te, obv_prior)
                else:
                    per_buffer.push(observation, action, reward, next_obv, done, R_te)
                observation = next_obv
                batch_data = None
                if len(per_buffer) > learning_start:
                    batch_data = per_buffer.sample(batch_size, beta=beta)
                efe_N = batch_size * 1.0
                if batch_data is not None:
                    if use_env_prior:
                        [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done, batch_R_te, batch_prior, batch_indices, batch_weights] = batch_data
                    else:
                        [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done, batch_R_te, batch_indices, batch_weights] = batch_data
                    batch_action = tf.one_hot(batch_action, depth=dim_a)
                    if use_env_prior:
                        grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, efe_t, efe_target, priorities = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, batch_R_te, batch_weights, obv_prior=batch_prior, reward=batch_reward)
                    else:
                        grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, efe_t, efe_target, priorities = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, batch_R_te, batch_weights, reward=batch_reward)
                    per_buffer.update_priorities(batch_indices, priorities.numpy())
            else:
                if use_env_prior:
                    replay_buffer.push(obv_raycast, action, reward, next_vec_o, done, R_te, obv_prior)
                else:
                    replay_buffer.push(obv_raycast, action, reward, next_vec_o, done, R_te)
                
                obv_raycast = next_vec_o

                batch_data = None
                if len(replay_buffer) > learning_start and frame_idx % train_freq == 0:
                    for _ in range(gradient_steps):
                        batch_data = replay_buffer.sample(batch_size)
                        if use_env_prior:
                            [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done, batch_R_te, batch_prior] = batch_data
                            batch_action = tf.one_hot(batch_action, depth=dim_a)
                            grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, efe_t, efe_target = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, batch_R_te, obv_prior=batch_prior, reward=batch_reward)
                        else:
                            [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done, batch_R_te] = batch_data
                            batch_action = tf.one_hot(batch_action, depth=dim_a)
                            grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, efe_t, efe_target = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, batch_R_te, reward=batch_reward)

                        if tf.math.is_nan(loss_efe):
                            print("loss_efe nan at frame #", frame_idx)
                            break

                        ### clip gradients  ###
                        crash = False
                        grads_model_clipped = []
                        for grad in grads_model:
                            if grad is not None:
                                if clip_type == "hard_clip":
                                    grad = tf.clip_by_value(grad, -grad_norm_clip, grad_norm_clip)
                                else:
                                    grad = tf.clip_by_norm(grad, clip_norm=grad_norm_clip)
                                if tf.math.reduce_any(tf.math.is_nan(grad)):
                                    print("grad_model nan at frame # ", frame_idx)
                                    crash = True
                            grads_model_clipped.append(grad)

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
                        opt.apply_gradients(zip(grads_model_clipped, pplModel.param_var))
                        opt.apply_gradients(zip(grads_efe_clipped, pplModel.param_var))
                
            if learning_rate_decay > 0.0:
                lower_bound_lr = 1e-7
                lr.assign(max(lower_bound_lr, float(lr * learning_rate_decay)))

            if frame_idx % target_update_freq == 0:
                pplModel.update_target()

        pplModel.clear_state()
        # env.close()
        ### after each training episode is done ###

        if loss_efe is not None:
            print("-----------------------------------------------------------------")
            print("frame {0}, L.model = {1}, L.efe = {2}  eps = {3}, rho = {4}".format(frame_idx, loss_model.numpy(),
                  (loss_efe/efe_N).numpy(), pplModel.epsilon.numpy(), pplModel.rho.numpy()))
        
        ### evaluate the PPL model using a number of episodes ###
        if eval_model is True:
            #pplModel.rho.assign(0.0)
            pplModel.epsilon.assign(0.0) # use greedy policy when testing
            reward_list = []
            for _ in range(test_episodes):
                env = create_env_single_config(configuration_file=configuration_file)
                visual_obv, vec_obv = env.reset()
                episode_reward = 0
                done_test = False
                while not done_test:
                    obv = tf.convert_to_tensor(vec_obv, dtype=tf.float32)
                    obv = tf.expand_dims(obv, axis=0)
                    action, _, _ = pplModel.act(obv)
                    action = action.numpy().squeeze()
                    observation, reward, done_test, _ = env.step(action)
                    visual_obv, vec_obv = observation
                    episode_reward += reward
                reward_list.append(episode_reward)
            #pplModel.rho.assign(1.0)
            mean_reward = np.mean(reward_list)
            #std_reward = np.std(reward_list)
            mean_ep_reward.append(mean_reward)
            #std_ep_reward.append(std_reward)
            episode_reward = mean_reward
        
        pplModel.clear_state()
        # else --> just use training reward as episode reward (online learning)
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

        # annealing of the epistemic term based on the average test rewards
        if epistemic_anneal:
            if not rho_anneal_start and reward_window_mean > anneal_start_reward:
                start_ep = ep_idx
                rho_anneal_start = True
            if rho_anneal_start:
                rho = rho_by_episode(ep_idx - start_ep)
                pplModel.rho.assign(rho)

        if record_video and ep_idx % record_interval == 0:
            video.release()
        
    env.close()
    all_win_mean.append(np.asarray(trial_win_mean))
    agent_fname = "{0}trial{1}".format(out_dir, trial)
    print("==> Saving reward sequence to ", agent_fname)
    np.save("{0}_R".format(agent_fname), np.array(global_reward))
    print("==> Saving QAI model: {0}".format(agent_fname))
    save_object(pplModel, fname="{0}.agent".format(agent_fname))

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