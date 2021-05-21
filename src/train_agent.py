import os
import logging
import sys, getopt, optparse
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import numpy as np
import gym
sys.path.insert(0, 'utils/')
from utils import parse_int_list, save_object, load_object
from config import Config
sys.path.insert(0, 'model/')
from qai_model import QAIModel
from interception_py_env import InterceptionEnv
from buffers import ReplayBuffer, NaivePrioritizedBuffer
from scheduler import Linear_schedule, Exponential_schedule

"""
Simulates the training of an active inference (AI) agent implemented by simple
artificial neural networks trainable by backpropagation of errors (backprop).

The particular agent this code trains is called the "QAIModel" which is a
variant of active inference that just focuses on learning a transition model
and expected free energy (EFE) network jointly using simple Q-learning.
Note that this agent uses the dynamic scalar normalization proposed in
Ororbia & Mali (2021) "Adapting to Dynamic Environments with Active Neural Generative Coding".

@author Alexander G. Ororbia
"""

def calc_window_mean(window):
    """
        Calculates the mean/average over a finite window of values
    """
    mu = 0.0
    for i in range(len(window)):
        v_i = window[i]
        mu += v_i
    mu = mu / (len(window) * 1.0)
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
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=eta_v,beta1=0.9, beta2=0.999, epsilon=epsilon) #1e-08)
    elif opt_type == "rmsprop":
        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=eta_v,decay=0.9, momentum=moment_v, epsilon=epsilon) #epsilon=1e-10)
    else:
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=eta_v)
    return optimizer

################################################################################
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["cfg_fname=","gpu_id="])
# Collect arguments from argv
cfg_fname = None
use_gpu = False
gpu_id = -1
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
    tf.config.experimental.set_memory_growth(gpu_devices[mid], True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'
################################################################################

n_trials = int(args.getArg("n_trials"))
out_dir = args.getArg("out_dir") #"/home/agoroot/IdeaProjects/playful_learning/exp/mcar/act_inf1/"
prior_model_save_path = None
if args.hasArg("prior_model_save_path"):
    prior_model_save_path = args.getArg("prior_model_save_path") #"/home/agoroot/IdeaProjects/playful_learning/exp/mcar/prior/prior.agent"
expert_data_path = None
if args.hasArg("expert_data_path"):
    expert_data_path = args.getArg("expert_data_path") #"/home/agoroot/IdeaProjects/playful_learning/exp/mcar/expert/zoo-agent-mcar.npy"
eval_model = (args.getArg("eval_model").strip().lower() == 'true')

num_frames = 200000  # total number of training steps
num_episodes = int(args.getArg("num_episodes")) #500 # 2000  # total number of training episodes
test_episodes = 5  # number of episodes for testing
target_update_freq = 500  # in terms of steps
target_update_ep = int(args.getArg("target_update_ep")) #2  # in terms of episodes
buffer_size = int(args.getArg("buffer_size")) #200000 # 100000
prob_alpha = 0.6
batch_size = int(args.getArg("batch_size")) #256
grad_norm_clip = float(args.getArg("grad_norm_clip")) #1.0 #10.0
clip_type = args.getArg("clip_type")
log_interval = 4
keep_expert_batch = True
if expert_data_path is None:
    keep_expert_batch = False
use_per_buffer = (args.getArg("use_per_buffer").strip().lower() == 'true')
equal_replay_batches = (args.getArg("equal_replay_batches").strip().lower() == 'true')
vae_reg = False
epistemic_anneal = False
seed = 44
# epsilon exponential decay schedule
epsilon_start = float(args.getArg("epsilon_start")) #0.025 #0.9
epsilon_final = 0.02
epsilon_decay = num_frames / 20
epsilon_by_frame = Exponential_schedule(epsilon_start, epsilon_final, epsilon_decay)
# gamma linear schedule for VAE regularization
gamma_start = 0.01
gamma_final = 0.99
gamma_ep_duration = 300
gamma_by_episode = Linear_schedule(gamma_start, gamma_final, gamma_ep_duration)
# rho linear schedule for annealing epistemic term
anneal_start_reward = -180
rho_start = 1.0
rho_final = 0.0
rho_ep_duration = 300
rho_by_episode = Linear_schedule(rho_start, rho_final, rho_ep_duration)
# beta linear schedule for prioritized experience replay
beta_start = 0.4
beta_final = 1.0
beta_ep_duration = 600
beta_by_episode = Linear_schedule(beta_start, beta_final, beta_ep_duration)

### initialize optimizer and buffers ###
opt_type = args.getArg("optimizer").strip().lower()
lr  = tf.Variable( float(args.getArg("learning_rate")) )
learning_rate_decay = float(args.getArg("learning_rate_decay"))
opt = create_optimizer(opt_type, eta=lr, epsilon=1e-5)

if use_per_buffer:
    per_buffer = NaivePrioritizedBuffer(buffer_size * 2, prob_alpha=prob_alpha)
else:
    expert_buffer = ReplayBuffer(buffer_size, seed=seed)
    replay_buffer = ReplayBuffer(buffer_size, seed=seed)

#env = gym.make(args.getArg("env_name"))
env = InterceptionEnv(target_speed_idx=2, approach_angle_idx=0)
# set seeds
tf.random.set_seed(seed)
np.random.seed(seed)
env.seed(seed=seed)

################################################################################
### load and pre-process human expert-batch data ###
all_data = None
if expert_data_path is not None:
    print("RL-zoo expert data path: ", expert_data_path)
    all_data = np.load(expert_data_path, allow_pickle=True)
    idx_done = np.where(all_data[:, 6] == 1)[0]
    idx_done = idx_done - 1  # fix error on next_obv when done in original data
    mask = np.not_equal(all_data[:, 6], 1)
    for idx in idx_done:
        all_data[idx, 6] = 1
    all_data = all_data[mask]

for trial in range(n_trials):
    print(" >> Setting up memory replay buffers...")
    if all_data is not None:
        n_tossed = 0
        for i in range(min(len(all_data), buffer_size)):
            o_t, action, reward, o_tp1, done = all_data[i, :2], all_data[i, 2], all_data[i, 3], all_data[i, 4:6], all_data[i, 6]
            if float(done) < 1.0:
                if use_per_buffer:
                    per_buffer.push(o_t, action, reward, o_tp1, done)
                else:
                    expert_buffer.push(o_t, action, reward, o_tp1, done)
            else:
                n_tossed += 1
        print(" > Threw out {0} invalid samples in expert batch".format(n_tossed))

        if not keep_expert_batch and not use_per_buffer:
            replay_buffer = expert_buffer
    ################################################################################

    print(" >> Setting up prior and posterior models...")
    ### load the prior preference model ###
    priorModel = None
    if prior_model_save_path is not None:
        priorModel = load_object(prior_model_save_path)
    # initial our model using parameters in the config file
    pplModel = QAIModel(priorModel, args=args)

    global_reward = []
    reward_window = []
    mean_ep_reward = []
    std_ep_reward = []
    frame_idx = 0
    crash = False
    rho_anneal_start = False

    print(" >> Starting simulation...")
    observation = env.reset()

    # for frame_idx in range(1, num_frames + 1): # deprecated
    for ep_idx in range(num_episodes):  # training using episode as cycle
        ### training the PPL model ###
        done = False
        ## linear schedule for VAE model regularization
        if vae_reg:
            gamma = gamma_by_episode(ep_idx)
            pplModel.gamma.assign(gamma)
        if use_per_buffer:
            beta = beta_by_episode(ep_idx)
        episode_reward = 0
        while not done:
            frame_idx += 1
            epsilon = epsilon_by_frame(frame_idx)
            pplModel.epsilon.assign(epsilon)

            obv = tf.convert_to_tensor(observation, dtype=tf.float32)
            obv = tf.expand_dims(obv, axis=0)

            action = pplModel.act(obv)
            action = action.numpy().squeeze()

            next_obv, reward, done, _ = env.step(action)
            episode_reward += reward

            efe_N = 1.0
            if use_per_buffer is True:
                per_buffer.push(observation, action, reward, next_obv, done)
                observation = next_obv
                batch_data = per_buffer.sample(batch_size, beta=beta)
                efe_N = batch_size * 1.0
                [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done, batch_indices, batch_weights] = batch_data
                batch_action = tf.one_hot(batch_action, depth=int(args.getArg("dim_a")))
                grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, R_te, efe_t, efe_target, priorities = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, batch_weights, reward=batch_reward)
                per_buffer.update_priorities(batch_indices, priorities.numpy())
            else:
                replay_buffer.push(observation, action, reward, next_obv, done)
                observation = next_obv

                if keep_expert_batch:
                    total_samples = len(expert_buffer) + len(replay_buffer)
                    if total_samples / len(replay_buffer) < batch_size and equal_replay_batches is False:
                        # sample from both expert buffer and replay buffer
                        n_replay_samples = np.floor(batch_size * len(replay_buffer) / total_samples)
                        n_expert_samples = batch_size - n_replay_samples
                        expert_batch = expert_buffer.sample(int(n_expert_samples))
                        replay_batch = replay_buffer.sample(int(n_replay_samples))
                        batch_data = [np.concatenate((expert_sample, replay_sample)) for expert_sample, replay_sample in zip(expert_batch, replay_batch)]
                    elif equal_replay_batches is True and (len(replay_buffer) > (batch_size/2)):
                        n_expert_samples = batch_size // 2
                        n_replay_samples = batch_size // 2
                        expert_batch = expert_buffer.sample(int(n_expert_samples))
                        replay_batch = replay_buffer.sample(int(n_replay_samples))
                        batch_data = [np.concatenate((expert_sample, replay_sample)) for expert_sample, replay_sample in zip(expert_batch, replay_batch)]
                    else:
                        # sample from expert buffer only
                        batch_data = expert_buffer.sample(batch_size)
                else:
                    if len(replay_buffer) > batch_size:
                        batch_data = replay_buffer.sample(batch_size)
                    else:
                        batch_data = replay_buffer.sample(len(replay_buffer))

                [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done] = batch_data

                batch_action = tf.one_hot(batch_action, depth=int(args.getArg("dim_a")))
                grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, R_te, efe_t, efe_target = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, reward=batch_reward)

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

            # if frame_idx % target_update_freq == 0:
            #     pplModel.update_target()

            #if frame_idx % 200 == 0:
            #    print("frame {}, loss_model {:.3f}, loss_efe {:.3f}".format(frame_idx, loss_model.numpy(), loss_efe.numpy()))

        pplModel.clear_state()
        if ep_idx % target_update_ep == 0:
            pplModel.update_target()

        ### after each training episode is done ###
        observation = env.reset()

        print("-----------------------------------------------------------------")
        '''
        print("frame {}, L.model = {:.3f}, L.efe = {:.3f}  eps = {:.3f}".format(frame_idx, loss_model.numpy(),
              (loss_efe/efe_N).numpy(),float(pplModel.epsilon.numpy())))
        '''
        print("frame {0}, L.model = {1}, L.efe = {2}  eps = {3}".format(frame_idx, loss_model.numpy(),
              (loss_efe/efe_N).numpy(),float(pplModel.epsilon.numpy())))
        ### evaluate the PPL model using a number of episodes ###
        if eval_model is True:
            #pplModel.rho.assign(0.0)
            pplModel.epsilon.assign(0.0) # use greedy policy when testing
            reward_list = []
            for _ in range(test_episodes):
                episode_reward = 0
                done_test = False
                while not done_test:
                    obv = tf.convert_to_tensor(observation, dtype=tf.float32)
                    obv = tf.expand_dims(obv, axis=0)
                    action = pplModel.act(obv)
                    action = action.numpy().squeeze()
                    observation, reward, done_test, _ = env.step(action)
                    episode_reward += reward
                observation = env.reset()
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
        if len(reward_window) > 100:
            reward_window.pop(0)
        reward_window_mean = calc_window_mean(reward_window)

        print("episode {}, r.mu = {:.3f}  win.mu = {:.3f}".format(ep_idx+1, episode_reward, reward_window_mean))
        print("-----------------------------------------------------------------")
        if ep_idx % 50 == 0:
            agent_fname = "{0}trial{1}_current".format(out_dir,trial,(ep_idx+1))
            print(" => Saving reward sequence to ",agent_fname)
            np.save("{0}_R".format(agent_fname), np.array(global_reward))

        '''
        # annealing of the epistemic term based on the average test rewards
        if epistemic_anneal:
            if not rho_anneal_start and mean_reward > anneal_start_reward:
                start_ep = ep_idx
                rho_anneal_start = True
            if rho_anneal_start:
                rho = rho_by_episode(ep_idx - start_ep)
                pplModel.rho.assign(rho)
        '''

    env.close()
    agent_fname = "{0}trial{1}_current".format(out_dir,trial,(ep_idx+1))
    print(" => Saving reward sequence to ",agent_fname)
    np.save("{0}_R".format(agent_fname), np.array(global_reward))
