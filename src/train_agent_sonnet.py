import os
import logging
import sys
import getopt
import shutil
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
import numpy as np
import gym

sys.path.insert(0, 'utils/')
from utils import save_object, load_object
from utils.config import Config
from utils.buffers import ReplayBuffer, NaivePrioritizedBuffer, HindsightBuffer
from utils.scheduler import Linear_schedule

sys.path.insert(0, 'model/')
from model.qai_model import QAIModel
from interception_py_env import InterceptionEnv

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
    print(f" Opt.type = {opt_type}  eta = {eta}  eps = {epsilon}")
    moment_v = tf.Variable(momentum)
    eta_v = tf.Variable(eta)
    if opt_type == "nag":
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=eta_v, momentum=moment_v, use_nesterov=True)
    elif opt_type == "momentum":
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=eta_v, momentum=moment_v, use_nesterov=False)
    elif opt_type == "adam":
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=eta_v, beta1=0.9, beta2=0.999)
    elif opt_type == "rmsprop":
        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=eta_v, decay=0.9, momentum=moment_v, epsilon=epsilon)
    else:
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=eta_v)
    return optimizer

def load_config():
    """
    Load and parse configuration from command line arguments.
    """
    options, remainder = getopt.getopt(sys.argv[1:], '', ["cfg_fname=", "gpu_id="])

    # Set default values
    cfg_fname = "run_interception_ai.cfg"
    use_gpu = True
    gpu_id = 0

    # Parse command line arguments
    for opt, arg in options:
        if opt == "--cfg_fname":
            cfg_fname = arg.strip()
        elif opt == "--gpu_id":
            gpu_id = int(arg.strip())
            use_gpu = True

    args = Config(cfg_fname)

    # Configure GPU usage
    if use_gpu:
        print(f" > Using GPU ID {gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        gpu_tag = '/GPU:0'
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        gpu_tag = '/CPU:0'

    return args

def load_expert_data(expert_data_path):
    """
    Load and pre-process expert data.
    """
    if expert_data_path is not None:
        print("RL-zoo expert data path: ", expert_data_path)
        all_data = np.load(expert_data_path, allow_pickle=True)
        return all_data
    return None

def setup_replay_buffers(buffer_size, h_buffer_size, use_per_buffer, prob_alpha, seed, all_data):
    """
    Set up experience replay buffers.
    """
    print(" >> Setting up experience replay buffers...")
    if use_per_buffer:
        per_buffer = NaivePrioritizedBuffer(buffer_size, prob_alpha=prob_alpha)
    else:
        expert_buffer = ReplayBuffer(buffer_size, seed=seed)
        replay_buffer = ReplayBuffer(buffer_size, seed=seed)

    hindsight_buffer = HindsightBuffer(h_buffer_size, seed=seed)

    if all_data is not None:
        for i in range(min(len(all_data), buffer_size)):
            o_t, action, reward, o_tp1, done = all_data[i, :dim_o], all_data[i, dim_o], all_data[i, dim_o+1], all_data[i, dim_o+2:-1], all_data[i, -1]
            if use_per_buffer:
                per_buffer.push(o_t, action, reward, o_tp1, done)
            else:
                expert_buffer.push(o_t, action, reward, o_tp1, done)

        if not keep_expert_batch and not use_per_buffer:
            replay_buffer = expert_buffer

    if use_per_buffer:
        return per_buffer, hindsight_buffer
    else:
        return expert_buffer, replay_buffer, hindsight_buffer

def setup_models(args, priorModel, prior_model_save_path, epistemic_off):
    """
    Set up prior and posterior models.
    """
    print(" >> Setting up prior and posterior models...")
    if prior_model_save_path is not None:
        priorModel = load_object(prior_model_save_path)

    pplModel = QAIModel(priorModel, args=args)

    if epistemic_off:
        pplModel.rho.assign(0.0)  # Turn off epistemic term

    return pplModel

def setup_schedules(num_frames, epsilon_start, epsilon_final):
    """
    Set up epsilon, gamma, and rho schedules.
    """
    # Epsilon exponential decay schedule
    epsilon_by_frame = Linear_schedule(epsilon_start, epsilon_final, num_frames * 0.2)  # Linear schedule for more exploration

    # Gamma linear schedule for VAE regularization
    gamma_start = 0.01
    gamma_final = 0.99
    gamma_ep_duration = 300
    gamma_by_episode = Linear_schedule(gamma_start, gamma_final, gamma_ep_duration)

    # Rho linear schedule for annealing epistemic term
    anneal_start_reward = 0.5
    rho_start = 1.0
    rho_final = 0.1
    rho_ep_duration = 100
    rho_by_episode = Linear_schedule(rho_start, rho_final, rho_ep_duration)

    # Beta linear schedule for prioritized experience replay
    beta_start = 0.4
    beta_final = 1.0
    beta_ep_duration = 600
    beta_by_episode = Linear_schedule(beta_start, beta_final, beta_ep_duration)

    return epsilon_by_frame, gamma_by_episode, rho_by_episode, beta_by_episode

def train_agent(args, pplModel, env, per_buffer, expert_buffer, replay_buffer, hindsight_buffer, all_data, epsilon_by_frame, gamma_by_episode, rho_by_episode, beta_by_episode):
    """
    Main training loop for the active inference agent.
    """
    global_reward = []
    reward_window = deque(maxlen=100)
    trial_win_mean = []
    mean_ep_reward = []
    std_ep_reward = []
    record_stats = args.getArg("record_stats").strip().lower() == 'true'
    if record_stats:
        target_1st_order_TTC_list = []
        target_actual_mean_TTC_list = []
        agent_TTC_list = []
        EFE_values_trial_list = []
        hindsight_error_list = []
        TTC_diff_list = []
        failed_offset_list = []
        offset_list = []
        H_TTC_diff_list = []
        hdst_hat_list = []
        alpha_list = []
        beta_list = []
        f_speed_idx_list = []
        target_front_count = 0
        subject_front_count = 0

    frame_idx = 0
    crash = False
    rho_anneal_start = False
    loss_efe = None

    print(" >> Starting simulation...")

    for ep_idx in range(args.getInt("num_episodes")):
        if args.getArg("env_name") == "InterceptionEnv":
            f_speed_idx = np.random.randint(3)
            env = InterceptionEnv(target_speed_idx=f_speed_idx, approach_angle_idx=3, return_prior=env_prior, use_slope=False, perfect_prior=perfect_prior)
        observation = env.reset()
        if action_delay:
            action_buffer = deque()
            obv_buffer = deque()
            ep_frame_idx = -1
            obv_buffer.append(observation)
        done = False
        if record_stats and ep_idx % args.getInt("record_interval") == 0:
            TTC_calculated = False
            efe_list = []
            offset_dynamic_list = []
            H_dynamic_list = []
            H_hat_dynamic_list = []
            if args.getArg("record_video").strip().lower() == 'true':
                video = setup_video_writer(out_dir, width, height, fourcc, FPS, ep_idx, f_speed_idx)

        # Linear schedule for VAE model regularization
        if args.getArg("vae_reg").strip().lower() == 'true':
            gamma = gamma_by_episode(ep_idx)
            pplModel.gamma.assign(gamma)

        if args.getArg("use_per_buffer").strip().lower() == 'true':
            beta = beta_by_episode(ep_idx)

        episode_reward = 0
        while not done:
            frame_idx += 1
            if action_delay:
                ep_frame_idx += 1

            if args.getArg("epsilon_greedy").strip().lower() == 'true':
                epsilon = epsilon_by_frame(frame_idx)
            else:
                epsilon = 0.0

            pplModel.epsilon.assign(epsilon)

            obv = tf.convert_to_tensor(observation, dtype=tf.float32)
            obv = tf.expand_dims(obv, axis=0)

            ## infer action given current observation ##
            if record_stats and ep_idx % args.getInt("record_interval") == 0:
                action, efe_values, isRandom = pplModel.act(obv)
                efe_values = efe_values.numpy().squeeze()
                efe_list.append(efe_values)
            else:
                action, _, _ = pplModel.act(obv)
            action = action.numpy().squeeze()

            ## if use predictive component to learn from hindsight error ##
            if args.getArg("hindsight_learn").strip().lower() == 'true':
                offset, Hdst_hat = pplModel.infer_offset(obv)  # per-step offset
                offset = offset.numpy().squeeze()
                Hdst_hat = Hdst_hat.numpy().squeeze()

            ## take a step in the environment ##
            if action_delay:
                if ep_frame_idx <= args.getInt("delay_frames"):
                    action_buffer.append(action)
                    next_obv, reward, done, info = env.advance()
                    obv_buffer.append(next_obv)
                else:
                    action_buffer.append(action)
                    delayed_action = action_buffer.popleft()
                    if args.getArg("use_env_prior").strip().lower() == 'true' and args.getArg("hindsight_learn").strip().lower() == 'true':
                        next_obv, reward, done, obv_prior, info = env.step(delayed_action, offset)
                    elif args.getArg("use_env_prior").strip().lower() == 'true' and not args.getArg("hindsight_learn").strip().lower() == 'true':
                        next_obv, reward, done, obv_prior, info = env.step(delayed_action)
                    else:
                        next_obv, reward, done, info = env.step(delayed_action)
                    obv_buffer.append(next_obv)
            else:
                if args.getArg("use_env_prior").strip().lower() == 'true' and args.getArg("hindsight_learn").strip().lower() == 'true':
                    next_obv, reward, done, obv_prior, info = env.step(action, offset)
                elif args.getArg("use_env_prior").strip().lower() == 'true' and not args.getArg("hindsight_learn").strip().lower() == 'true':
                    next_obv, reward, done, obv_prior, info = env.step(action)
                else:
                    next_obv, reward, done, info = env.step(action)

            ## infer epistemic signal using generative model ##
            obv_tp1 = tf.convert_to_tensor(next_obv, dtype=tf.float32)
            obv_tp1 = tf.expand_dims(obv_tp1, axis=0)
            a_t = tf.expand_dims(action, axis=0)
            a_t = tf.one_hot(a_t, depth=args.getInt("dim_a"))
            R_te = pplModel.infer_epistemic(obv, obv_tp1, action=a_t)
            R_te = R_te.numpy().squeeze()

            ## train the predictive component ##
            if args.getArg("hindsight_learn").strip().lower() == 'true' and env.time >= 10.0 / env.FPS:
                H_TTC_diff = env.state[2] / env.state[3] - env.state[0] / env.state[1]
                hindsight_buffer.push(observation, H_TTC_diff)
                if len(hindsight_buffer) > args.getInt("h_batch_size") and frame_idx % args.getInt("train_freq") == 0:
                    for _ in range(args.getInt("h_gradient_steps")):
                        h_batch_data = hindsight_buffer.sample(args.getInt("h_batch_size"))
                        [batch_init_obv, batch_hd_error] = h_batch_data
                        grads_pc, loss_h_reconst, loss_h_error = pplModel.train_pc(batch_init_obv, batch_hd_error)
                        grads_pc_clipped = []
                        for grad in grads_pc:
                            if grad is not None:
                                grad = tf.clip_by_norm(grad, clip_norm=args.getFloat("grad_norm_clip")
                                grads_pc_clipped.append(grad)
                            opt2.apply_gradients(zip(grads_pc_clipped, pplModel.param_var))

            if record_stats and ep_idx % args.getInt("record_interval") == 0:
                # Calculate and record TTC and write 1 frame to the video
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
                if args.getArg("record_video").strip().lower() == 'true':
                    img = env.render(mode='rgb_array', offset=offset, isRandom=isRandom)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    video.write(img)
                if args.getArg("hindsight_learn").strip().lower() == 'true' and env.time >= 10.0 / env.FPS:
                    offset_dynamic_list.append(offset)
                    H_dynamic_list.append(H_TTC_diff)
                    H_hat_dynamic_list.append(Hdst_hat)

            episode_reward += reward

            efe_N = 1.0
            ## Save transition tuple to the replay buffer, then train on batch w/ or w/o schedule ##
            if args.getArg("use_per_buffer").strip().lower() == 'true':
                if args.getArg("use_env_prior").strip().lower() == 'true':
                    per_buffer.push(observation, action, reward, next_obv, done, R_te, obv_prior)
                else:
                    per_buffer.push(observation, action, reward, next_obv, done, R_te)
                observation = next_obv
                batch_data = None
                if len(per_buffer) > args.getInt("learning_start"):
                    batch_data = per_buffer.sample(args.getInt("batch_size"), beta=beta)
                efe_N = args.getInt("batch_size") * 1.0
                if batch_data is not None:
                    if args.getArg("use_env_prior").strip().lower() == 'true':
                        [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done, batch_R_te, batch_prior, batch_indices, batch_weights] = batch_data
                    else:
                        [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done, batch_R_te, batch_indices, batch_weights] = batch_data
                    batch_action = tf.one_hot(batch_action, depth=args.getInt("dim_a"))
                    if args.getArg("use_env_prior").strip().lower() == 'true':
                        grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, efe_t, efe_target, priorities = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, batch_R_te, batch_weights, obv_prior=batch_prior, reward=batch_reward)
                    else:
                        grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, efe_t, efe_target, priorities = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, batch_R_te, batch_weights, reward=batch_reward)
                    per_buffer.update_priorities(batch_indices, priorities.numpy())
            else:
                if action_delay:
                    if ep_frame_idx <= args.getInt("delay_frames"):
                        pass
                    else:
                        origin_obv = obv_buffer.popleft()
                        if args.getArg("use_env_prior").strip().lower() == 'true':
                            replay_buffer.push(origin_obv, delayed_action, reward, next_obv, done, R_te, obv_prior)
                        else:
                            replay_buffer.push(origin_obv, delayed_action, reward, next_obv, done, R_te)
                else:
                    if args.getArg("use_env_prior").strip().lower() == 'true':
                        replay_buffer.push(observation, action, reward, next_obv, done, R_te, obv_prior)
                    else:
                        replay_buffer.push(observation, action, reward, next_obv, done, R_te)

                observation = next_obv

                if args.getArg("keep_expert_batch").strip().lower() == 'true':
                    total_samples = len(expert_buffer) + len(replay_buffer)
                    if total_samples / len(replay_buffer) < args.getInt("batch_size") and not args.getArg("equal_replay_batches").strip().lower() == 'true':
                        # Sample from both expert buffer and replay buffer
                        n_replay_samples = np.floor(args.getInt("batch_size") * len(replay_buffer) / total_samples)
                        n_expert_samples = args.getInt("batch_size") - n_replay_samples
                        expert_batch = expert_buffer.sample(int(n_expert_samples))
                        replay_batch = replay_buffer.sample(int(n_replay_samples))
                        batch_data = [np.concatenate((expert_sample, replay_sample)) for expert_sample, replay_sample in zip(expert_batch, replay_batch)]
                    elif args.getArg("equal_replay_batches").strip().lower() == 'true' and (len(replay_buffer) > (args.getInt("batch_size") / 2)):
                        n_expert_samples = args.getInt("batch_size") // 2
                        n_replay_samples = args.getInt("batch_size") // 2
                        expert_batch = expert_buffer.sample(int(n_expert_samples))
                        replay_batch = replay_buffer.sample(int(n_replay_samples))
                        batch_data = [np.concatenate((expert_sample, replay_sample)) for expert_sample, replay_sample in zip(expert_batch, replay_batch)]
                    else:
                        # Sample from expert buffer only
                        batch_data = expert_buffer.sample(args.getInt("batch_size"))
                else:
                    batch_data = None
                    if len(replay_buffer) > args.getInt("learning_start") and frame_idx % args.getInt("train_freq") == 0:
                        for _ in range(args.getInt("gradient_steps")):
                            batch_data = replay_buffer.sample(args.getInt("batch_size"))
                            if args.getArg("use_env_prior").strip().lower() == 'true':
                                [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done, batch_R_te, batch_prior] = batch_data
                                batch_action = tf.one_hot(batch_action, depth=args.getInt("dim_a"))
                                grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, efe_t, efe_target = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, batch_R_te, obv_prior=batch_prior, reward=batch_reward)
                            else:
                                [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done, batch_R_te] = batch_data
                                batch_action = tf.one_hot(batch_action, depth=args.getInt("dim_a"))
                                grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, efe_t, efe_target = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, batch_R_te, reward=batch_reward)

                            if tf.math.is_nan(loss_efe):
                                print("loss_efe nan at frame #", frame_idx)
                                break

                            ### Clip gradients ###
                            crash = False
                            grads_model_clipped = []
                            for grad in grads_model:
                                if grad is not None:
                                    if args.getArg("clip_type").strip().lower() == "hard_clip":
                                        grad = tf.clip_by_value(grad, -args.getFloat("grad_norm_clip"), args.getFloat("grad_norm_clip"))
                                    else:
                                        grad = tf.clip_by_norm(grad, clip_norm=args.getFloat("grad_norm_clip"))
                                    if tf.math.reduce_any(tf.math.is_nan(grad)):
                                        print("grad_model nan at frame # ", frame_idx)
                                        crash = True
                                grads_model_clipped.append(grad)

                            grads_efe_clipped = []
                            for grad in grads_efe:
                                if grad is not None:
                                    if args.getArg("clip_type").strip().lower() == "hard_clip":
                                        grad = tf.clip_by_value(grad, -args.getFloat("grad_norm_clip"), args.getFloat("grad_norm_clip"))
                                    else:
                                        grad = tf.clip_by_norm(grad, clip_norm=args.getFloat("grad_norm_clip"))
                                    if tf.math.reduce_any(tf.math.is_nan(grad)):
                                        print("grad_efe nan at frame # ", frame_idx)
                                        crash = True
                                grads_efe_clipped.append(grad)

                            if crash:
                                break

                            ### Gradient descend by Adam optimizer excluding variables with no gradients ###
                            opt.apply_gradients(zip(grads_model_clipped, pplModel.param_var))
                            opt.apply_gradients(zip(grads_efe_clipped, pplModel.param_var))

            if args.getFloat("learning_rate_decay") > 0.0:
                lower_bound_lr = 1e-7
                lr.assign(max(lower_bound_lr, float(lr * args.getFloat("learning_rate_decay"))))

            if frame_idx % args.getInt("target_update_step") == 0:
                pplModel.update_target()

        pplModel.clear_state()

        if args.getArg("hindsight_learn").strip().lower() == 'true':
            print(f"frame {frame_idx}, L.h_reconst = {loss_h_reconst.numpy()}, L.h_error = {loss_h_error.numpy()}")
            print(f"alpha {pplModel.alpha.numpy()}, beta {pplModel.beta.numpy()}")
            # Record hindsight error
            hindsight_error_list.append(env.hindsight_error)
            offset_list.append(np.asarray(offset_dynamic_list))
            H_TTC_diff_list.append(np.asarray(H_dynamic_list))
            hdst_hat_list.append(np.asarray(H_hat_dynamic_list))
            alpha_list.append(pplModel.alpha.numpy())
            beta_list.append(pplModel.beta.numpy())

        if reward == 0:
            # Record who passes the interception point first
            if env.state[0] < env.state[2]:
                target_front_count += 1
            else:
                subject_front_count += 1
            # Record TTC difference
            TTC_diff = env.state[2] / env.state[3] - env.state[0] / env.state[1]
            TTC_diff_list.append((ep_idx, TTC_diff))
            if args.getArg("hindsight_learn").strip().lower() == 'true':
                failed_offset_list.append((ep_idx, offset))

        env.close()
        ### After each training episode is done ###

        if loss_efe is not None:
            print("-----------------------------------------------------------------")
            print(f"frame {frame_idx}, L.model = {loss_model.numpy()}, L.efe = {(loss_efe / efe_N).numpy()}  eps = {pplModel.epsilon.numpy()}, rho = {pplModel.rho.numpy()}")

        ### Evaluate the PPL model using a number of episodes ###
        if args.getArg("eval_model").strip().lower() == 'true':
            pplModel.epsilon.assign(0.0)  # Use greedy policy when testing
            reward_list = []
            for _ in range(args.getInt("test_episodes")):
                if args.getArg("env_name") == "InterceptionEnv":
                    f_speed_idx = np.random.randint(3)
                    env = InterceptionEnv(target_speed_idx=f_speed_idx, approach_angle_idx=3, return_prior=env_prior, use_slope=False, perfect_prior=perfect_prior)
                observation = env.reset()
                episode_reward = 0
                done_test = False
                while not done_test:
                    obv = tf.convert_to_tensor(observation, dtype=tf.float32)
                    obv = tf.expand_dims(obv, axis=0)
                    action, _, _ = pplModel.act(obv)
                    action = action.numpy().squeeze()
                    observation, reward, done_test, _ = env.step(action)
                    episode_reward += reward
                reward_list.append(episode_reward)
            mean_reward = np.mean(reward_list)
            mean_ep_reward.append(mean_reward)
            episode_reward = mean_reward

        pplModel.clear_state()
        # Otherwise, just use training reward as episode reward (online learning)
        global_reward.append(episode_reward)
        reward_window.append(episode_reward)
        reward_window_mean = calc_window_mean(reward_window)
        trial_win_mean.append(reward_window_mean)

        print(f"episode {ep_idx + 1}, r.mu = {episode_reward:.3f}  win.mu = {reward_window_mean:.3f}")
        print("-----------------------------------------------------------------")
        if ep_idx % 50 == 0:
            rewards_fname = f"{out_dir}trial{trial}"
            print(f" => Saving reward sequence to {rewards_fname}")
            np.save(f"{rewards_fname}_R", np.array(global_reward))
            agent_fname = f"{out_dir}trial_{trial}_epd_{ep_idx}.agent"
            save_object(pplModel, fname=agent_fname)

        # Annealing of the epistemic term based on the average test rewards
        if args.getArg("epistemic_anneal").strip().lower() == 'true':
            if not rho_anneal_start and reward_window_mean > args.getFloat("anneal_start_reward"):
                start_ep = ep_idx
                rho_anneal_start = True
            if rho_anneal_start:
                rho = rho_by_episode(ep_idx - start_ep)
                pplModel.rho.assign(rho)

        if record_stats and ep_idx % args.getInt("record_interval") == 0:
            EFE_values_trial_list.append(np.asarray(efe_list))
            if args.getArg("record_video").strip().lower() == 'true':
                video.release()

    env.close()
    all_win_mean.append(np.asarray(trial_win_mean))
    agent_fname = f"{out_dir}trial{trial}"
    print(f"==> Saving reward sequence to {agent_fname}")
    np.save(f"{agent_fname}_R", np.array(global_reward))
    print(f"==> Saving QAI model: {agent_fname}")
    save_object(pplModel, fname=f"{agent_fname}.agent")

    if record_stats:
        trial_TTCs = np.vstack((target_1st_order_TTC_list, target_actual_mean_TTC_list, agent_TTC_list, f_speed_idx_list))
        print("==> Saving TTC sequence...")
        np.save(f"{out_dir}trial_{trial}_TTCs.npy", trial_TTCs)
        print("==> Saving EFE sequence...")
        np.save(f"{out_dir}trial_{trial}_EFE_values.npy", EFE_values_trial_list)
        print("==> Saving hindsight error sequence...")
        np.save(f"{out_dir}trial_{trial}_hindsight_errors.npy", hindsight_error_list)
        np.save(f"{out_dir}trial_{trial}_H_dynamic.npy", H_TTC_diff_list)
        np.save(f"{out_dir}trial_{trial}_H_hat_dynamic.npy", hdst_hat_list)
        print("==> Saving alpha and beta sequence...")
        np.save(f"{out_dir}trial_{trial}_alpha.npy", alpha_list)
        np.save(f"{out_dir}trial_{trial}_beta.npy", beta_list)
        print("==> Saving TTC_diff sequence...")
        np.save(f"{out_dir}trial_{trial}_TTC_diffs.npy", np.asarray(TTC_diff_list))
        print("==> Saving offset sequence...")
        np.save(f"{out_dir}trial_{trial}_offsets.npy", offset_list)
        np.save(f"{out_dir}trial_{trial}_failed_offsets.npy", np.asarray(failed_offset_list))
        with open(f"{out_dir}who_passes_first.txt", 'a+') as f:
            total_fail_cases = target_front_count + subject_front_count
            f.write(f"trial_{trial}_target_passes_first: {100 * target_front_count / total_fail_cases:.2f}%\n")
            f.write(f"trial_{trial}_subject_passes_first: {100 * subject_front_count / total_fail_cases:.2f}%\n")

def setup_video_writer(out_dir, width, height, fourcc, FPS, ep_idx, f_speed_idx):
    """
    Set up a video writer for recording the agent's behavior.
    """
    video = cv2.VideoWriter(f"{out_dir}trial_{trial}_epd_{ep_idx}_tsidx_{f_speed_idx}.avi", fourcc, float(FPS), (width, height))
    return video

if __name__ == "__main__":
    args = load_config()
    n_trials = args.getInt("n_trials")
    out_dir = args.getArg("out_dir")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    shutil.copy(args.cfg_fname, out_dir)
    prior_model_save_path = args.getArg("prior_model_save_path") if args.hasArg("prior_model_save_path") else None
    expert_data_path = args.getArg("expert_data_path") if args.hasArg("expert_data_path") else None
    eval_model = args.getArg("eval_model").strip().lower() == 'true'

    all_data = load_expert_data(expert_data_path)

    plot_rewards = args.getArg("plot_rewards").strip().lower() == 'true'
    if plot_rewards:
        import matplotlib.pyplot as plt
        # ... (plotting code)

    all_win_mean = []

    for trial in range(args.getInt("start_trial"), n_trials):
        per_buffer, hindsight_buffer = setup_replay_buffers(args.getInt("buffer_size"), args.getInt("h_buffer_size"), args.getArg("use_per_buffer").strip().lower() == 'true', args.getFloat("prob_alpha"), args.getInt("seed"), all_data)

        if not args.getArg("use_per_buffer").strip().lower() == 'true':
            expert_buffer, replay_buffer, hindsight_buffer = setup_replay_buffers(args.getInt("buffer_size"), args.getInt("h_buffer_size"), args.getArg("use_per_buffer").strip().lower() == 'true', args.getFloat("prob_alpha"), args.getInt("seed"), all_data)

        pplModel = setup_models(args, None, prior_model_save_path, args.getArg("epistemic_off").strip().lower() == 'true')

        epsilon_by_frame, gamma_by_episode, rho_by_episode, beta_by_episode = setup_schedules(args.getInt("num_frames"), args.getFloat("epsilon_start"), args.getFloat("epsilon_final"))

        train_agent(args, pplModel, env, per_buffer, expert_buffer, replay_buffer, hindsight_buffer, all_data, epsilon_by_frame, gamma_by_episode, rho_by_episode, beta_by_episode)