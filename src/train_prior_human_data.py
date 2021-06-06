import os
import logging
import sys, getopt, optparse
import pickle
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'model/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import numpy as np
from utils import parse_int_list, mse, save_object, g_nll_from_logvar, load_object, g_nll
from prob_mlp import ProbMLP
from config import Config

"""
Trains the prior (preference) model for use with the proposed QAIModel agent.

"""

def eval_model(model, test_set, mem_batch_size):
    """
        Evaluates the (negative) log likelihood of the prior preference model, i.e.,
        -log p(z_t+1|z_t)
    """
    L = 0.0
    for s_ptr in range(0, len(test_set), mem_batch_size):
        e_ptr = s_ptr + mem_batch_size
        if e_ptr >= len(test_set):
            e_ptr = len(test_set)
        o_t_batch = []
        o_tp1_batch = []
        for s in range(s_ptr, e_ptr):
            o_t, o_tp1 = test_set[s, 0:1], test_set[s, 1:]
            o_t_batch.append(o_t)
            o_tp1_batch.append(o_tp1)
        o_t = tf.cast(np.array(o_t_batch), dtype=tf.float32)
        o_tp1 = tf.cast(np.array(o_tp1_batch), dtype=tf.float32)
        o_mu, o_sigma, o_log_sigma = model.predict(o_t)
        if model.model_variance is True:
            if use_log_form is True:
                L_prior = g_nll_from_logvar(o_tp1, o_mu, o_log_sigma) * o_tp1.shape[0] # un-normalize avg GNLL
            else:
                L_prior = g_nll(o_tp1, o_mu, o_sigma * o_sigma) * o_tp1.shape[0] # un-normalize avg GNLL
        else:
            L_prior = g_nll(o_tp1, o_mu, o_sigma * 0 + 1.0) * o_tp1.shape[0] # un-normalize avg GNLL
            #L_prior = g_nll_from_logvar(o_tp1, o_mu, o_log_sigma * 0) * o_tp1.shape[0] # un-normalize avg GNLL
        L = L_prior + L
    return L / (len(test_set) * 1.0)

################################################################################
# read in configuration file and extract necessary variables/constants
################################################################################
options, remainder = getopt.getopt(sys.argv[1:], '', ["cfg_fname=","gpu_id="])
# Collect arguments from argv
cfg_fname = "fit_interception_prior.cfg"
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
    tf.config.experimental.set_memory_growth(gpu_devices[mid], True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

# global experiment arguments
out_dir = args.getArg("out_dir")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
data_fname = args.getArg("data_fname")

# extract arguments and init the prior generative model
wght_sd = float(args.getArg("wght_sd"))
act_fun = args.getArg("act_fun")
z_dims = args.getArg("z_dims")
z_dims = parse_int_list(z_dims)
obs_dim = int(args.getArg("dim_o"))
init_type = args.getArg("init_type")
model_variance = (args.getArg("model_variance").strip().lower() == 'true')
sigma_fun = args.getArg("sigma_fun")
use_layer_norm = (args.getArg("use_layer_norm").strip().lower() == 'true')

num_iter = int(args.getArg("num_iter"))
opt_type = args.getArg("opt_type")
eta = float(args.getArg("eta"))
l2_reg = float(args.getArg("l2_reg"))
param_radius = float(args.getArg("param_radius"))
update_radius = float(args.getArg("update_radius"))
batch_size = int(args.getArg("batch_size"))
n_train = int(args.getArg("n_train"))
use_log_form = (args.getArg("use_log_form").strip().lower() == 'true')

################################################################################
# set up data and prior model
################################################################################
all_data = np.load(data_fname)
n_test_samples = int(np.floor(len(all_data) * 0.3))
testset = all_data[:n_test_samples, :]
trainset = all_data[n_test_samples:, :]
print(" -> Train.length = {0}  Test.length = {1}".format(len(trainset), len(testset)))

prior_dims = [obs_dim]
prior_dims = prior_dims + z_dims
prior_dims.append(obs_dim)
prior = ProbMLP(name="Prior",z_dims=prior_dims, act_fun=act_fun, wght_sd=wght_sd,
                init_type=init_type, model_variance=model_variance, sigma_fun=sigma_fun,
                use_layer_norm=use_layer_norm)
prior.set_weight_norm(param_radius=prior.param_radius)
prior.update_radius = prior.update_radius
prior.set_optimizer(opt_type, eta, momentum=0.9)
#prior.l2_reg = l2_reg

################################################################################
# begin training prior model
################################################################################

test_loss = []
train_loss = []

L_train = eval_model(prior, trainset, batch_size)
train_loss.append(L_train)
L_test = eval_model(prior, testset, batch_size)
test_loss.append(L_test)
best_loss = L_test
print(" {0}: Train.L = {1}  Test.L = {2}".format(-1, L_train, L_test))
patience = 20
for it in range(num_iter):
    ptrs = np.random.permutation(len(trainset))
    L_train = 0.0 # <-- this is a fast proxy for proper full training loss on a fixed point
    for s_ptr in range(0, len(trainset), batch_size):
        e_ptr = s_ptr + batch_size
        if e_ptr >= len(trainset):
            e_ptr = len(trainset)
        # craft training mini-batch
        o_t_batch = []
        o_tp1_batch = []
        for s in range(s_ptr, e_ptr):
            ptr = int(ptrs[s])
            o_t, o_tp1 = trainset[ptr, 0:1], trainset[ptr, 1:]
            o_t_batch.append(o_t)
            o_tp1_batch.append(o_tp1)
        o_t = tf.cast(np.array(o_t_batch), dtype=tf.float32)
        o_tp1 = tf.cast(np.array(o_tp1_batch), dtype=tf.float32)
        # update model given mini-batch
        with tf.GradientTape(persistent=True) as tape:
            o_mu, o_sigma, o_log_sigma = prior.predict(o_t)
            if prior.model_variance is True:
                if use_log_form is True:
                    L_prior = g_nll_from_logvar(o_tp1, o_mu, o_log_sigma)
                else:
                    L_prior = g_nll(o_tp1, o_mu, o_sigma * o_sigma)
            else:
                L_prior = g_nll(o_tp1, o_mu, o_sigma * 0 + 1.0)
            if l2_reg > 0.0:
                loss_l2 = (tf.add_n([tf.nn.l2_loss(var) for var in prior.param_var if 'W' in var.name])) * l2_reg
                L_prior = L_prior + loss_l2

        prior_grad = tape.gradient(L_prior, prior.param_var)
        prior.update_params(prior_grad)
        # update biased estimate of training set loss
        L_train = (L_prior * o_tp1.shape[0]) + L_train # un-normalize avg GNLL

    # get train GNLL
    L_train = L_train / (len(trainset) * 1.0)
    train_loss.append(L_train)
    # get test GNLL
    L_test = eval_model(prior, testset, batch_size)
    test_loss.append(L_test)
    print(" {0}: Train.L = {1}  Test.L = {2}".format(it, L_train, L_test))
    if L_test < best_loss:
        prior_fname = "{0}prior".format(out_dir)
        print(" --> saving prior model: {0} ".format(prior_fname))
        save_object(prior, fname="{0}.agent".format(prior_fname))
        best_loss = L_test
        patience = 20
    else:
        patience = patience - 1
    if patience <= 0:
        print(" > Patience exhausted...early stopping point here!")
        break

np.save("{0}prior_train_gnll".format(out_dir), np.array(train_loss))
np.save("{0}prior_test_gnll".format(out_dir), np.array(test_loss))
