# -*- coding: utf-8 -*-
"""
A Q-learning interpretation of active inference.
Generative model/likelihood and approximate posterior assumed to be
identity matrices and thus are cancelled out in this implementation

Zhizhuo (George) Yang
"""
from prob_mlp import ProbMLP
from utils import mse, huber
import tensorflow as tf
import sys
sys.path.insert(0, '../utils/')


class QAIModel:
    def __init__(self, args):
        self.args = args
        self.dim_o = int(args.getArg("dim_o"))  # observation size
        self.dim_a = int(args.getArg("dim_a"))  # action size
        self.global_mu = tf.zeros((1, self.dim_o), name="speed_diff")
        self.layer_norm = args.getArg("layer_norm").strip().lower() == 'true'
        self.l2_reg = float(args.getArg("l2_reg"))
        self.act_fx = args.getArg("act_fx")
        self.efe_act_fx = args.getArg("efe_act_fx")
        self.gamma_d = float(args.getArg("gamma_d"))
        self.use_sum_q = args.getArg("use_sum_q").strip().lower() == 'true'
        self.instru_term = args.getArg("instru_term")
        self.normalize_signals = args.getArg(
            "normalize_signals").strip().lower() == 'true'
        self.seed = int(args.getArg("seed"))
        self.rho = float(args.getArg("rho"))
        self.use_prior_space = args.getArg(
            "env_prior").strip().lower() == 'prior_error'
        self.efe_loss = str(args.getArg("efe_loss"))

        hid_dims = args.getArg("net_arch")
        if hid_dims[0] == '[':
            hid_dims = hid_dims[1:-1]
        hid_dims = [int(s) for s in hid_dims.split(',')]

        ## EFE value dims ##
        efe_dims = [self.dim_o]
        efe_dims = efe_dims + hid_dims
        efe_dims.append(self.dim_a)

        act_fun = self.act_fx  # "relu"
        efe_act_fun = self.efe_act_fx  # "relu6"
        wght_sd = 0.025
        init_type = "alex_uniform"

        ## EFE value model ##
        self.efe = ProbMLP(name="EFE", z_dims=efe_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                           init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)
        self.efe_target = ProbMLP(name="EFE_targ", z_dims=efe_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                                  init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)

        # clump all parameter variables of sub-models/modules into one shared pointer list
        self.param_var = []
        self.param_var = self.param_var + self.efe.extract_params()

        # epsilon greedy parameter
        self.epsilon = tf.Variable(1.0, name='epsilon', trainable=False)
        # gamma weighting factor for balance KL-D on transition vs unit Gaussian
        self.gamma = tf.Variable(1.0, name='gamma', trainable=False)
        # weight term on the epistemic value
        self.rho = tf.Variable(self.rho, name='rho', trainable=False)
        self.tau = -1  # if set to 0, then no Polyak averaging is used for target network
        self.update_target()

    def update_target(self):
        self.efe_target.set_weights(self.efe, tau=self.tau)

    def act(self, o_t, return_efe=False):
        if return_efe:  # get EFE values always
            efe_t, _, _ = self.efe.predict(o_t)
            if tf.random.uniform(shape=()) > self.epsilon:
                action = tf.argmax(efe_t, axis=-1, output_type=tf.int32)
                isRandom = False
            else:
                action = tf.random.uniform(
                    shape=(), maxval=self.dim_a, dtype=tf.int32)
                isRandom = True
            return action, efe_t, isRandom
        else:
            if tf.random.uniform(shape=()) > self.epsilon:
                # run EFE model given state at time t
                efe_t, _, _ = self.efe.predict(o_t)
                action = tf.argmax(efe_t, axis=-1, output_type=tf.int32)
            else:  # save computation
                action = tf.random.uniform(
                    shape=(), maxval=self.dim_a, dtype=tf.int32)
            return action

    def train_step(self, obv_t, obv_next, action, done, reward=None, obv_prior=None):
        with tf.GradientTape(persistent=True) as tape:
            ### predict EFE value at time t ###
            efe_t, _, _ = self.efe.predict(obv_t)

            with tape.stop_recording():
                ### instrumental term ###
                R_ti = None
                if self.instru_term == "prior_local":
                    if self.use_prior_space:  # calculate instrumental term in prior space
                        error_prior_space = tf.reduce_sum(
                            action * obv_prior, axis=1, keepdims=True)
                        # R_ti = -1.0 * mse(error_prior_space, tf.zeros_like(error_prior_space), keep_batch=True)
                        error_prior = huber(error_prior_space, 0.0)
                        R_ti = -1.0 * tf.expand_dims(error_prior, axis=1)
                    else:  # calculate instrumental term in observation space
                        R_ti = -1.0 * mse(x_true=obv_next,
                                          x_pred=obv_prior, keep_batch=True)
                    R_ti = tf.clip_by_value(R_ti, -50.0, 50.0)
                elif self.instru_term == "prior_global":
                    R_ti = -1.0 * mse(x_true=obv_next,
                                      x_pred=self.global_mu, keep_batch=True)
                else:
                    R_ti = reward
                    if len(R_ti.shape) < 2:
                        R_ti = tf.expand_dims(R_ti, axis=1)

            done = tf.cast(done, dtype=tf.float32)
            done = tf.expand_dims(done, axis=1)
            efe_old = efe_t

            with tape.stop_recording():
                efe_target, _, _ = self.efe_target.predict(obv_next)
                # max EFE values at t+1
                efe_new = tf.reduce_max(efe_target, axis=1, keepdims=True)
                y_j = R_ti + (efe_new * self.gamma_d) * (1.0 - done)
                y_j = (action * y_j) + (efe_old * (1.0 - action))

            if self.efe_loss == "huber":
                loss_efe = tf.reduce_mean(huber(y_j, efe_old))
            else:
                loss_efe = mse(x_true=y_j, x_pred=efe_old)

        # calculate gradient w.r.t TD loss
        grads_efe = tape.gradient(loss_efe, self.param_var)

        return grads_efe, loss_efe, R_ti, efe_t, efe_target
