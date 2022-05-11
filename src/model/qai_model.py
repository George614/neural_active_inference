# -*- coding: utf-8 -*-
"""
A Q-learning interpretation of active inference.
Generative model/likelihood and approximate posterior assumed to be
identity matrices and thus are cancelled out in this implementation

@author: Alexander G. Ororbia, Zhizhuo (George) Yang
"""
import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../utils/')
from utils import softmax, sample_gaussian, sample_gaussian_with_logvar, \
                  g_nll_from_logvar, kl_div_loss, g_nll, kl_d, mse, load_object, \
                  entropy_gaussian_from_logvar, huber
from prob_mlp import ProbMLP
from config import Config

class QAIModel:
    def __init__(self, prior, args):
        self.prior = prior
        self.args = args

        self.dim_o = int(args.getArg("dim_o"))  # observation size
        self.dim_a = int(args.getArg("dim_a"))  # action size
        self.global_mu = tf.zeros((1, self.dim_o), name="speed_diff")
        self.layer_norm = args.getArg("layer_norm").strip().lower() == 'true'
        self.l2_reg = float(args.getArg("l2_reg"))
        self.act_fx = args.getArg("act_fx")
        self.efe_act_fx = args.getArg("efe_act_fx")
        self.gamma_d = float(args.getArg("gamma_d"))
        self.lambda_h_error = float(args.getArg("lambda_h_error"))
        self.use_sum_q = args.getArg("use_sum_q").strip().lower() == 'true'
        self.instru_term = args.getArg("instru_term")
        self.normalize_signals = args.getArg("normalize_signals").strip().lower() == 'true'
        self.seed = int(args.getArg("seed"))
        self.rho = float(args.getArg("rho"))
        self.use_prior_space = args.getArg("env_prior").strip().lower() == 'prior_error'
        self.use_combined_nn = args.getArg("combined_nn").strip().lower() == 'true'
        self.use_bonus = args.getArg("use_bonus").strip().lower() == 'true'
        self.dueling_q = args.getArg("dueling_q").strip().lower() == 'true'
        self.EFE_bound = 1.0
        self.max_R_ti = 1.0
        self.min_R_ti = 0.01 #-1.0 for GLL #0.01
        self.max_R_te = 1.0
        self.min_R_te = 0.01 #-1.0 for GLL #0.01
        self.max_R_t = -0.01
        self.min_R_t = -1.0
        self.obv_clip = 20.0
        self.obv_bound = 1.0
        self.max_obv = 1.0
        self.min_obv = -1.0
        self.efe_loss = str(args.getArg("efe_loss"))

        hid_dims = args.getArg("net_arch")
        if hid_dims[0] == '[':
            hid_dims = hid_dims[1:-1]
        hid_dims = [int(s) for s in hid_dims.split(',')]

        ## transition dims ##
        trans_dims = [(self.dim_o + self.dim_a)]
        trans_dims = trans_dims + hid_dims
        trans_dims.append(self.dim_o)

        ## EFE value dims ##
        efe_dims = [self.dim_o]
        efe_dims = efe_dims + hid_dims
        efe_dims.append(self.dim_a)

        ## Recognition model dims ##
        recog_dims = [self.dim_o]
        recog_dims = recog_dims + hid_dims

        ## EFE head dims ##
        efe_head_dims = [hid_dims[-1]]
        efe_head_dims.append(self.dim_a)

        if self.dueling_q:
            self.v_ad_dims = hid_dims[-1] // 2
            efe_vhead_dims = [self.v_ad_dims]
            efe_vhead_dims.append(1)
            efe_adhead_dims = [self.v_ad_dims]
            efe_adhead_dims.append(self.dim_a)
            efe_vhead_tar_dim = [self.dim_o] + hid_dims[:-1]
            efe_vhead_tar_dim.append(self.v_ad_dims)
            efe_vhead_tar_dim.append(1)
            efe_adhead_tar_dim = [self.dim_o] + hid_dims[:-1]
            efe_adhead_tar_dim.append(self.v_ad_dims)
            efe_adhead_tar_dim.append(self.dim_a)
            efe_v_dims = [self.dim_o] 
            efe_v_dims = efe_v_dims + hid_dims[:-1]
            efe_v_dims.append(self.v_ad_dims)
            efe_v_dims.append(1)
            efe_ad_dims = [self.dim_o] 
            efe_ad_dims = efe_v_dims + hid_dims[:-1]
            efe_ad_dims.append(self.v_ad_dims)
            efe_ad_dims.append(self.dim_a)

        ## Observation head dims ##
        obv_head_dims = [hid_dims[-1]]
        obv_head_dims.append(self.dim_o)

        ## offset model dims ##
        offsetM_dims = [4, 64, 1]

        act_fun = self.act_fx #"relu"
        efe_act_fun = self.efe_act_fx #"relu6"
        wght_sd = 0.025
        init_type = "alex_uniform"

        self.efe_target = ProbMLP(name="EFE_targ",z_dims=efe_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                                  init_type=init_type,use_layer_norm=self.layer_norm, seed=self.seed)
        ## offset prediction model / predictive component ##
        self.offset_model = ProbMLP(name="OffsetModel",z_dims=offsetM_dims, act_fun=act_fun, wght_sd=wght_sd,
                                  init_type=init_type,use_layer_norm=self.layer_norm, seed=self.seed)

        if self.use_combined_nn:
            ## Recognition model that combines partial functionalities of EFe model and forward/transition model ##
            self.recognition = ProbMLP(name="Recognition", z_dims=recog_dims, act_fun=act_fun, out_fun=act_fun,
                                        wght_sd=wght_sd, init_type=init_type, use_layer_norm=self.layer_norm,
                                        seed=self.seed)
            ## Add-on neural layer to the Recognition model for predicting future observation ##
            self.obv_head = ProbMLP(name="Obv_head", z_dims=obv_head_dims, act_fun=act_fun, wght_sd=wght_sd,
                                      init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)
            ## Add-on neural layer to the Recognition model for estimating EFE values ##
            if self.dueling_q:
                self.efe_vhead = ProbMLP(name="EFE_vhead", z_dims=efe_vhead_dims, act_fun=act_fun, wght_sd=wght_sd,
                                      init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)
                self.efe_adhead = ProbMLP(name="EFE_adhead", z_dims=efe_adhead_dims, act_fun=act_fun, wght_sd=wght_sd,
                                      init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)
                self.efe_vhead_tar = ProbMLP(name="EFE_vhead_tar", z_dims=efe_vhead_tar_dim, act_fun=act_fun, wght_sd=wght_sd,
                                      init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)
                self.efe_adhead_tar = ProbMLP(name="EFE_adhead_tar", z_dims=efe_adhead_tar_dim, act_fun=act_fun, wght_sd=wght_sd,
                                      init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)
            else:
                self.efe_head = ProbMLP(name="EFE_head", z_dims=efe_head_dims, act_fun=act_fun, wght_sd=wght_sd,
                                      init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)
        else:
            ## transition model ##
            self.transition = ProbMLP(name="Trans",z_dims=trans_dims, act_fun=act_fun, wght_sd=wght_sd,
                                      init_type=init_type,use_layer_norm=self.layer_norm, seed=self.seed)
            ## EFE value model ##
            if self.dueling_q:
                self.efe_v = ProbMLP(name="EFE_Value", z_dims=efe_v_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                               init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)
                self.efe_ad = ProbMLP(name="EFE_Advantage", z_dims=efe_ad_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                               init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)
                self.efe_v_tar = ProbMLP(name="EFE_V_target", z_dims=efe_v_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                               init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)
                self.efe_ad_tar = ProbMLP(name="EFE_Adv_target", z_dims=efe_ad_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                               init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)
            else:
                self.efe = ProbMLP(name="EFE", z_dims=efe_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                                   init_type=init_type, use_layer_norm=self.layer_norm, seed=self.seed)

        # clump all parameter variables of sub-models/modules into one shared pointer list
        self.param_var = []

        if self.use_combined_nn:
            self.param_var = self.param_var + self.recognition.extract_params()
            self.param_var = self.param_var + self.obv_head.extract_params()
            if self.dueling_q:
                self.param_var = self.param_var + self.efe_vhead.extract_params()
                self.param_var = self.param_var + self.efe_adhead.extract_params()
            else:
                self.param_var = self.param_var + self.efe_head.extract_params()
        else:
            self.param_var = self.param_var + self.transition.extract_params()
            if self.dueling_q:
                self.param_var = self.param_var + self.efe_v.extract_params()
                self.param_var = self.param_var + self.efe_ad.extract_params()
            else:
                self.param_var = self.param_var + self.efe.extract_params()
        self.param_var = self.param_var + self.offset_model.extract_params()

        self.epsilon = tf.Variable(1.0, name='epsilon', trainable=False)  # epsilon greedy parameter
        self.gamma = tf.Variable(1.0, name='gamma', trainable=False)  # gamma weighting factor for balance KL-D on transition vs unit Gaussian
        self.rho = tf.Variable(self.rho, name='rho', trainable=False)  # weight term on the epistemic value
        self.alpha = tf.Variable(1.0, name='alpha', trainable=True)  # linear tranformation scale factor for offset
        self.beta = tf.Variable(0.0, name='beta', trainable=True)  # linear transformation shift factor for offset
        self.param_var.append(self.alpha)
        self.param_var.append(self.beta)
        self.tau = -1 # if set to 0, then no Polyak averaging is used for target network
        self.update_target()

    def update_target(self):
        if self.use_combined_nn:
            if self.dueling_q:
                efe_v_weights = self.recognition.param_var[:-2] +\
                            [self.recognition.param_var[-2][:, :self.v_ad_dims], self.recognition.param_var[-1][:, :self.v_ad_dims]] +\
                            self.efe_vhead.param_var
                efe_ad_weights = self.recognition.param_var[:-2] +\
                            [self.recognition.param_var[-2][:, self.v_ad_dims:], self.recognition.param_var[-1][:, self.v_ad_dims:]] +\
                            self.efe_adhead.param_var
                self.efe_vhead_tar.set_weights(efe_v_weights, tau=self.tau)
                self.efe_adhead_tar.set_weights(efe_ad_weights, tau=self.tau)
            else:
                efe_weights = self.recognition.param_var + self.efe_head.param_var
                self.efe_target.set_weights(efe_weights, tau=self.tau)
        else:
            if self.dueling_q:
                self.efe_v_tar.set_weights(self.efe_v, tau=self.tau)
                self.efe_ad_tar.set_weights(self.efe_ad, tau=self.tau)
            else:
                self.efe_target.set_weights(self.efe, tau=self.tau)

    def act(self, o_t):
    # get EFE values always
        if self.use_combined_nn:
            if self.dueling_q:
                hidden_vars, _, _ = self.recognition.predict(o_t)
                efe_t_v, _, _ = self.efe_vhead.predict(hidden_vars[:, :self.v_ad_dims])
                efe_t_ad, _, _ = self.efe_adhead.predict(hidden_vars[:, self.v_ad_dims:])
                efe_t = efe_t_v + (efe_t_ad - tf.reduce_mean(efe_t_ad, axis=-1, keepdims=True))
            else:
                hidden_vars, _, _ = self.recognition.predict(o_t)
                efe_t, _, _ = self.efe_head.predict(hidden_vars)
        else:
            if self.dueling_q:
                efe_t_v, _, _ = self.efe_v.predict(o_t)
                efe_t_ad, _, _ = self.efe_ad.predict(o_t)
                efe_t = efe_t_v + (efe_t_ad - tf.reduce_mean(efe_t_ad, axis=-1, keepdims=True))
            else:
                efe_t, _, _ = self.efe.predict(o_t)
        if tf.random.uniform(shape=()) > self.epsilon:
            action = tf.argmax(efe_t, axis=-1, output_type=tf.int32)
            isRandom = False
        else:
            action = tf.random.uniform(shape=(), maxval=self.dim_a, dtype=tf.int32)
            isRandom = True
        return action, efe_t, isRandom

    def clear_state(self):
        pass

    def infer_offset(self, init_obv):
        offset, _, _ = self.offset_model.predict(init_obv)
        offset = tf.clip_by_value(offset, -12.0, 12.0)
        H_hat = offset * self.alpha + self.beta
        return offset, H_hat

    def train_pc(self, init_obv, H_error):
        with tf.GradientTape() as tape:
            offset, _, _ = self.offset_model.predict(init_obv)
            offset = tf.clip_by_value(offset, -12.0, 12.0)
            H_hat = offset * self.alpha + self.beta  # estimated hindsight error given offset in speed_diff
            # offset = tf.clip_by_value(offset, -3.0, 3.0)
            loss_h_reconst = mse(H_hat, H_error)
            loss_h_error = mse(H_hat, tf.zeros_like(H_hat))
            loss_pc = loss_h_reconst + loss_h_error * self.lambda_h_error
        grads_pc = tape.gradient(loss_pc, self.param_var)

        return grads_pc, loss_h_reconst, loss_h_error

    def infer_epistemic(self, obv_t, obv_next, action=None):
        if self.use_combined_nn:
            # use the intermediate neural activation as inputs to the predictive model
            hidden_vars, _, _ = self.recognition.predict(obv_t)
            o_next_hat, _, _ = self.obv_head.predict(hidden_vars)
        else:
            ### run s_t and a_t through transition model ###
            o_next_hat, _, _ = self.transition.predict(tf.concat([obv_t, action], axis=-1))
        delta = obv_next - o_next_hat
        R_te = tf.reduce_sum(delta * delta, axis=1, keepdims=True)
        # clip the epistemic value
        R_te = tf.clip_by_value(R_te, -50.0, 50.0)
        return R_te

    def train_step(self, obv_t, obv_next, action, done, R_te, weights=None, reward=None, obv_prior=None):
        with tf.GradientTape(persistent=True) as tape:
            if self.use_combined_nn:
                hidden_vars, _, _ = self.recognition.predict(obv_t)
                o_next_tran_mu, _, _ = self.obv_head.predict(hidden_vars)
                if self.dueling_q:
                    efe_t_v, _, _ = self.efe_vhead.predict(hidden_vars[:, :self.v_ad_dims])
                    efe_t_ad, _, _ = self.efe_adhead.predict(hidden_vars[:, self.v_ad_dims:])
                    efe_t = efe_t_v + (efe_t_ad - tf.reduce_mean(efe_t_ad, axis=-1, keepdims=True))
                else:
                    efe_t, _, _ = self.efe_head.predict(hidden_vars)
            else:
                ### run s_t and a_t through transition model ###
                o_next_tran_mu, _, _ = self.transition.predict(tf.concat([obv_t, action], axis=-1))
                ### predict EFE value at time t ###
                if self.dueling_q:
                    efe_t_v, _, _ = self.efe_v.predict(obv_t)
                    efe_t_ad, _, _ = self.efe_ad.predict(obv_t)
                    efe_t = efe_t_v + (efe_t_ad - tf.reduce_mean(efe_t_ad, axis=-1, keepdims=True))
                else:
                    efe_t, _, _ = self.efe.predict(obv_t)

            with tape.stop_recording():
                ### instrumental term ###
                R_ti = None
                if self.instru_term == "prior_local":
                    if obv_prior is None:  # use learned / emperical prior model
                        o_prior_mu, o_prior_std, _ = self.prior.predict(obv_t)
                        # difference between preferred future and actual future, i.e. instrumental term
                        #R_ti = -1.0 * g_nll(obv_next, o_prior_mu, o_prior_std * o_prior_std, keep_batch=True)
                        R_ti = -1.0 * mse(x_true=obv_next, x_pred=o_prior_mu, keep_batch=True)
                        # R_ti = -1.0 * huber(obv_next, o_prior_mu)
                    else:
                        if self.use_prior_space: # calculate instrumental term in prior space
                            error_prior_space = tf.reduce_sum(action * obv_prior, axis=1, keepdims=True)
                            # R_ti = -1.0 * mse(error_prior_space, tf.zeros_like(error_prior_space), keep_batch=True)
                            error_prior = huber(error_prior_space, 0.0)
                            R_ti = -1.0 * tf.expand_dims(error_prior, axis=1)
                        else:  # calculate instrumental term in observation space
                            R_ti = -1.0 * mse(x_true=obv_next, x_pred=obv_prior, keep_batch=True)
                    if self.normalize_signals:
                        a = -self.EFE_bound
                        b = self.EFE_bound
                        self.max_R_ti = max(self.max_R_ti, float(tf.reduce_max(R_ti)))
                        self.min_R_ti = min(self.min_R_ti, float(tf.reduce_min(R_ti)))
                        R_ti = ((R_ti - self.min_R_ti) * (b - a))/(self.max_R_ti - self.min_R_ti) + a
                    else:
                        # clip the instrumental value
                        R_ti = tf.clip_by_value(R_ti, -50.0, 50.0)
                    if self.use_bonus:
                        R_ti = R_ti + tf.expand_dims(reward, axis=1) * 100
                elif self.instru_term == "prior_global":
                    R_ti = -1.0 * mse(x_true=obv_next, x_pred=self.global_mu, keep_batch=True)
                else:
                    R_ti = reward
                    if len(R_ti.shape) < 2:
                        R_ti = tf.expand_dims(R_ti, axis=1)

                # the nagative EFE value, i.e. the reward. Note the sign here
                R_t = R_ti + self.rho * R_te

            ## model reconstruction loss ##
            loss_reconst = mse(x_true=obv_next, x_pred=o_next_tran_mu, keep_batch=True) #g_nll(obv_next, o_next_mu, o_next_std * o_next_std)
            done = tf.cast(done, dtype=tf.float32)
            loss_reconst *= (1 - done) # done samples contain invalid observations
            loss_reconst = tf.math.reduce_mean(loss_reconst)

            # regularization for weights
            loss_l2 = 0.0
            if self.l2_reg > 0.0:
                loss_l2 = tf.add_n([tf.nn.l2_loss(var) for var in self.transition.param_var if 'W' in var.name]) * self.l2_reg
            ## compute full loss ##
            loss_model = loss_reconst + loss_l2
            
            done = tf.expand_dims(done, axis=1)

            if self.use_sum_q:
                # take the old EFE values given action indices
                efe_old = tf.math.reduce_sum(efe_t * action, axis=-1)
                with tape.stop_recording():
                    if self.dueling_q:
                        if self.use_combined_nn:
                            efe_tp1_v, _, _ = self.efe_vhead_tar.predict(obv_next)
                            efe_tp1_ad, _, _ = self.efe_adhead_tar.predict(obv_next)
                            efe_target = efe_tp1_v + (efe_tp1_ad - tf.reduce_mean(efe_tp1_ad, axis=-1, keepdims=True))
                        else:
                            efe_tp1_v, _, _ = self.efe_v_tar.predict(obv_next)
                            efe_tp1_ad, _, _ = self.efe_ad_tar.predict(obv_next)
                            efe_target = efe_tp1_v + (efe_tp1_ad - tf.reduce_mean(efe_tp1_ad, axis=-1, keepdims=True))
                    else:
                        efe_target, _, _ = self.efe_target.predict(obv_next)
                    idx_a_next = tf.math.argmax(efe_target, axis=-1, output_type=tf.dtypes.int32)
                    onehot_a_next = tf.one_hot(idx_a_next, depth=self.dim_a)
                    # take the new EFE values
                    efe_new = tf.math.reduce_sum(efe_target * onehot_a_next, axis=-1)
                    y_j = R_t + (efe_new * self.gamma_d) * (1 - done)
            else:
                efe_old = efe_t
                with tape.stop_recording():
                    if self.dueling_q:
                        if self.use_combined_nn:
                            efe_tp1_v, _, _ = self.efe_vhead_tar.predict(obv_next)
                            efe_tp1_ad, _, _ = self.efe_adhead_tar.predict(obv_next)
                            efe_target = efe_tp1_v + (efe_tp1_ad - tf.reduce_mean(efe_tp1_ad, axis=-1, keepdims=True))
                        else:
                            efe_tp1_v, _, _ = self.efe_v_tar.predict(obv_next)
                            efe_tp1_ad, _, _ = self.efe_ad_tar.predict(obv_next)
                            efe_target = efe_tp1_v + (efe_tp1_ad - tf.reduce_mean(efe_tp1_ad, axis=-1, keepdims=True))
                    else:
                        efe_target, _, _ = self.efe_target.predict(obv_next)
                    efe_new = tf.reduce_max(efe_target, axis=1, keepdims=True) # max EFE values at t+1
                    y_j = R_t + (efe_new * self.gamma_d) * (1.0 - done)
                    y_j = (action * y_j) + (efe_old * (1.0 - action))

            ## Temporal Difference loss ##
            if weights is not None:
                loss_efe_batch = None
                if self.efe_loss == "huber":
                    loss_efe_batch = huber(y_j, efe_old)
                else:
                    loss_efe_batch = mse(x_true=y_j, x_pred=efe_old, keep_batch=True)
                priorities = tf.math.abs(loss_efe_batch) + 1e-5
                loss_efe_batch = tf.squeeze(loss_efe_batch)
                loss_efe_batch *= weights
                loss_efe = tf.math.reduce_mean(loss_efe_batch)
            else:
                if self.efe_loss == "huber":
                    # loss_efe = huber(x_true=y_j, x_pred=efe_old)
                    loss_efe = tf.reduce_mean(huber(y_j, efe_old))
                else:
                    loss_efe = mse(x_true=y_j, x_pred=efe_old)

        # calculate gradient w.r.t model reconstruction and TD respectively
        grads_model = tape.gradient(loss_model, self.param_var)
        grads_efe = tape.gradient(loss_efe, self.param_var)

        if weights is not None:
            return grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, efe_t, efe_target, priorities

        return grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, efe_t, efe_target
