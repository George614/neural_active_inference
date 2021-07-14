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
    """
        A Q-learning interpretation of active inference.
        Generative model/likelihood and approximate posterior assumed to be
        identity matrices and thus are cancelled out in this implementation

        @author Alexander G. Ororbia
    """
    def __init__(self, prior, args):
        self.prior = prior
        self.args = args

        self.dim_o = int(args.getArg("dim_o"))  # observation size
        self.dim_a = int(args.getArg("dim_a"))  # action size
        self.global_mu = tf.zeros((1, self.dim_o), name="speed_diff")
        self.layer_norm = (args.getArg("layer_norm").strip().lower() == 'true')
        self.l2_reg = float(args.getArg("l2_reg"))
        self.act_fx = args.getArg("act_fx")
        self.efe_act_fx = args.getArg("efe_act_fx")
        self.gamma_d = float(args.getArg("gamma_d"))
        self.use_sum_q = (args.getArg("use_sum_q").strip().lower() == 'true')
        self.instru_term = args.getArg("instru_term")
        self.normalize_signals = (args.getArg("normalize_signals").strip().lower() == 'true')
        self.seed = int(args.getArg("seed"))
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

        hid_dims = [512, 512]

        ## transition dims ##
        trans_dims = [(self.dim_o + self.dim_a)]
        trans_dims = trans_dims + hid_dims
        trans_dims.append(self.dim_o)

        ## EFE value dims ##
        efe_dims = [self.dim_o]
        efe_dims = efe_dims + hid_dims
        efe_dims.append(self.dim_a)

        act_fun = self.act_fx #"relu"
        efe_act_fun = self.efe_act_fx #"relu6"
        wght_sd = 0.025
        init_type = "alex_uniform"

        ## transition model ##
        self.transition = ProbMLP(name="Trans",z_dims=trans_dims, act_fun=act_fun, wght_sd=wght_sd,
                                  init_type=init_type,use_layer_norm=self.layer_norm, seed=self.seed)
        ## EFE value model ##
        self.efe = ProbMLP(name="EFE",z_dims=efe_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                           init_type=init_type,use_layer_norm=self.layer_norm, seed=self.seed)
        self.efe_target = ProbMLP(name="EFE_targ",z_dims=efe_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                                  init_type=init_type,use_layer_norm=self.layer_norm, seed=self.seed)

        # clump all parameter variables of sub-models/modules into one shared pointer list
        self.param_var = []
        self.param_var = self.param_var + self.transition.extract_params()
        self.param_var = self.param_var + self.efe.extract_params()

        self.epsilon = tf.Variable(1.0, trainable=False)  # epsilon greedy parameter
        self.gamma = tf.Variable(1.0, trainable=False)  # gamma weighting factor for balance KL-D on transition vs unit Gaussian
        self.rho = tf.Variable(1.0, trainable=False)  # weight term on the epistemic value
        self.tau = -1 # if set to 0, then no Polyak averaging is used for target network
        self.update_target()

    def update_target(self):
        self.efe_target.set_weights(self.efe, tau=self.tau)

    def act(self, o_t, return_efe=False):
        if tf.random.uniform(shape=()) > self.epsilon:
            # run EFE model given state at time t
            efe_t, _, _ = self.efe.predict(o_t)
            action = tf.argmax(efe_t, axis=-1, output_type=tf.int32)
        else:
            action = tf.random.uniform(shape=(), maxval=self.dim_a, dtype=tf.int32)

        if return_efe:
            return action, efe_t
        return action

    def clear_state(self):
        pass

    def train_step(self, obv_t, obv_next, action, done, weights=None, reward=None, obv_prior=None):
        with tf.GradientTape(persistent=True) as tape:
            ### run s_t and a_t through transition model ###
            o_next_tran_mu, _, _ = self.transition.predict(tf.concat([obv_t, action], axis=-1))
            ### predict EFE value at time t ###
            efe_t, _, _ = self.efe.predict(obv_t)

            with tape.stop_recording():
                ### instrumental term ###
                R_ti = None
                if self.instru_term == "prior_local":
                    if obv_prior is None:
                        o_prior_mu, o_prior_std, _ = self.prior.predict(obv_t)
                        # difference between preferred future and actual future, i.e. instrumental term
                        #R_ti = -1.0 * g_nll(obv_next, o_prior_mu, o_prior_std * o_prior_std, keep_batch=True)
                        R_ti = -1.0 * mse(x_true=obv_next, x_pred=o_prior_mu, keep_batch=True)
                    else:
                        R_ti = -1.0 * mse(x_true=obv_next, x_pred=obv_prior, keep_batch=True)
                    if self.normalize_signals is True:
                        a = -self.EFE_bound
                        b = self.EFE_bound
                        self.max_R_ti = max(self.max_R_ti, float(tf.reduce_max(R_ti)))
                        self.min_R_ti = min(self.min_R_ti, float(tf.reduce_min(R_ti)))
                        R_ti = ((R_ti - self.min_R_ti) * (b - a))/(self.max_R_ti - self.min_R_ti) + a
                    else:
                        # clip the instrumental value
                        R_ti = tf.clip_by_value(R_ti, -50.0, 50.0)
                elif self.instru_term == "prior_global":
                    R_ti = -1.0 * mse(x_true=obv_next, x_pred=self.global_mu, keep_batch=True)
                    # if self.normalize_signals is True:
                    #     a = -self.EFE_bound
                    #     b = self.EFE_bound
                    #     self.max_R_ti = max(self.max_R_ti, float(tf.reduce_max(R_ti)))
                    #     self.min_R_ti = min(self.min_R_ti, float(tf.reduce_min(R_ti)))
                    #     R_ti = ((R_ti - self.min_R_ti) * (b - a))/(self.max_R_ti - self.min_R_ti) + a
                    # else:
                    #     # clip the instrumental value
                    #     R_ti = tf.clip_by_value(R_ti, -50.0, 50.0)
                else:
                    R_ti = reward
                    if len(R_ti.shape) < 2:
                        R_ti = tf.expand_dims(R_ti, axis=1)
                ### epistemic term ###
                delta = (obv_next - o_next_tran_mu)
                R_te = tf.reduce_sum(delta * delta, axis=1, keepdims=True)
                if self.normalize_signals is True:
                    a = -self.EFE_bound
                    b = self.EFE_bound
                    self.max_R_te = max(self.max_R_te, float(tf.reduce_max(R_te)))
                    self.min_R_te = min(self.min_R_te, float(tf.reduce_min(R_te)))
                    R_te = ((R_te - self.min_R_te) * (b - a))/(self.max_R_te - self.min_R_te) + a
                else:
                    # clip the epistemic value
                    R_te = tf.clip_by_value(R_te, -50.0, 50.0)

                # the nagative EFE value, i.e. the reward. Note the sign here
                R_t = R_ti + self.rho * R_te
                # if self.normalize_signals is True:
                #     a = -self.EFE_bound
                #     b = self.EFE_bound
                #     self.max_R_t = max(self.max_R_t, float(tf.reduce_max(R_t)))
                #     self.min_R_t = min(self.min_R_t, float(tf.reduce_min(R_t)))
                #     R_t = ((R_t - self.min_R_t) * (b - a)) / (self.max_R_t - self.min_R_t) + a

            ## model reconstruction loss ##
            loss_reconst = mse(x_true=obv_next, x_pred=o_next_tran_mu) #, keep_batch=True) #g_nll(obv_next, o_next_mu, o_next_std * o_next_std)
            done = tf.cast(done, dtype=tf.float32)

            # regularization for weights
            loss_l2 = 0.0
            if self.l2_reg > 0.0:
                loss_l2 = tf.add_n([tf.nn.l2_loss(var) for var in self.param_var if 'W' in var.name]) * self.l2_reg
            ## compute full loss ##
            loss_model = loss_reconst + loss_l2
            
            done = tf.expand_dims(done, axis=1)

            if self.use_sum_q is True:
                # take the old EFE values given action indices
                efe_old = tf.math.reduce_sum(efe_t * action, axis=-1)
                with tape.stop_recording():
                    # EFE values for next state, s_t+1 is from transition model instead of encoder
                    efe_target, _, _ = self.efe_target.predict(obv_next)
                    idx_a_next = tf.math.argmax(efe_target, axis=-1, output_type=tf.dtypes.int32)
                    onehot_a_next = tf.one_hot(idx_a_next, depth=self.dim_a)
                    # take the new EFE values
                    efe_new = tf.math.reduce_sum(efe_target * onehot_a_next, axis=-1)
                    y_j = R_t + (efe_new * self.gamma_d) * (1 - done)
            else:
                efe_old = efe_t
                with tape.stop_recording():
                    # EFE values for next state, s_t+1 is from transition model instead of encoder
                    efe_target, _, _ = self.efe_target.predict(obv_next)
                    efe_new = tf.reduce_max(efe_target, axis=1, keepdims=True) # max EFE values at t+1
                    y_j = R_t + (efe_new * self.gamma_d) * (1.0 - done)
                    y_j = (action * y_j) + (efe_old * (1.0 - action))

            ## Temporal Difference loss ##
            if weights is not None:
                loss_efe_batch = None
                if self.efe_loss == "huber":
                    loss_efe_batch = huber(x_true=y_j, x_pred=efe_old, keep_batch=True)
                else:
                    loss_efe_batch = mse(x_true=y_j, x_pred=efe_old, keep_batch=True)
                priorities = tf.math.abs(loss_efe_batch) + 1e-5
                loss_efe_batch = tf.squeeze(loss_efe_batch)
                loss_efe_batch *= weights
                loss_efe = tf.math.reduce_mean(loss_efe_batch)
            else:
                if self.efe_loss == "huber":
                    # loss_efe = huber(x_true=y_j, x_pred=efe_old)
                    loss_efe = tf.reduce_mean(tf.losses.huber(y_j, efe_old))
                else:
                    loss_efe = mse(x_true=y_j, x_pred=efe_old)

        # calculate gradient w.r.t model reconstruction and TD respectively
        grads_model = tape.gradient(loss_model, self.param_var)
        grads_efe = tape.gradient(loss_efe, self.param_var)

        #Ns = obv_t.shape[0]
        if weights is not None:
            return grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, R_te, efe_t, efe_target, priorities

        return grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, R_te, efe_t, efe_target
