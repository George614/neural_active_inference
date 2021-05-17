import tensorflow as tf
import sys
from utils import init_weights, softmax, decide_fun
import numpy as np
import copy

"""
Basic multi-layer perceptron designed to model distributions with up to
two free parameters, i.e., a multivariate Gaussian with mean and variance.

@author: Alex Ororbia
"""
class ProbMLP:
    def __init__(self, name, z_dims, act_fun="tanh", out_fun="identity", init_type="gaussian",
                 wght_sd=0.05, model_variance=False, sigma_fun="softplus", load_dict=None,
                 use_layer_norm=False):
        self.name = name
        self.seed = 69
        self.z_dims = z_dims # [input_dim, n_hid, ...., output_dim]
        self.model_variance = model_variance
        self.use_layer_norm = use_layer_norm

        self.param_axis = 0
        self.param_radius = -1.0 #-10.0 # -10.0 #-5.0
        self.update_radius = -1.0 #5.0 # -1.0 #-10.0 #-1.0 #-5.0 #10.0 #-1.0
        self.eta = 0.05
        if sigma_fun == "softplus":
            self.sigma_fx = tf.math.softplus
        else:
            self.sigma_fx = tf.math.exp
        # set up activation functions
        fx, dfx = decide_fun(act_fun)
        self.dfx = dfx
        self.fx = fx
        fx, dfx = decide_fun(out_fun)
        self.dofx = dfx
        self.ofx = fx

        self.Sigma_W = None
        self.Sigma_b = None
        self.W = [] # W0=null, W1, W2...
        self.b = []
        self.alpha_w = []
        self.beta_b = []
        self.W.append(0.0)
        self.b.append(0.0)
        self.alpha_w.append(0.0)
        self.beta_b.append(0.0)
        self.param_var = []

        if load_dict is not None:
            #print("----")
            cnt = 0
            for item in load_dict.items():
                vname, var = item
                #print(vname)
                #print(var.shape)
                if "mu_b" in vname:
                    var = tf.expand_dims(tf.cast(var,dtype=tf.float32),axis=0)
                    b_l = tf.Variable(var, name="b-mu".format(cnt) )
                    self.mu_b = b_l
                elif "mu_w" in vname:
                    W_l = tf.Variable(tf.cast(var,dtype=tf.float32), name="W-mu".format(cnt) )
                    self.mu_W = W_l
                    #print("LOCK: ",self.Sigma_W.shape)
                elif "std_b" in vname:
                    var = tf.expand_dims(tf.cast(var,dtype=tf.float32),axis=0)
                    b_l = tf.Variable(var, name="b-stddev".format(cnt) )
                    self.Sigma_b = b_l
                elif "std_w" in vname:
                    W_l = tf.Variable(tf.cast(var,dtype=tf.float32), name="W-stddev".format(cnt) )
                    self.Sigma_W = W_l
                    #print("LOCK: ",self.Sigma_W.shape)
                elif "_b" in vname:
                    var = tf.expand_dims(tf.cast(var,dtype=tf.float32),axis=0)
                    b_l = tf.Variable(var, name="b{0}".format(cnt) )
                    self.b.append(b_l)
                elif "_w" in vname:
                    W_l = tf.Variable(tf.cast(var,dtype=tf.float32), name="W{0}".format(cnt) )
                    self.W.append(W_l)

                cnt += 1
            #print("----")
        else:
            for l in range(1, len(z_dims)-1):
                n_z_l = z_dims[l]
                n_z_lm1 = z_dims[l-1]
                W_l = init_weights(init_type, [n_z_lm1, n_z_l], stddev=wght_sd, seed=self.seed)
                W_l = tf.Variable(W_l, name="W{0}".format(l) )
                self.W.append(W_l)
                self.param_var.append(W_l)
                b_l = tf.zeros([1, n_z_l]) #init_weights(init_type, [1, n_z_l], stddev=wght_sd, seed=self.seed)
                b_l = tf.Variable(b_l, name="b{0}".format(l) )
                self.b.append(b_l)
                self.param_var.append(b_l)
                if self.use_layer_norm is True:
                    W_l = tf.ones([1, n_z_l])
                    W_l = tf.Variable(W_l, name="alpha_w{0}".format(l) )
                    self.alpha_w.append(W_l)
                    self.param_var.append(W_l)
                    W_l = tf.zeros([1, n_z_l])
                    W_l = tf.Variable(W_l, name="beta_b{0}".format(l) )
                    self.beta_b.append(W_l)
                    self.param_var.append(W_l)
            n_z_l = z_dims[len(z_dims)-1]
            n_z_lm1 = z_dims[len(z_dims)-2]
            mu_W = init_weights(init_type, [n_z_lm1, n_z_l], stddev=wght_sd, seed=self.seed) #* wght_sd
            mu_W = tf.Variable(mu_W, name="W_mu" )
            self.mu_W = mu_W
            #print(self.name)
            #print(mu_W.shape)
            self.param_var.append(mu_W)
            mu_b = tf.zeros([1, n_z_l]) #init_weights(init_type, [1, n_z_l], stddev=wght_sd, seed=self.seed) * 0 #* wght_sd
            mu_b = tf.Variable(mu_b, name="b_mu" )
            self.mu_b = mu_b
            #print(mu_b.shape)
            self.param_var.append(mu_b)

            if self.model_variance is True:
                n_z_l = z_dims[len(z_dims)-1]
                n_z_lm1 = z_dims[len(z_dims)-2]
                Sigma_W = init_weights(init_type, [n_z_lm1, n_z_l], stddev=wght_sd, seed=self.seed) #* wght_sd
                Sigma_W = tf.Variable(Sigma_W, name="W_sigma" )
                self.Sigma_W = Sigma_W
                self.param_var.append(Sigma_W)
                Sigma_b = tf.zeros([1, n_z_l]) #init_weights(init_type, [1, n_z_l], stddev=wght_sd, seed=self.seed) * 0 #* wght_sd
                Sigma_b = tf.Variable(Sigma_b, name="b_sigma" )
                self.Sigma_b = Sigma_b
                self.param_var.append(Sigma_b)

        self.opt = self.set_optimizer(opt_type="sgd",eta=0.01)

    def extract_params(self):
        param_list = []
        for i in range(len(self.param_var)):
            param_list.append(self.param_var[i])
        return param_list

    def set_optimizer(self, opt_type, eta, momentum=0.9, epsilon=1e-08):
        moment_v = tf.Variable( momentum )
        eta_v  = tf.Variable( eta )
        if opt_type == "nag":
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=eta_v,momentum=moment_v,use_nesterov=True)
        elif opt_type == "momentum":
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=eta_v,momentum=moment_v,use_nesterov=False)
        elif opt_type == "adam":
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=eta_v,beta1=0.9, beta2=0.999, epsilon=epsilon) #1e-08)
        elif opt_type == "rmsprop":
            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=eta_v,decay=0.9, momentum=moment_v, epsilon=1e-10)
        else:
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=eta_v)
        #print(" Opt.Properties: {0} w/ eta = {1}".format(opt_type, alpha))
        self.eta = eta_v #eta
        self.opt = optimizer

    def set_weight_norm(self, param_radius=1.0):
        self.param_radius = param_radius
        self.normalize_weights()

    #@tf.function
    def normalize_weights(self):
        """
            Enforces weight column normalization constraint
        """
        if self.param_radius > 0:
            for l in range(0, len(self.param_var)):
                self.param_var[l].assign( tf.clip_by_norm(self.param_var[l], self.param_radius, axes=[self.param_axis]) )

    def set_weights(self, source, tau=-1): #0.001):
        """
            Deep copies weights of another ncn/ngc model into this model
        """
        #self.param_var = copy.deepcopy(source.param_var)
        if tau >= 0.0:
            for l in range(0, len(self.param_var)):
                self.param_var[l].assign( self.param_var[l] * (1 - tau) + source.param_var[l] * tau )
        else:
            for l in range(0, len(self.param_var)):
                self.param_var[l].assign( source.param_var[l] )

    def view_wnorms(self):
        print()
        print("=====================")
        for l in range(0, len(self.param_var)):
            print(tf.norm(self.param_var[l]))
        print("=====================")

    #@tf.function
    def predict(self, o_t): # predict but do not store activity states
        z_in = o_t
        for l in range(1, len(self.W)):
            z_l = tf.matmul(z_in, self.W[l]) + self.b[l]
            zf_l = self.fx(z_l)
            if self.use_layer_norm is True:
                var_eps = 1e-12
                # apply standardization based on layer normalization
                u = tf.reduce_mean(zf_l, keepdims=True)
                s = tf.reduce_mean(tf.pow(zf_l - u, 2), axis=-1, keepdims=True)
                zf_l = (zf_l - u) / tf.sqrt(s + var_eps)
                # apply layer normalization re-scaling
                zf_l = tf.multiply(self.alpha_w[l], zf_l) + self.beta_b[l]
            # apply post-activation

            z_in = zf_l
        # apply top-most layer -- (output) mean and variance calculation
        mu = self.ofx( tf.matmul(z_in, self.mu_W) + self.mu_b )
        sigma = tf.cast(0.1,dtype=tf.float32)
        log_sigma = tf.math.log(sigma)
        if self.model_variance is True:
            log_sigma = tf.matmul(z_in, self.Sigma_W) + self.Sigma_b
            sigma = self.sigma_fx(log_sigma) #tf.math.exp(log_sigma)
            sigma = tf.clip_by_value(sigma, 0.01, 10.0)
        return mu, sigma, log_sigma

    #@tf.function
    def update_params(self, delta_list, update_radius=-1.):
        if update_radius > 0.0:
            for d in range(len(delta_list)):
                delta = delta_list[d]
                if self.update_radius > 0.0:
                    #delta = tf.clip_by_value(delta, -self.update_radius,self.update_radius)
                    delta = tf.clip_by_norm(delta, self.update_radius)
                delta_list[d] = delta
        # apply (negative) synaptic delta matrices according to an optimization rule, e.g., SGD
        self.opt.apply_gradients(zip(delta_list, self.param_var))

        self.normalize_weights()

        return delta_list

    def clear_memory(self):
        pass
