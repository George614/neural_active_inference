"""
Utilities function file

@author: Alex Ororbia, Ankur Mali
"""
import tensorflow as tf
import numpy as np
import pickle
seed = 69
#tf.random.set_random_seed(seed=seed)

tf.random.set_seed(seed=seed)
np.random.seed(seed)

def save_object(model, fname):
    fd = open(fname, 'wb')
    pickle.dump(model, fd)
    fd.close()

def load_object(fname):
    fd = open(fname, 'rb')
    model = pickle.load( fd )
    fd.close()
    return model

def parse_int_list(str_arg):
    tok = str_arg.replace("[","").replace("]","").split(",")
    int_list = []
    for i in range(len(tok)):
        int_list.append(int(tok[i]))
    return int_list

def expand_action_paths(self, a_dim, H=1, Ks=-1):
    set = []
    for d in range(H):
        set_new = []
        if d > 0:
            for pi in set:
                if Ks > 0:
                    s_paths = np.random.permutation(a_dim)
                    for i in range(Ks):
                        a_k = s_paths[i]
                        pi_new = pi + [a_k]
                        set_new.append(pi_new)
                else:
                    for a_k in range(a_dim):
                        pi_new = pi + [a_k]
                        set_new.append(pi_new)
        else:
            for a_k in range(a_dim):
                pi_new = [a_k]
                set_new.append(pi_new)
        set = set_new
    return set

def init_weights(init_type, shape, seed, stddev=1.0):
    if init_type == "he_uniform":
        initializer = tf.compat.v1.keras.initializers.he_uniform()
        params = initializer(shape) #, seed=seed )
    elif init_type == "he_normal":
        initializer = tf.compat.v1.keras.initializers.he_normal()
        params = initializer(shape) #, seed=seed )
    elif init_type == "classic_glorot":
        N = (shape[0] + shape[1]) * 1.0
        bound = 4.0 * np.sqrt(6.0/N)
        params = tf.random.uniform(shape, minval=-bound, maxval=bound, seed=seed)
    elif init_type == "glorot_normal":
        initializer = tf.compat.v1.keras.initializers.glorot_normal()
        params = initializer(shape) #, seed=seed )
    elif init_type == "glorot_uniform":
        initializer = tf.compat.v1.keras.initializers.glorot_uniform()
        params = initializer(shape) #, seed=seed )
    elif init_type == "orthogonal":
        initializer = tf.compat.v1.keras.initializers.orthogonal(gain=stddev)
        params = initializer(shape)
    elif init_type == "truncated_normal":
        params = tf.random.truncated_normal(shape, stddev=stddev, seed=seed)
    elif init_type == "normal":
        params = tf.random.normal(shape, stddev=stddev, seed=seed)
    else: # alex_uniform
        k = 1.0 / (shape[0] * 1.0) # 1/in_features
        bound = np.sqrt(k)
        params = tf.random.uniform(shape, minval=-bound, maxval=bound, seed=seed)

    return params

def decide_fun(fun_type):
    fx = None
    d_fx = None
    if fun_type == "tanh":
        fx = tf.nn.tanh
        d_fx = d_tanh
    elif fun_type == "sign":
        fx = tf.math.sign
        d_fx = d_identity
    elif fun_type == "clip_fx":
        fx = clip_fx
        d_fx = d_identity
    elif fun_type == "ltanh":
        fx = ltanh
        d_fx = d_ltanh
    elif fun_type == "selu":
        fx = tf.nn.selu
        d_fx = d_identity
    elif fun_type == "elu":
        fx = tf.nn.elu
        d_fx = d_identity
    elif fun_type == "erf":
        fx = tf.math.erf
        d_fx = d_identity
    elif fun_type == "lrelu":
        fx = tf.nn.leaky_relu
        d_fx = d_identity
    elif fun_type == "relu":
        fx = tf.nn.relu
        d_fx = d_relu
    elif fun_type == "softsign":
        fx = tf.nn.softsign
        d_fx = d_identity
    elif fun_type == "softplus":
        fx = tf.math.softplus
        d_fx = d_softplus
    elif fun_type == "relu6":
        fx = tf.nn.relu6
        d_fx = d_relu6
    elif fun_type == "sigmoid":
        fx = tf.nn.sigmoid
        d_fx = d_sigmoid
    elif fun_type == "kwta":
        fx = kwta
        d_fx = bkwta #d_identity
    elif fun_type == "softmax":
        fx = softmax
        d_fx = tf.identity
    else:
        fx = tf.identity
        d_fx = d_identity
    return fx, d_fx

def identity(z):
    return z

def d_identity(x):
    return x * 0 + 1.0

def ltanh(z):
    a = 1.7159
    b = 2.0/3.0
    z_scale = z * b
    z_scale = tf.clip_by_value(z_scale, -50.0, 50.0) #-85.0, 85.0)
    neg_exp = tf.exp(-z_scale)
    pos_exp = tf.exp(z_scale)
    denom = tf.add(pos_exp, neg_exp)
    numer = tf.subtract(pos_exp, neg_exp)
    return tf.math.divide(numer, denom) * a

def d_ltanh(z):
    a = 1.7159
    b = 2.0/3.0
    z_scale = z * b
    z_scale = tf.clip_by_value(z_scale, -50.0, 50.0) #-85.0, 85.0)
    neg_exp = tf.exp(-z_scale)
    pos_exp = tf.exp(z_scale)
    denom = tf.add(pos_exp, neg_exp)
    dx = tf.math.divide((4.0 * a * b), denom * denom)
    return dx
    
def elu(z,alpha=1.0):
    return z if z >= 0 else (tf.math.exp(z) - 1.0) * alpha

def d_elu(z,alpha=1.0):
	return 1 if z > 0 else tf.math.exp(z) * alpha

def d_sigmoid(x):
    sigm_x = tf.nn.sigmoid(x)
    return (-sigm_x + 1.0) * sigm_x

def d_tanh(x):
    tanh_x = tf.nn.tanh(x)
    return -(tanh_x * tanh_x) + 1.0

def d_relu(x):
    # df/dx = 1 if 0<x<6 else 0
    val = tf.math.greater_equal(x, 0.0)
    return tf.cast(val,dtype=tf.float32) # sign(max(0,x))

def d_relu6(x):
    # df/dx = 1 if 0<x<6 else 0
    # I_x = (z >= a_min) *@ (z <= b_max) //create an indicator function  a = 0 b = 6
    Ix1 = tf.cast(tf.math.greater_equal(x, 0.0),dtype=tf.float32)
    Ix2 = tf.cast(tf.math.less_equal(x, 6.0),dtype=tf.float32)
    Ix = Ix1 * Ix2
    return Ix

def d_softplus(x):
    return tf.nn.sigmoid(x) # d/dx of softplus = logistic sigmoid

def softmax(x, tau=0.0):
    """
        Softmax function with overflow control built in directly. Contains optional
        temperature parameter to control sharpness (tau > 1 softens probs, < 1 sharpens --> 0 yields point-mass)
    """
    if tau > 0.0:
        x = x / tau
    max_x = tf.expand_dims( tf.reduce_max(x, axis=1), axis=1)
    exp_x = tf.exp(tf.subtract(x, max_x))
    return exp_x / tf.expand_dims( tf.reduce_sum(exp_x, axis=1), axis=1)

def mellowmax(x, omega=1.0,axis=1):
    n = x.shape[axis] * 1.0
    #(F.logsumexp(omega * values, axis=axis) - np.log(n)) / omega
    return ( tf.reduce_logsumexp(x * omega, axis=axis, keepdims=True) - tf.math.log(n) ) / omega

def clip_fx(x):
    return tf.clip_by_value(x, 0.0, 1.0)

def drop_out(input, rate=0.0, seed=69):
    """
        Custom drop-out function -- returns output as well as binary mask
    """
    mask = tf.math.less_equal( tf.random.uniform(shape=(input.shape[0],input.shape[1]), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed),(1.0 - rate))
    mask = tf.cast(mask, tf.float32) * (1.0 / (1.0 - rate))
    output = input * mask
    return output, mask

def calc_cos_sim(x_i, x_j):
    ni = tf.norm(x_i)
    nj = tf.norm(x_j)
    dot_prod = tf.matmul(x_i, x_j,transpose_b=True)
    return dot_prod / (ni * nj)

def calc_dist(x_i, x_j, unit_space=False):
    a = x_i
    if unit_space is True:
        a = a / (tf.norm(a) + 1e-6)
    b = x_j
    if unit_space is True:
        b = b / (tf.norm(b) + 1e-6)
    euclid_dist = tf.norm(a - b)
    return euclid_dist

def scale_feat(x, a=-1.0, b=1.0):
    max_x = tf.reduce_max(x,axis=1,keepdims=True)
    min_x = tf.reduce_min(x,axis=1,keepdims=True)
    x_prime = a + ( ( (x - min_x) * (b - a) )/(max_x - min_x) )
    return tf.cast(x_prime, dtype=tf.float32)

def squash(x, t_min=-1, t_max=1):
    max_x = float(tf.reduce_max(x))
    min_x = float(tf.reduce_min(x))
    x_prime = (x - min_x)/(max_x - min_x)
    #x_prime = a + ( ( (x - min_x) * (b - a) )/(max_x - min_x) )
    return x_prime * (t_max - t_min) + t_min

def unsquash(x, t_min, t_max):
    x_prime = x * (t_max - t_min) + t_min
    return x_prime

def binarize(x, is_bipolar=False, thr=0.5):#thr=0.0
    """Converts real-valued vector(s) x to binary encoding (polar or bipolar)"""
    x = tf.greater(x, thr)
    x = tf.cast(x, dtype=tf.float32)
    if is_bipolar is True: # convert to bipolar binary encoding
        x = x * 2 - 1.0 # values that are 2 will become 1 and values at 0 will become -1
    return x

def sample_uniform(n_s, n_dim):
    eps = tf.random.uniform(shape=(n_s,n_dim), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)
    return eps

def kl_div_loss_analytically_from_logvar(mu1, logvar1, mu2, logvar2):
    #D = mu1.shape[1] * 1.0
    eps = 1e-6
    return 0.5*(logvar2 - logvar1) + (tf.exp(logvar1) + tf.math.square(mu1 - mu2)) / (2.0 * tf.exp(logvar2) + eps) - 0.5

def kl_div_loss(mu1, logvar1, mu2, logvar2, axis=1):
    return tf.reduce_sum(kl_div_loss_analytically_from_logvar(mu1, logvar1, mu2, logvar2), axis)

def kl_div_unitPrior(mu, log_sigma):
    #D = mu.shape[1] * 1.0
    sigma = tf.exp(log_sigma)
    unit_kl = (1/2) * (
        tf.reduce_sum(sigma, axis=-1, keepdims=True) + \
        tf.reduce_sum(mu**2, axis=-1, keepdims=True) - \
        1.0 - \
        tf.reduce_sum(log_sigma, axis=-1, keepdims=True)
    )
    return unit_kl

log_2_pi_e = np.log(2.0*np.pi*np.e)

def entropy_gaussian_from_logvar(logvar):
    return 0.5*(log_2_pi_e + logvar)

def sample_gaussian_with_logvar(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean # exp(logvar * 0.5) = standard deviation

def g_nll_from_logvar(X, mu, logvar, keep_batch=False):
    """ Gaussian Negative Log Likelihood loss function
        --> assumes N(x; mu, log(sig^2))

        mu <- external mean
        log_var <- pre-computed log(sigma^2) (numerically stable form)
        keep_batch <-
    """
    ############################################################################
    # Alex-style - avoid logarithms whenever you can pre-compute them
    # I like the variance-form of GNLL, I find it generally to be more stable
    sigSqr = tf.exp(logvar)
    eps = 1e-6
    diff = X - mu # pre-compute this quantity
    term1 = -( (diff * diff)/(sigSqr *2 + eps) ) # central term
    term2 = -logvar * 0.5 # numerically more stable form of this term
    #term3 = -tf.math.log(np.pi * 2) * 0.5 # constant term
    #nll = -( term1 + term2 + term3 ) # -( LL ) = NLL
    nll = -( term1 + term2 ) #
    nll = tf.reduce_sum(nll, axis=-1) # gets GNLL per sample in batch (a column vector)
    ############################################################################
    if not keep_batch:
        nll = tf.reduce_mean(nll)
    else:
        nll = tf.expand_dims(nll,axis=1)
    return nll

def sample_gaussian(n_s, mu=0.0, sig=1.0, n_dim=-1):
    """
        Samples a multivariate Gaussian assuming at worst a diagonal covariance
    """
    dim = n_dim
    if dim <= 0:
        dim = mu.shape[1]
    eps = tf.random.normal([n_s, dim], mean=0.0, stddev=1.0, seed=seed)
    return mu + eps * sig

def squared_error_vec(x_reconst, x_true):
    """ Squared error dimension-wise """
    ############################################################################
    # Alex-style -- avoid complex operators like pow(.) whenever you can, below
    #               is the same as above
    ############################################################################
    diff = x_reconst - x_true
    se = diff * diff # squared error
    ############################################################################
    return se

def huber_Alex(x_true, x_pred, keep_batch=False):
    ''' Huber Loss (smooth L1 loss) '''
    diff = x_pred - x_true
    beta = 1.0
    # compute Huber loss
    abs_diff = tf.math.abs(diff)
    mask = tf.cast( tf.math.less(abs_diff, beta) ,dtype=tf.float32)
    term1 = (diff * diff) * 0.5/beta
    term2 = abs_diff - 0.5 * beta
    v_loss = term1 * mask + term2 * (1 - mask)
    #huber = tf.reduce_sum(v_loss, keepdims=True) * (1.0/(x_true.shape[0] * 1.0))
    if not keep_batch:
        huber = tf.math.reduce_mean(v_loss)
    else:
        huber = tf.reduce_sum(v_loss, axis=-1)
        huber = tf.expand_dims(huber,axis=1)
    return huber

def huber(x_true, x_pred, delta=1.0, keep_batch=False):
    error = x_true - x_pred
    within_d = tf.math.less_equal(tf.abs(error), delta)
    within_d = tf.cast(within_d, dtype=tf.float32)
    loss_in = 0.5 * error * error
    loss_out = 0.5 * delta * delta + delta * (tf.abs(error) - delta)
    loss = within_d * loss_in + (1 - within_d) * loss_out
    if tf.greater(tf.rank(loss), 1):
        loss = tf.reduce_sum(loss, axis=-1)
    if keep_batch:
        return loss
    return tf.math.reduce_mean(loss, axis=0)

def aleatoric_loss(y_true, y_mu, y_log_var):
    N = y_true.shape[0]
    diff = (y_true-y_mu)
    se = diff * diff
    inv_std = tf.math.exp(-y_log_var)
    mse = tf.reduce_mean(inv_std*se)
    reg = tf.reduce_mean(y_log_var)
    return 0.5*(mse + reg)

def mse(x_true, x_pred, keep_batch=False):
    ''' Mean Squared Error '''
    diff = x_pred - x_true
    se = diff * diff # squared error
    # NLL = -( -se )
    if not keep_batch:
        mse = tf.math.reduce_mean(se)
    else:
        mse = tf.reduce_sum(se, axis=-1)
        mse = tf.expand_dims(mse, axis=1)
    return mse

def kl_d(mu_p, sigSqr_p, log_sig_p, mu_q, sigSqr_q, log_sig_q, keep_batch=False):
    """
         Kullback-Leibler (KL) Divergence function for 2 multivariate Gaussian distributions
         strictly assuming diagonal covariances (this assumption allows for a formulation
         that only uses simple functions that are friendly for reverse-mode differnetation).

         Follows formula, where p(x) = N(x ; mu_p,sig^2_p) and q(x) = N(x ; mu_q,sig^2_q):
         KL(p||q) = log(sig_q/sig_p) + (sig^2_p + (mu_p - mu_q)^2)/(2 * sig^2_q) - 1/2
                  = [ log(sig_q) - log(sig_p) ] + (sig^2_p + (mu_p - mu_q)^2)/(2 * sig^2_q) - 1/2
    """
    ############################################################################
    # Alex-style - avoid logarithms whenever you can pre-compute them
    # I like the variance-form of G-KL, I find it generally to be more stable
    # Note that I expanded the formula a bit further using log difference rule
    ############################################################################
    eps = 1e-6
    term1 = log_sig_q - log_sig_p
    diff = mu_p - mu_q
    term2 = (sigSqr_p + (diff * diff))/(sigSqr_q * 2 + eps)
    KLD = term1 + term2 - 1/2
    KLD = tf.reduce_sum(KLD, axis=-1) #gets KL per sample in batch (a column vector)
    ############################################################################
    if not keep_batch:
        KLD = tf.math.reduce_mean(KLD)
    else:
        KLD = tf.expand_dims(KLD,axis=1)
    return KLD

def g_nll(X, mu, sigSqr, log_sig=None, keep_batch=False):
    """ Gaussian Negative Log Likelihood loss function
        --> assumes N(x; mu, sig^2)

        mu <- external mean
        sigSqr <- external variance
        log_sig <- pre-computed log(sigma) (numerically stable form)
        keep_batch <-
    """
    ############################################################################
    # Alex-style - avoid logarithms whenever you can pre-compute them
    # I like the variance-form of GNLL, I find it generally to be more stable
    eps = 1e-6
    diff = X - mu # pre-compute this quantity
    term1 = -( (diff * diff)/(sigSqr *2 + eps) ) # central term
    if log_sig is not None:
        # expanded out the log(sigma * sigma) = log(sigma) + log(sigma)
        term2 = -(log_sig + log_sig) * 0.5 # numerically more stable form
    else:
        term2 = -tf.math.log(sigSqr) * 0.5
    term3 = -tf.math.log(np.pi * 2) * 0.5 # constant term
    nll = -( term1 + term2 + term3 ) # -( LL ) = NLL
    nll = tf.reduce_sum(nll, axis=-1) # gets GNLL per sample in batch (a column vector)
    ############################################################################
    if not keep_batch:
        nll = tf.reduce_mean(nll)
    else:
        nll = tf.expand_dims(nll,axis=1)
    return nll

def calc_gaussian_LL(x, mu, sigSqr, log_sigSqr=None):
    """
        Calculates the Gaussian log likelihood for particular x
    """
    eps = 1e-5 # controls for numerical instability
    N = x.shape[0]
    delta = (x - mu)
    dist_term = tf.reduce_sum( -(delta * delta)/(sigSqr * 2 + eps) ,axis=0,keepdims=True)
    var_term = tf.math.log(sigSqr + eps) * (-0.5 * N)
    const_term = tf.math.log(2 * np.pi) * (-0.5 * N)
    # collapse along dimensions
    log_likeli = tf.reduce_sum( dist_term + var_term, axis=1, keepdims=True) + const_term # compose full log likelihood
    return log_likeli

def calc_gaussian_entropy(mu, sigSqr):
    """
        Calculates differential entropy of multivariate Gaussian, assuming diagonal covariance.

         Note: https://sgfin.github.io/2017/03/11/Deriving-the-information-entropy-of-the-multivariate-gaussian/
         Note: https://proofwiki.org/wiki/Determinant_of_Diagonal_Matrix

         Worked-out formula for entropy of multivariate Gaussian is:
         H = D/2*(1 + ln(2*pi)) + 1/2*ln det(Sigma)  // where det(.) is the determinant operator
         where Sigma is diagonal covariance matrix, allowing us to simplify further to:
         H = D/2*(1 + ln(2*pi)) + 1/2*ln( sig^2_1 * sig^2_2 * ... * sig^2_D )
         H = D/2*(1 + ln(2*pi)) + 1/2*( ln(sig^2_1) + ln(sig^2_2) + ... + ln(sig^2_D) )
         @author Alex Ororbia
    """
    eps = 1e-5
    #D = mu.shape[1] # dimensionality of data
    #diff_ent = (D * 0.5) * (1.0 + tf.math.log(2.0 * np.pi)) + tf.reduce_sum(tf.math.log(sigSqr + eps),axis=1,keepdims=True) * 0.5
    #return diff_ent
    return 0.5 + 0.5 * tf.math.log(2 * np.pi) + tf.reduce_sum( tf.math.log(sigSqr + eps),axis=1,keepdims=True )

def calc_gaussian_KL(mu1, sigSqr1, mu2, sigSqr2):
    """
        Calculates Kullback-Leibler (KL) divergence between two multivariate (diagonal covariance) Gaussians

        KL = log(sig2/sig1) + (sig1^2 + (mu1 - mu2)^2)/(2 * sig2^2) - 1/2
           = log(sig2) - log(sig1) + (sig1^2 + (mu1 - mu2)^2)/(2 * sig2^2) - 1/2
    """
    eps = 1e-5
    sig1 = tf.math.sqrt(sigSqr1)
    sig2 = tf.math.sqrt(sigSqr2)
    delta_mu = (mu1 - mu2)
    KL = (tf.math.log(sig2 + eps) - tf.math.log(sig1 + eps)) + (sigSqr1 + (delta_mu * delta_mu) )/(sigSqr2 * 2 + eps) - 0.5
    return tf.reduce_sum(KL,axis=1,keepdims=True)


def fast_log_loss(probs, y_ind_):
    loss = 0.0
    y_ind = tf.expand_dims(y_ind_, 1)
    py = probs.numpy()
    for i in range(0, y_ind.shape[0]):
        ti = y_ind[i,0] # get ith target in sequence
        if ti >= 0: # entry for masked token, which should be non-negative
            py = probs[i,ti]
            if py <= 0.0:
                py = 1e-8
            loss += np.log(py) # all other columns in row i ( != ti) are 0, so do nothing
    return -loss # return negative summed log probs

def neg_cat_loglikeli(y_pred, y_true, epsilon=1e-8, msk=None):
    '''
    negative Categorical log likelihood
    y_pred - Tensor of shape (batch_size, size_output)
    y_true - Tensor of shape (batch_size, size_output)
    '''
    N = y_true.shape[0] * 1.0
    nll = -tf.reduce_sum( (tf.math.log(tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon))) * y_true ) / N
    return nll

def bce_w_logits(p_logits, x):
    '''
        Binary cross entropy (BCE) but simplified to logit-form, i.e., L(p_logits, x_target)
    '''
    max_val = tf.nn.relu(-p_logits)
    loss = p_logits - p_logits * x + max_val + tf.math.log( tf.math.exp(-max_val) + tf.math.exp(-p_logits - max_val) )
    return loss
