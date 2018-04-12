import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import convolution2d as conv
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.layers import xavier_initializer as xavier
from collections import deque, namedtuple

####################################################################################################
########################################### Core modules ###########################################
####################################################################################################

def conv_module(input_layer, convs, activation_fn=tf.nn.relu):
    """ convolutional module
    """
    out = input_layer
    for num_outputs, kernel_size, stride in convs:
        out = conv(out,
                   num_outputs=num_outputs,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding='VALID',
                   activation_fn=activation_fn)
    return out

def fc_module(input_layer, fully_connected, activation_fn=tf.nn.relu):
    """ fully connected module
    """
    out = input_layer
    for num_outputs in fully_connected:
        out = fc(out,
                 num_outputs=num_outputs,
                 activation_fn=activation_fn,
                 weights_initializer=xavier())
    return out

def func_module(input_layer, num_inputs, num_outputs):
    """ final module which estimates some function (value, q, policy, etc)
    """
    out = input_layer
    out_weights = tf.Variable(xavier()([num_inputs, num_outputs]))
    out = tf.matmul(out, out_weights)
    return out

def full_module(input_layer, convs, fully_connected, num_outputs,
                        activation_fn=tf.nn.relu):
    """ convolutional + fully connected + functional module
    """
    out = input_layer
    out = conv_module(out, convs, activation_fn)
    out = layers.flatten(out)
    out = fc_module(out, fully_connected, activation_fn)
    out = func_module(out, fully_connected[-1], num_outputs)
    return out

####################################################################################################
########################################## Deep Q-Network ##########################################
####################################################################################################

class DeepQNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 activation_fn=tf.nn.relu,
                 scope="dqn", reuse=False):

        with tf.variable_scope(scope, reuse=reuse):

            ###################### Neural network architecture ######################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

            self.q_values = full_module(self.input_states, convs, fully_connected,
                                        num_actions, activation_fn)

            ######################### Optimization procedure ########################

            # one-hot encode actions to get q-values for state-action pairs
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            actions_onehot = tf.one_hot(self.input_actions, num_actions, dtype=tf.float32)
            q_values_selected = tf.reduce_sum(tf.multiply(self.q_values, actions_onehot), axis=1)

            # choose best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            # create loss function and update rule
            self.targets = tf.placeholder(dtype=tf.float32, shape=[None])
            self.td_error = tf.losses.huber_loss(self.targets, q_values_selected)
            self.loss = tf.reduce_sum(self.td_error)
            self.update_model = optimizer.minimize(self.loss)

    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax

    def get_q_values(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values

    def update(self, sess, states, actions, targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.targets:targets}
        sess.run(self.update_model, feed_dict)

####################################################################################################
###################################### Dueling Deep Q-Network ######################################
####################################################################################################

class DuelingDeepQNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[64],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 activation_fn=tf.nn.relu,
                 scope="duel_dqn", reuse=False):

        with tf.variable_scope(scope, reuse=reuse):

            ###################### Neural network architecture ######################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

            out = conv_module(self.input_states, convs, activation_fn)
            val, adv = tf.split(out, num_or_size_splits=2, axis=3)

            with tf.variable_scope(scope+"/value", reuse=reuse):
                val = layers.flatten(val)
                val = fc_module(val, fully_connected, activation_fn)
                self.v_values = func_module(val, fully_connected[-1], 1)

            with tf.variable_scope(scope+"/advantage", reuse=reuse):
                adv = layers.flatten(adv)
                adv = fc_module(adv, fully_connected, activation_fn)
                self.a_values = func_module(adv, fully_connected[-1], num_actions)

            a_values_mean = tf.reduce_mean(self.a_values, axis=1, keepdims=True)
            a_values_centered = tf.subtract(self.a_values, a_values_mean)
            self.q_values = self.v_values + a_values_centered

            ######################### Optimization procedure ########################

            # one-hot encode actions to get q-values for state-action pairs
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            actions_onehot = tf.one_hot(self.input_actions, num_actions, dtype=tf.float32)
            q_values_selected = tf.reduce_sum(tf.multiply(self.q_values, actions_onehot), axis=1)

            # choose best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            # create loss function and update rule
            self.targets = tf.placeholder(dtype=tf.float32, shape=[None])
            self.td_error = tf.losses.huber_loss(self.targets, q_values_selected)
            self.loss = tf.reduce_sum(self.td_error)
            self.update_model = optimizer.minimize(self.loss)

    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax

    def get_q_values(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values

    def update(self, sess, states, actions, targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.targets:targets}
        sess.run(self.update_model, feed_dict)

####################################################################################################
#################################### Categorical Deep Q-Network ####################################
####################################################################################################

class CategoricalDeepQNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128],
                 num_atoms=21, v=(-10, 10),
                 activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 scope="cat_dqn", reuse=False):

        with tf.variable_scope(scope, reuse=reuse):

            ###################### Neural network architecture ######################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

            # distribution parameters
            self.num_atoms = num_atoms
            self.v_min, self.v_max = v
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]
            self.tensor_z = tf.convert_to_tensor(self.z, dtype=tf.float32)

            out = conv_module(self.input_states, convs, activation_fn)
            out = layers.flatten(out)
            out = fc_module(out, fully_connected, activation_fn)
            out = fc_module(out, [num_actions * self.num_atoms], None)
            self.logits = tf.reshape(out, shape=[-1, num_actions, self.num_atoms])
            self.probs = tf.nn.softmax(self.logits, axis=2)
            self.q_values = tf.reduce_sum(tf.multiply(self.probs, self.tensor_z), axis=2)

            ######################### Optimization procedure ########################

            # one-hot encode actions to get q-values for state-action pairs
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            actions_onehot = tf.one_hot(self.input_actions, num_actions, dtype=tf.float32)
            actions_onehot_reshaped = tf.reshape(actions_onehot, [-1, num_actions, 1])
            q_values_selected = tf.reduce_sum(tf.multiply(self.q_values, actions_onehot), axis=1)

            # choose best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            probs_selected = tf.multiply(self.probs, actions_onehot_reshaped)
            self.probs_selected = tf.reduce_sum(probs_selected, axis=1)

            # create loss function and update rule
            self.targets = tf.placeholder(dtype=tf.float32, shape=[None, self.num_atoms])
            self.loss = -tf.reduce_sum(self.targets * tf.log(self.probs_selected + 1e-6))
            self.update_model = optimizer.minimize(self.loss)

    def get_probs(self, sess, states):
        feed_dict = {self.input_states:states}
        probs = sess.run(self.probs, feed_dict)
        return probs

    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax

    def get_q_values(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values

    def update(self, sess, states, actions, targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.targets:targets}
        sess.run(self.update_model, feed_dict)

    def cat_proj(self, sess, rewards, states_, actions_, end, gamma=0.99):
        """
        Categorical algorithm from https://arxiv.org/abs/1707.06887
        """

        feed_dict = {self.input_states:states_, self.input_actions:actions_}
        probs = sess.run(self.probs_selected, feed_dict=feed_dict)
        m = np.zeros_like(probs)
        rewards = np.array(rewards, dtype=np.float32)
        end = np.array(end, dtype=np.float32)
        batch_size = rewards.size

        for j in range(self.num_atoms):
            Tz = rewards + gamma * end * self.z[j]
            Tz = np.minimum(self.v_max, np.maximum(self.v_min, Tz))
            b = (Tz - self.v_min) / self.delta_z
            l = np.floor(b)
            u = np.ceil(b)
            m[np.arange(batch_size), l.astype(int)] += probs[:,j] * (u - b)
            m[np.arange(batch_size), u.astype(int)] += probs[:,j] * (b - l)

        return m

####################################################################################################
######################################## Soft Actor-Critic #########################################
####################################################################################################

class SoftActorCriticNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128],
                 activation_fn=tf.nn.relu,
                 optimizers=[tf.train.AdamOptimizer(2.5e-4),
                             tf.train.AdamOptimizer(2.5e-4),
                             tf.train.AdamOptimizer(2.5e-4)],
                 scope="sac", reuse=False):

        with tf.variable_scope(scope, reuse=reuse):

            ###################### Neural network architecture ######################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

            with tf.variable_scope("value", reuse=reuse):
                self.v_values = full_module(self.input_states, convs, fully_connected,
                                            1, activation_fn)

            with tf.variable_scope("qfunc", reuse=reuse):
                self.q_values = full_module(self.input_states, convs, fully_connected,
                                            num_actions, activation_fn)

            with tf.variable_scope("policy", reuse=reuse):
                self.p_logits = full_module(self.input_states, convs, fully_connected,
                                            num_actions, activation_fn)
            self.p_values = layers.softmax(self.p_logits)

            ######################### Optimization procedure ########################

            # one-hot encode actions to get q-values for state-action pairs
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            actions_onehot = tf.one_hot(self.input_actions, num_actions, dtype=tf.float32)
            q_values_selected = tf.reduce_sum(tf.multiply(self.q_values, actions_onehot), axis=1)
            p_logits_selected = tf.reduce_sum(tf.multiply(self.p_logits, actions_onehot), axis=1)

            # choose best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            # create loss function and update rule
            self.q_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            self.v_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            self.p_targets = tf.placeholder(dtype=tf.float32, shape=[None])

            q_loss = tf.losses.huber_loss(self.q_targets, q_values_selected)
            self.q_loss = tf.reduce_sum(q_loss)
            q_optimizer = optimizers[0]

            v_targets_reshaped = tf.reshape(self.v_targets, (-1, 1))
            v_loss = tf.losses.huber_loss(v_targets_reshaped, self.v_values)
            self.v_loss = tf.reduce_sum(v_loss)
            v_optimizer = optimizers[1]

            p_loss = tf.losses.huber_loss(self.p_targets, p_logits_selected)
            self.p_loss = tf.reduce_sum(p_loss)
            p_optimizer = optimizers[2]

            self.update_q_values = q_optimizer.minimize(self.q_loss)
            self.update_v_values = v_optimizer.minimize(self.v_loss)
            self.update_p_logits = p_optimizer.minimize(self.p_loss)

    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax

    def get_q_values(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values

    def get_v_values(self, sess, states):
        feed_dict = {self.input_states:states}
        v_values = sess.run(self.v_values, feed_dict)
        return v_values

    def get_p_logits(self, sess, states):
        feed_dict = {self.input_states:states}
        p_logits = sess.run(self.p_logits, feed_dict)
        return p_logits

    def get_p_values(self, sess, states):
        feed_dict = {self.input_states:states}
        p_values = sess.run(self.p_values, feed_dict)
        return p_values

    def update_q(self, sess, states, actions, q_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.q_targets:q_targets}
        sess.run(self.update_q_values, feed_dict)

    def update_v(self, sess, states, v_targets):

        feed_dict = {self.input_states:states,
                     self.v_targets:v_targets}
        sess.run(self.update_v_values, feed_dict)

    def update_p(self, sess, states, actions, p_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.p_targets:p_targets}
        sess.run(self.update_p_logits, feed_dict)

####################################################################################################
########################################## Deep Q-Network ##########################################
####################################################################################################

class GaussianDeepQNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 activation_fn=tf.nn.relu,
                 scope="gaussian_boi", loss='wasserstein', reuse=False):

        ###################### Neural network architecture ######################

        input_shape = [None] + state_shape
        self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

        with tf.variable_scope(scope, reuse=reuse):
            out = conv_module(self.input_states, convs, activation_fn)
            out = layers.flatten(out)
            out = fc_module(out, fully_connected, activation_fn)
            out = fc_module(out, [num_actions * 2], None)
            self.mu, sigma = tf.split(out, num_or_size_splits=2, axis=1)
            self.sigma = 0.05 + 3 * tf.nn.sigmoid(sigma)

            ######################### Optimization procedure ########################

            # one-hot encode actions to get q-values for state-action pairs
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            actions_onehot = tf.one_hot(self.input_actions, num_actions, dtype=tf.float32)
            mu_selected = tf.reduce_sum(tf.multiply(self.mu, actions_onehot), axis=1)
            sigma_selected = tf.reduce_sum(tf.multiply(self.sigma, actions_onehot), axis=1)

            # choose best actions (according to q-values)
            self.q_argmax = tf.argmax(self.mu, axis=1)

            # create loss function and update rule
            self.mu_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            

            self.sigma_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            

#             #loss_term1 = tf.log(tf.div(sigma_selected, (self.sigma_targets + 1e-6)))
#             loss_term1 = tf.subtract(tf.log(sigma_selected), tf.log(self.sigma_targets))
#             mu_diff = tf.square(tf.subtract(self.mu_targets, mu_selected))
#             nom = tf.square(self.sigma_targets) + mu_diff
#             denom = 2*tf.square(sigma_selected) + 1e-9
#             loss_term2 = tf.div(nom, denom)
            #self.loss = tf.reduce_sum(loss_term1 + loss_term2)
            
            if loss == 'wasserstein':
                mu_loss = tf.losses.mean_squared_error(self.mu_targets, mu_selected)
                sigma_loss = tf.losses.mean_squared_error(self.sigma_targets, sigma_selected)
                self.loss = mu_loss + sigma_loss
            
            if loss == 'kl':
                loss_term1 = tf.subtract(tf.log(sigma_selected), tf.log(self.sigma_targets))
                mu_diff = tf.square(tf.subtract(self.mu_targets, mu_selected))
                nom = tf.square(self.sigma_targets) + mu_diff
                denom = 2*tf.square(sigma_selected) + 1e-6
                loss_term2 = tf.div(nom, denom)  
                
                self.loss = tf.reduce_sum(loss_term1 + loss_term2)
                
            if loss == 'symmetric_kl':
                term1 = tf.square(sigma_selected) + tf.square(tf.subtract(self.mu_targets, mu_selected))
                term1 = tf.div(term1, 2 * tf.square(self.sigma_targets) + 1e-6)
                
                term2 = tf.square(self.sigma_targets) + tf.square(tf.subtract(self.mu_targets, mu_selected))
                
                term2 = tf.div(term2, 2 * tf.square(sigma_selected) + 1e-6)
                
                self.loss = tf.reduce_sum(term1 + term2)
                
                
                
            
            #self.loss = tf.reduce_sum(mu_loss + sigma_loss)
            self.update_model = optimizer.minimize(self.loss)

    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax

    def get_q_values(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.mu, feed_dict)
        return q_values

    def get_mu_sigma(self, sess, states):
        feed_dict = {self.input_states:states}
        mu, sigma = sess.run([self.mu, self.sigma], feed_dict)
        return mu, sigma

    def update(self, sess, states, actions, mu_targets, sigma_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.mu_targets:mu_targets,
                     self.sigma_targets:sigma_targets}
        sess.run(self.update_model, feed_dict)

        
        
class GaussianLogDeepQNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 activation_fn=tf.nn.relu,
                 scope="gaussian_log_boi", loss='wasserstein', reuse=False):

        ###################### Neural network architecture ######################

        input_shape = [None] + state_shape
        self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

        with tf.variable_scope(scope, reuse=reuse):
            out = conv_module(self.input_states, convs, activation_fn)
            out = layers.flatten(out)
            out = fc_module(out, fully_connected, activation_fn)
            out = fc_module(out, [num_actions * 2], None)
            self.mu, sigma = tf.split(out, num_or_size_splits=2, axis=1)
            self.sigma = sigma

            ######################### Optimization procedure ########################

            # one-hot encode actions to get q-values for state-action pairs
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            actions_onehot = tf.one_hot(self.input_actions, num_actions, dtype=tf.float32)
            mu_selected = tf.reduce_sum(tf.multiply(self.mu, actions_onehot), axis=1)
            sigma_selected = tf.reduce_sum(tf.multiply(self.sigma, actions_onehot), axis=1)

            # choose best actions (according to q-values)
            self.q_argmax = tf.argmax(self.mu, axis=1)

            # create loss function and update rule
            self.mu_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            

            self.sigma_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            

#             #loss_term1 = tf.log(tf.div(sigma_selected, (self.sigma_targets + 1e-6)))
#             loss_term1 = tf.subtract(tf.log(sigma_selected), tf.log(self.sigma_targets))
#             mu_diff = tf.square(tf.subtract(self.mu_targets, mu_selected))
#             nom = tf.square(self.sigma_targets) + mu_diff
#             denom = 2*tf.square(sigma_selected) + 1e-9
#             loss_term2 = tf.div(nom, denom)
            #self.loss = tf.reduce_sum(loss_term1 + loss_term2)
            
            if loss == 'wasserstein':
                mu_loss = tf.losses.mean_squared_error(self.mu_targets, mu_selected)
                sigma_loss = tf.losses.mean_squared_error(tf.exp(self.sigma_targets), tf.exp(sigma_selected))
                self.loss = mu_loss + sigma_loss
            
            if loss == 'kl':
                loss_term1 = tf.subtract(sigma_selected, self.sigma_targets)
                mu_diff = tf.square(tf.subtract(self.mu_targets, mu_selected))
                nom = tf.square(tf.exp(self.sigma_targets)) + mu_diff
                denom = 2*tf.square(tf.exp(sigma_selected)) + 1e-6
                loss_term2 = tf.div(nom, denom)  
                
                self.loss = tf.reduce_sum(loss_term1 + loss_term2)
                
            if loss == 'symmetric_kl':
                term1 = tf.square(tf.exp(sigma_selected)) + tf.square(tf.subtract(self.mu_targets, mu_selected))
                term1 = tf.div(term1, 2 * tf.square(tf.exp(self.sigma_targets)) + 1e-6)
                
                term2 = tf.square(tf.exp(self.sigma_targets)) + tf.square(tf.subtract(self.mu_targets, mu_selected))
                
                term2 = tf.div(term2, 2 * tf.square(tf.exp(sigma_selected)) + 1e-6)
                
                self.loss = tf.reduce_sum(term1 + term2)
                
                
                
            
            #self.loss = tf.reduce_sum(mu_loss + sigma_loss)
            self.update_model = optimizer.minimize(self.loss)

    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax

    def get_q_values(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.mu, feed_dict)
        return q_values

    def get_mu_sigma(self, sess, states):
        feed_dict = {self.input_states:states}
        mu, sigma = sess.run([self.mu, self.sigma], feed_dict)
        return mu, sigma

    def update(self, sess, states, actions, mu_targets, sigma_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.mu_targets:mu_targets,
                     self.sigma_targets:sigma_targets}
        sess.run(self.update_model, feed_dict)

class GMMDeepQNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 activation_fn=tf.nn.relu,
                 mixture_size=2,
                 metric="silly",
                 min_reward=None,
                 max_reward=None,
                 scope="gmm_boi", reuse=False):

        ###################### Neural network architecture ######################

        input_shape = [None] + state_shape
        self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

        if min_reward is None:
            self.min_reward = -1.0
        
        if max_reward is None:
            self.max_reward = np.prod(state_shape[:2]) - 3.0
        with tf.variable_scope(scope, reuse=reuse):
            out = conv_module(self.input_states, convs, activation_fn)
            out = layers.flatten(out)
            out = fc_module(out, fully_connected, activation_fn)
            out = fc_module(out, [num_actions * mixture_size * 3], None)
            out = tf.reshape(out, (-1, num_actions, mixture_size, 3))

            mu, sigma, pi = tf.unstack(out, axis=-1)
            self.mu = mu
            self.sigma = 10 * tf.sigmoid(sigma)
            self.pi = tf.nn.softmax(pi, axis=2)

            self.q_values = tf.reduce_sum(tf.multiply(self.pi, self.mu), axis=2)
            

            ######################### Optimization procedure ########################

            # one-hot encode actions to get q-values for state-action pairs
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            actions_onehot = tf.one_hot(self.input_actions, num_actions, dtype=tf.float32)
            actions_onehot_reshaped = tf.reshape(actions_onehot, [-1, num_actions, 1])

            mu_selected = tf.reduce_sum(tf.multiply(self.mu, actions_onehot_reshaped), axis=1)
            sigma_selected = tf.reduce_sum(tf.multiply(self.sigma, actions_onehot_reshaped), axis=1)
            pi_selected = tf.reduce_sum(tf.multiply(self.pi, actions_onehot_reshaped), axis=1)

            # choose best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            # create loss function and update rule
            self.mu_targets = tf.placeholder(dtype=tf.float32, shape=[None, mixture_size])
            self.sigma_targets = tf.placeholder(dtype=tf.float32, shape=[None, mixture_size])
            self.pi_targets = tf.placeholder(dtype=tf.float32, shape=[None, mixture_size])

            # def kl_gaussians(m1, s1, m2, s2):
            #     m1 = m1[:, :, None]
            #     m2 = m2[:, None, :]
            #     s1_2 = tf.square(s1)
            #     s2_2 = tf.square(s2)
            #     s1_2 = s1_2[:, :, None]
            #     s2_2 = s2_2[:, None, :]
            #     t1 = 0.5 * (tf.log(s2_2 / s1_2 + 1e-5))
            #     t2 = (s1_2 + (m2 - m1) ** 2) / (2 * s2_2)
            #     return t1 + t2 - 0.5
            #
            # def kl_mixtures(p1, m1, s1, p2, m2, s2):
            #     kl1 = kl_gaussians(m1, s1, m1, s1)
            #     kl2 = kl_gaussians(m1, s1, m2, s2)
            #     kl1 = tf.exp(-kl1)
            #     kl2 = tf.exp(-kl2)
            #
            #     t1 = tf.einsum('abc,ac->ab', kl1, p1)
            #     t2 = tf.einsum('abc,ac->ab', kl2, p2)
            #     return tf.reduce_mean(tf.multiply(p1, tf.log(t1 / t2 + 1e-5)))

            #self.loss = kl_mixtures(pi_selected, mu_selected, sigma_selected,
            #                        self.pi_targets, self.mu_targets, self.sigma_targets)
            
            
            def gauss_gramm(m1, s1, m2, s2):
                m1 = m1[:, :, None]
                s1 = s1[:, :, None]   
                m2 = m2[:, None, :]
                s2 = s2[:, None, :]  
                
                m = (m1 - m2)**2
    
                s = s1 ** 2 + s2 ** 2
    
                num = tf.exp(-m / (2 * s)) * np.sqrt(np.pi / 2)
    
                denom = np.pi * s1 * s2 * tf.sqrt(1.0 / s1 ** 2 + 1.0 / s2 ** 2)   
                return num / denom
        
        
            if metric == "silly":
                self.loss = tf.reduce_mean((mu_selected - self.mu_targets) ** 2) + tf.reduce_mean((sigma_selected - self.sigma_targets) ** 2) + tf.reduce_mean((pi_selected - self.pi_targets) ** 2)
                
            elif metric == "l2":
                
                gr1 = gauss_gramm(mu_selected, sigma_selected, mu_selected, sigma_selected)
                
                gr2 = gauss_gramm(mu_selected, sigma_selected, self.mu_targets, self.sigma_targets)
                
                gr3 = gauss_gramm(self.mu_targets, self.sigma_targets, self.mu_targets, self.sigma_targets)
                
                t1 = tf.einsum('nab,na,nb->n',gr1, pi_selected, pi_selected)
                
                t2 = tf.einsum('nab,na,nb->n',gr2, pi_selected, self.pi_targets)
                
                t3 = tf.einsum('nab,na,nb->n',gr3, self.pi_targets, self.pi_targets)
                
                self.loss = tf.reduce_mean(t1 - 2 * t2 + t3)
            
            elif metric == 'wasserstein':
                z = tf.linspace(self.min_reward, self.max_reward, 200)
                h = z[1] - z[0]
                
                def get_cdf(pi, mu, sigma, x):
                    tmp = pi[:, None, :] * 0.5 * (1 + tf.erf((x[None, :, None] - mu[:, None, :]) / (np.sqrt(2) * sigma[:, None, :])))
                    return tf.reduce_sum(tmp, axis=-1)
                
                cdf1 = get_cdf(pi_selected, mu_selected, sigma_selected, z)
                cdf2 = get_cdf(self.pi_targets, self.mu_targets, self.sigma_targets, z)
                self.loss = tf.reduce_sum((cdf1 - cdf2) ** 2) * h
                
            #self.loss = tf.reduce_sum(mu_loss + sigma_loss)
            self.update_model = optimizer.minimize(self.loss)

    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax

    def get_q_values(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values

    def get_pi_mu_sigma(self, sess, states):
        feed_dict = {self.input_states:states}
        pi, mu, sigma = sess.run([self.pi, self.mu, self.sigma], feed_dict)
        return pi, mu, sigma

    def update(self, sess, states, actions, pi_targets, mu_targets, sigma_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.pi_targets:pi_targets,
                     self.mu_targets:mu_targets,
                     self.sigma_targets:sigma_targets}

        # print ('pi', pi_targets)
        # print ('mu', mu_targets)
        # print ('sigma', sigma_targets)


        sess.run(self.update_model, feed_dict)



####################################################################################################
######################################## Experience Replay #########################################
####################################################################################################

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = namedtuple('Transition',
                                     ('s', 'a', 'r', 's_', 'end'))

    def push(self, *args):
        """ push single transition into buffer
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = [*args]
        self.position = (self.position + 1) % self.capacity

    def push_episode(self, episode_list):
        """ push whole episode into buffer
        """
        self.memory += episode_list
        gap = len(self.memory) - self.capacity
        if gap > 0:
            self.memory[:gap] = []


    def get_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = np.reshape(batch, [batch_size, 5])
        s = np.stack(batch[:,0])
        a = batch[:,1]
        r = batch[:,2]
        s_ = np.stack(batch[:,3])
        end = 1 - batch[:,4]
        return self.transition(s, a, r, s_, end)

    def __len__(self):
        return len(self.memory)
