import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import t3f
from collections import deque, namedtuple
from tensorflow.contrib import rnn

####################################################################################################
########################################## Deep Q-Network ##########################################
####################################################################################################

class QNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 1],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 scope="q_network", reuse=False):

        """Class for neural network which estimates Q-function

        Parameters
        ----------
        num_actions: int
            number of actions the agent can take
        state_shape: list
            list of 3 parameters [frame_w, frame_h, num_frames]
            frame_w: frame width
            frame_h: frame height
            num_frames: number of successive frames considered as a state
        conv: list
            list of convolutional layers' parameters, each element
            has the form -- [num_outputs, kernel_size, stride]
        fully_connected: list
            list of fully connected layers' parameters, each element
            has the form -- num_outputs
        optimizer: tf.train optimizer
            optimization algorithm for stochastic gradient descend
        scope: str
            unique name of a specific network
        """

        xavier = layers.xavier_initializer()

        ###################### Neural network architecture ######################

        input_shape = [None] + state_shape
        self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

        with tf.variable_scope(scope, reuse=reuse):
            # convolutional part of the network
            out = self.input_states
            with tf.variable_scope("conv"):
                for num_outputs, kernel_size, stride in convs:
                    out = layers.convolution2d(out,
                                               num_outputs=num_outputs,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding='VALID',
                                               activation_fn=tf.nn.relu)
            out = layers.flatten(out)

            # fully connected part of the network
            with tf.variable_scope("fc"):
                for num_outputs in fully_connected:
                    out = layers.fully_connected(out,
                                                 num_outputs=num_outputs,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=xavier)
                    self.out = out

            # q-values estimation
            with tf.variable_scope("q_values"):
                q_weights = tf.Variable(xavier([fully_connected[-1], num_actions]))
                self.q_values = tf.matmul(out, q_weights)

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
        
    def get_features(self, sess, states):
        feed_dict = {self.input_states:states}
        features = sess.run(self.out, feed_dict)
        return features

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

class DuelQNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 1],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 scope="q_network", reuse=False):

        """Class for neural network which estimates Q-function

        Parameters
        ----------
        num_actions: int
            number of actions the agent can take
        state_shape: list
            list of 3 parameters [frame_w, frame_h, num_frames]
            frame_w: frame width
            frame_h: frame height
            num_frames: number of successive frames considered as a state
        conv: list
            list of convolutional layers' parameters, each element
            has the form -- [num_outputs, kernel_size, stride]
        fully_connected: list
            list of fully connected layers' parameters, each element
            has the form -- num_outputs
        optimizer: tf.train optimizer
            optimization algorithm for stochastic gradient descend
        scope: str
            unique name of a specific network
        """

        xavier = layers.xavier_initializer()

        ###################### Neural network architecture ######################

        input_shape = [None] + state_shape
        self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

        with tf.variable_scope(scope, reuse=reuse):
            # convolutional part of the network
            out = self.input_states
            with tf.variable_scope("conv"):
                for num_outputs, kernel_size, stride in convs:
                    out = layers.convolution2d(out,
                                               num_outputs=num_outputs,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding='VALID',
                                               activation_fn=tf.nn.relu)
            out = layers.flatten(out)
            adv, val = tf.split(out, num_or_size_splits=2, axis=1)

            # advantage function estimation
            with tf.variable_scope("advantage"):
                for num_outputs in fully_connected:
                    adv = layers.fully_connected(adv,
                                                 num_outputs=num_outputs,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=xavier)
                adv_weights = tf.Variable(xavier([fully_connected[-1], num_actions]))
                self.a_values = tf.matmul(adv, adv_weights)
                
            with tf.variable_scope("value"):
                for num_outputs in fully_connected:
                    val = layers.fully_connected(val,
                                                 num_outputs=num_outputs,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=xavier)   
                val_weights = tf.Variable(xavier([fully_connected[-1], 1]))
                self.v_values = tf.matmul(val, val_weights)

            # q-values estimation
            with tf.variable_scope("q_values"):
                avg_a_values = tf.reduce_mean(self.a_values, axis=1, keepdims=True)
                shifted_a_values = tf.subtract(self.a_values, avg_a_values)
                self.q_values = self.v_values + shifted_a_values

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
##################################### Distributional Q-Network #####################################
####################################################################################################

class DistQNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 1],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128], num_atoms=21, v=(-10, 10),
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 scope="distributional_q_network", reuse=False):

        """Class for neural network which estimates Q-function distribution

        Parameters
        ----------
        num_actions: int
            number of actions the agent can take
        state_shape: list
            list of 3 parameters [frame_w, frame_h, num_frames]
            frame_w: frame width
            frame_h: frame height
            num_frames: number of successive frames considered as a state
        conv: list
            list of convolutional layers' parameters, each element
            has the form -- [num_outputs, kernel_size, stride]
        fully_connected: list
            list of fully connected layers' parameters, each element
            has the form -- num_outputs
        num_atoms: int
            number of atoms in distribution support
        v: tuple
            tuple of 2 parameters (v_min, v_max)
            v_min: minimum q-function value
            v_max: maximum q-function value
        optimizer: tf.train optimizer
            optimization algorithm for stochastic gradient descend
        scope: str
            unique name of a specific network
        """

        xavier = layers.xavier_initializer()

        ###################### Neural network architecture ######################

        input_shape = [None] + state_shape
        self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

        with tf.variable_scope(scope, reuse=reuse):
            # convolutional part of the network
            out = self.input_states
            with tf.variable_scope("conv"):
                for num_outputs, kernel_size, stride in convs:
                    out = layers.convolution2d(out,
                                               num_outputs=num_outputs,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding='VALID',
                                               activation_fn=tf.nn.relu)
            out = layers.flatten(out)

            # fully connected part of the network
            with tf.variable_scope("fc"):
                for num_outputs in fully_connected:
                    out = layers.fully_connected(out,
                                                 num_outputs=num_outputs,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=xavier)

            # distribution parameters
            self.num_atoms = num_atoms
            self.v_min, self.v_max = v
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

            # distributional head
            with tf.variable_scope("probs"):
                action_probs = []
                for a in range(num_actions):
                    action_prob = layers.fully_connected(out,
                                                         num_outputs=self.num_atoms,
                                                         activation_fn=tf.nn.softmax,
                                                         weights_initializer=xavier)
                    action_probs.append(action_prob)
                self.probs = tf.stack(action_probs, axis=1)

            # q-values estimation
            with tf.variable_scope("q_values"):
                q_values = []
                for a in range(num_actions):
                    q_value = tf.reduce_sum(self.z * action_probs[a], axis=1)
                    q_value = tf.reshape(q_value, [-1, 1])
                    q_values.append(q_value)
                self.q_values = tf.concat(q_values, axis=1)

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
######################################## TensorTrain Q-Table #######################################
####################################################################################################

class QQTTTable:
    
    def __init__(self, num_actions, num_colors=2, state_shape=[8, 8, 3],
                 tt_rank=24, optimizer=tf.train.AdamOptimizer(2.5e-4), 
                 dtype=tf.float32, scope="qqtt_network", reuse=False):
        
        input_shape = np.prod(state_shape) * [num_colors,] + [num_actions,]
        
        with tf.variable_scope(scope, reuse=reuse):
            
            # random initialization of Q-tensor
            q0init = t3f.random_tensor(shape=input_shape, tt_rank=tt_rank, stddev=1e-3)
            q0init = t3f.cast(q0init, dtype=dtype)
            q0 = t3f.get_variable('Q', initializer=q0init)
        
            self.input_states = tf.placeholder(dtype=tf.int32, shape=[None]+state_shape)
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            self.input_targets = tf.placeholder(dtype=dtype, shape=[None])

            reshaped_s = tf.reshape(self.input_states, (-1, np.prod(state_shape)))
            reshaped_a = tf.reshape(self.input_actions, (-1, 1))
            input_s_and_a = tf.concat([reshaped_s, reshaped_a], axis=1) 
            self.q_selected = t3f.gather_nd(q0, input_s_and_a, dtype=dtype)

            reshaped_s_ = tf.reshape(self.input_states, [-1]+state_shape)
            
            # some shitty code
            s_a_idx = tf.concat(num_actions * [reshaped_s], axis=0) 
            actions_range = tf.range(start=0, limit=num_actions)
            a_idx = self.tf_repeat(actions_range, tf.shape(self.input_states)[0:1])
            s_a_idx = tf.concat([s_a_idx, a_idx], axis=1)
            vals = t3f.gather_nd(q0, s_a_idx, dtype=dtype)
            self.q_values = tf.transpose(tf.reshape(vals, shape=(num_actions, -1)))
            # shitty code ends here
            
            self.q_argmax = tf.argmax(self.q_values, axis=1)
            self.q_max = tf.reduce_max(self.q_values, axis=1)
            
            self.loss = tf.losses.huber_loss(self.q_selected, self.input_targets)
            self.update_model = optimizer.minimize(self.loss)
        
    def update(self, sess, states, actions, targets):
        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.input_targets:targets}
        sess.run(self.update_model, feed_dict)
        
    def get_q_action_values(self, sess, states, actions):
        feed_dict = {self.input_states:states,
                     self.input_actions:actions}
        return sess.run(self.q_selected, feed_dict=feed_dict)
        
    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        return sess.run(self.q_argmax, feed_dict=feed_dict)
    
    def get_q_max(self, sess, states):
        feed_dict = {self.input_states:states}
        return sess.run(self.q_max, feed_dict=feed_dict)
    
    def get_q_values(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values
    
    def tf_repeat(self, x, num):
        u = tf.reshape(x, (-1, 1))
        ones = tf.ones(1, dtype=tf.int32)
        u = tf.tile(u, tf.concat([ones, num], axis=0))
        u = tf.reshape(u, (-1, 1))
        return u

####################################################################################################
##################################### Policy Gradient Network ######################################
####################################################################################################

class PGNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 1],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 scope="q_network", reuse=False):

        """Class for neural network which estimates policy

        Parameters
        ----------
        num_actions: int
            number of actions the agent can take
        state_shape: list
            list of 3 parameters [frame_w, frame_h, num_frames]
            frame_w: frame width
            frame_h: frame height
            num_frames: number of successive frames considered as a state
        conv: list
            list of convolutional layers' parameters, each element
            has the form -- [num_outputs, kernel_size, stride]
        fully_connected: list
            list of fully connected layers' parameters, each element
            has the form -- num_outputs
        optimizer: tf.train optimizer
            optimization algorithm for stochastic gradient descend
        scope: str
            unique name of a specific network
        """

        xavier = layers.xavier_initializer()

        ###################### Neural network architecture ######################

        input_shape = [None] + state_shape
        self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

        with tf.variable_scope(scope, reuse=reuse):
            # convolutional part of the network
            out = self.input_states
            with tf.variable_scope("conv"):
                for num_outputs, kernel_size, stride in convs:
                    out = layers.convolution2d(out,
                                               num_outputs=num_outputs,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding='VALID',
                                               activation_fn=tf.nn.relu)
            out = layers.flatten(out)

            # fully connected part of the network
            with tf.variable_scope("fc"):
                for num_outputs in fully_connected:
                    out = layers.fully_connected(out,
                                                 num_outputs=num_outputs,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=xavier)

            # q-values estimation
            with tf.variable_scope("policy"):
                out = layers.fully_connected(out,
                                             num_outputs=num_actions,
                                             activation_fn=tf.nn.softmax,
                                             weights_initializer=xavier)
                self.probs = out

        ######################### Optimization procedure ########################

        # one-hot encode actions to get q-values for state-action pairs
        self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
        actions_onehot = tf.one_hot(self.input_actions, num_actions, dtype=tf.float32)
        probs_selected = tf.reduce_sum(tf.multiply(self.probs, actions_onehot), axis=1)

        # choose best actions (according to q-values)
        self.p_argmax = tf.argmax(self.probs, axis=1)

        # create loss function and update rule
        self.targets = tf.placeholder(dtype=tf.float32, shape=[None])
        self.loss = -tf.reduce_sum(self.targets * tf.log(probs_selected + 1e-6))
        self.update_model = optimizer.minimize(self.loss)

    def get_p_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        p_argmax = sess.run(self.p_argmax, feed_dict)
        return p_argmax

    def get_probs(self, sess, states):
        feed_dict = {self.input_states:states}
        probs = sess.run(self.probs, feed_dict)
        return probs

    def update(self, sess, states, actions, targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.targets:targets}
        sess.run(self.update_model, feed_dict)
        return sess.run(self.loss, feed_dict)

####################################################################################################
##################################### Policy Gradient Network ######################################
####################################################################################################

class ActorCriticNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 1],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 scope="q_network", reuse=False):

        """Class for neural network which estimates policy and value function

        Parameters
        ----------
        num_actions: int
            number of actions the agent can take
        state_shape: list
            list of 3 parameters [frame_w, frame_h, num_frames]
            frame_w: frame width
            frame_h: frame height
            num_frames: number of successive frames considered as a state
        conv: list
            list of convolutional layers' parameters, each element
            has the form -- [num_outputs, kernel_size, stride]
        fully_connected: list
            list of fully connected layers' parameters, each element
            has the form -- num_outputs
        optimizer: tf.train optimizer
            optimization algorithm for stochastic gradient descend
        scope: str
            unique name of a specific network
        """

        xavier = layers.xavier_initializer()

        ###################### Neural network architecture ######################

        input_shape = [None] + state_shape
        self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

        with tf.variable_scope(scope, reuse=reuse):
            # convolutional part of the network
            out = self.input_states
            with tf.variable_scope("conv"):
                for num_outputs, kernel_size, stride in convs:
                    out = layers.convolution2d(out,
                                               num_outputs=num_outputs,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding='VALID',
                                               activation_fn=tf.nn.relu)
            out = layers.flatten(out)

            # fully connected part of the network
            with tf.variable_scope("fc"):
                for num_outputs in fully_connected:
                    out = layers.fully_connected(out,
                                                 num_outputs=num_outputs,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=xavier)

            # q-values estimation
            with tf.variable_scope("policy"):
                self.probs = layers.fully_connected(out,
                                                    num_outputs=num_actions,
                                                    activation_fn=tf.nn.softmax,
                                                    weights_initializer=xavier)
                self.values = layers.fully_connected(out,
                                                     num_outputs=1,
                                                     activation_fn=None,
                                                     weights_initializer=xavier)

        ######################### Optimization procedure ########################

        # one-hot encode actions to get q-values for state-action pairs
        self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
        actions_onehot = tf.one_hot(self.input_actions, num_actions, dtype=tf.float32)
        probs_selected = tf.reduce_sum(tf.multiply(self.probs, actions_onehot), axis=1)

        # choose best actions (according to q-values)
        self.p_argmax = tf.argmax(self.probs, axis=1)

        self.pg_targets = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.pg_loss = -tf.reduce_sum(self.pg_targets * tf.log(probs_selected))

        self.value_targets = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.value_loss = tf.losses.huber_loss(self.value_targets, self.values)

        # create loss function and update rule
        self.loss = self.pg_loss + self.value_loss
        self.update_model = optimizer.minimize(self.loss)

    def get_p_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        p_argmax = sess.run(self.p_argmax, feed_dict)
        return p_argmax

    def get_probs(self, sess, states):
        feed_dict = {self.input_states:states}
        probs = sess.run(self.probs, feed_dict)
        return probs

    def get_values(self, sess, states):
        feed_dict = {self.input_states:states}
        values = sess.run(self.values, feed_dict)
        return values

    def update(self, sess, states, actions, pg_targets, value_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.pg_targets:pg_targets,
                     self.value_targets:value_targets}
        sess.run(self.update_model, feed_dict)
        return sess.run([self.pg_loss, self.value_loss], feed_dict)


####################################################################################################
######################################## ReflexNetwork #############################################
####################################################################################################


class ReflexDistQNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 1],
                 convs=[[32, 2, 2], [64, 2, 1]],
                 fully_connected=[64], num_atoms=21, v=(-10, 10),
                 reflex=4, lstm_units=64, optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 scope="reflex_distributional_q_network", reuse=False):

        """Class for neural network which estimates Q-function distribution

        Parameters
        ----------
        num_actions: int
            number of actions the agent can take
        state_shape: list
            list of 3 parameters [frame_w, frame_h, num_frames]
            frame_w: frame width
            frame_h: frame height
            num_frames: number of successive frames considered as a state
        conv: list
            list of convolutional layers' parameters, each element
            has the form -- [num_outputs, kernel_size, stride]
        fully_connected: list
            list of fully connected layers' parameters, each element
            has the form -- num_outputs
        num_atoms: int
            number of atoms in distribution support
        v: tuple
            tuple of 2 parameters (v_min, v_max)
            v_min: minimum q-function value
            v_max: maximum q-function value
        optimizer: tf.train optimizer
            optimization algorithm for stochastic gradient descend
        scope: str
            unique name of a specific network
        """

        xavier = layers.xavier_initializer()

        ###################### Neural network architecture ######################

        input_shape = [None] + state_shape
        self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

        with tf.variable_scope(scope, reuse=reuse):
            # convolutional part of the network
            out = self.input_states
            with tf.variable_scope("conv"):
                for num_outputs, kernel_size, stride in convs:
                    out = layers.convolution2d(out,
                                               num_outputs=num_outputs,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding='VALID',
                                               activation_fn=tf.nn.relu)
            out = layers.flatten(out)

            # fully connected part of the network
            with tf.variable_scope("fc"):
                for num_outputs in fully_connected:
                    out = layers.fully_connected(out,
                                                 num_outputs=num_outputs,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=xavier)

            # reflex head
            self.lstm_units = lstm_units
            self.reflex = reflex
            with tf.variable_scope("reflex"):
                frames = self.reflex * [out]
                lstm_layer = rnn.BasicLSTMCell(self.lstm_units, forget_bias=1)
                outputs, _ = rnn.static_rnn(lstm_layer, frames, dtype="float32")
                num_outputs = fully_connected[-1]
                out_final = layers.fully_connected(outputs[-1],
                                                   num_outputs=num_outputs,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=xavier)

            out_with_advice = tf.concat([out, out_final], axis=1)

            # distribution parameters
            self.num_atoms = num_atoms
            self.v_min, self.v_max = v
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

            # distributional head
            with tf.variable_scope("probs"):
                action_probs = []
                for a in range(num_actions):
                    action_prob = layers.fully_connected(out_with_advice,
                                                         num_outputs=self.num_atoms,
                                                         activation_fn=tf.nn.softmax,
                                                         weights_initializer=xavier)
                    action_probs.append(action_prob)
                self.probs = tf.stack(action_probs, axis=1)


            # q-values estimation
            with tf.variable_scope("q_values"):
                q_values = []
                for a in range(num_actions):
                    q_value = tf.reduce_sum(self.z * action_probs[a], axis=1)
                    q_value = tf.reshape(q_value, [-1, 1])
                    q_values.append(q_value)
                self.q_values = tf.concat(q_values, axis=1)

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

    def get_action_seq(self, sess, states):
        feed_dict = {self.input_states:states}
        a_seq = sess.run(self.action_seq, feed_dict)
        return a_seq

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
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = [*args]
        self.position = (self.position + 1) % self.capacity
        
    def push_episode(self, episode_list):
        
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
