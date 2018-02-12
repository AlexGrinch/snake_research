import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import deque, namedtuple

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
        self.loss = -tf.reduce_sum(self.targets * tf.log(self.probs_selected))
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
##################################### Policy Gradient Network ######################################
####################################################################################################

class PGNetwork:
    
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
        self.loss = -tf.reduce_sum(self.targets * tf.log(probs_selected))
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