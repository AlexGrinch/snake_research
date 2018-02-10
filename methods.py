import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import deque, namedtuple

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
                                               activation_fn=tf.nn.elu)
            out = layers.flatten(out)

            # fully connected part of the network
            with tf.variable_scope("fc"):
                for num_outputs in fully_connected:
                    out = layers.fully_connected(out,
                                                 num_outputs=num_outputs,
                                                 activation_fn=tf.nn.elu,
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