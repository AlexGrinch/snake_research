import os
import time
import random
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

import gym
from gym import spaces
from PIL import Image
from collections import deque, namedtuple

from IPython import display
import matplotlib.pyplot as plt

from environments import Snake
from methods import QNetwork, ReplayMemory

class SnakeDQNAgent:
    
    def __init__(self, state_shape=[4, 4, 1], convs=[[16, 2, 1], [32, 1, 1]], 
                 fully_connected=[128], model_name="baseline_agent"):
        
        """Class for training and evaluating DQN agent on Atari games
        
        Parameters
        ----------
        game_id: str
            game identifier in gym environment, e.g. "Pong"
        num_actions: int
            number of actions the agent can take
        model_name: str
            name of the model
        """
        
        ############################ Game environment ############################
        
        self.train_env = Snake(grid_size=state_shape[:-1])
        self.num_actions = 3
            
        self.path = "snake_models" + "/" + model_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        ############################# Agent & Target #############################
        
        tf.reset_default_graph()
        self.agent_net = QNetwork(self.num_actions, state_shape=state_shape, 
                                  convs=convs, fully_connected=fully_connected, 
                                  scope="agent")
        self.target_net = QNetwork(self.num_actions, state_shape=state_shape,
                                   convs=convs, fully_connected=fully_connected,
                                   scope="target")
        
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        
        all_vars = tf.trainable_variables()
        num_vars = len(all_vars) // 2
        self.agent_vars = all_vars[:num_vars]
        self.target_vars = all_vars[num_vars:]
        
    def set_parameters(self, 
                       replay_memory_size=50000,
                       replay_start_size=10000,
                       init_eps=1,
                       final_eps=0.02,
                       annealing_steps=100000,
                       discount_factor=0.99,
                       max_episode_length=2000):
        
        # create experience replay and fill it with random policy samples
        self.rep_buffer = ReplayMemory(replay_memory_size)
        frame_count = 0
        while (frame_count < replay_start_size):
            s = self.train_env.reset()
            for time_step in range(max_episode_length):
                a = np.random.randint(self.num_actions)
                s_, r, end = self.train_env.step(a)
                self.rep_buffer.push(s, a, r, s_, end)
                s = s_
                frame_count += 1
                if end: break
        
        # define epsilon decrement schedule for exploration
        self.eps = init_eps
        self.final_eps = final_eps
        self.eps_drop = (init_eps - final_eps) / annealing_steps
        
        self.gamma = discount_factor
        self.max_ep_length = max_episode_length
        
    def train(self,
              gpu_id=0,
              batch_size=32,
              agent_update_freq=4,
              target_update_freq=5000,
              tau=1,
              max_num_epochs=50000,
              performance_print_freq=500,
              save_freq=10000, 
              from_epoch=0):
        
        if (gpu_id == -1):
            config = tf.ConfigProto()
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            
        target_ops = self.update_target_graph(tau)
        
        with tf.Session(config=config) as sess:
            
            if from_epoch == 0:
                sess.run(self.init)
                train_rewards = []
                frame_counts = []
                frame_count = 0
                num_epochs = 0
                
            else:
                self.saver.restore(sess, self.path+"/model-"+str(from_epoch))
                train_rewards = list(np.load(self.path+"/learning_curve.npz")["r"])
                frame_counts = list(np.load(self.path+"/learning_curve.npz")["f"])
                frame_count = frame_counts[-1]
                num_epochs = from_epoch
    
            episode_count = 0
            ep_lifetimes = []
        
            
            while num_epochs < max_num_epochs:
                
                train_ep_reward = 0
                
                # reset the environment / start new game
                s = self.train_env.reset()
                for time_step in range(self.max_ep_length):
                    
                    # choose action e-greedily
                    if np.random.rand(1) < self.eps:
                        a = np.random.randint(self.num_actions)
                    else:
                        a = self.agent_net.get_q_argmax(sess, [s])
                        
                    # make step in the environment    
                    s_, r, end = self.train_env.step(a)
                    
                    # save transition into experience replay
                    self.rep_buffer.push(s, a, np.sign(r), s_, end)
                    
                    # update current state and statistics
                    s = s_
                    frame_count += 1
                    train_ep_reward += r
                    
                    # reduce epsilon according to schedule
                    if self.eps > self.final_eps:
                        self.eps -= self.eps_drop
                    
                    # update network weights
                    if frame_count % agent_update_freq == 0:
                        
                        batch = self.rep_buffer.get_batch(batch_size)
                        
                        # estimate the right hand side of Bellman equation
                        max_actions = self.agent_net.get_q_argmax(sess, batch.s_)
                        q_values = self.target_net.get_q_values(sess, batch.s_)
                        double_q = q_values[np.arange(batch_size), max_actions]
                        targets = batch.r + (self.gamma * double_q * batch.end)
                        
                        # update agent network
                        self.agent_net.update(sess, batch.s, batch.a, targets)
                        
                        # update target network
                        if tau == 1:
                            if frame_count % target_update_freq == 0:
                                self.update_target_weights(sess, target_ops)
                        else: self.update_target_weights(sess, target_ops)
                    
                    # make checkpoints of network weights and save learning curve
                    if frame_count % save_freq == 1:
                        num_epochs += 1
                        try:
                            self.saver.save(sess, self.path+"/model", global_step=num_epochs)
                            np.savez(self.path+"/learning_curve.npz", r=train_rewards, 
                                     f=frame_counts, l=ep_lifetimes)
                        except: pass
                    
                    # if game is over, reset the environment
                    if end: break
                         
                episode_count += 1
                train_rewards.append(train_ep_reward)
                frame_counts.append(frame_count)
                ep_lifetimes.append(time_step+1)
                
                # print performance once in a while
                if episode_count % performance_print_freq == 0:
                    avg_reward = np.mean(train_rewards[-performance_print_freq:])
                    avg_lifetime = np.mean(ep_lifetimes[-performance_print_freq:])
                    print("frame count:", frame_count)
                    print("average reward:", avg_reward)
                    print("epsilon:", round(self.eps, 3))
                    print("average lifetime:", avg_lifetime) 
                    print("-------------------------------")

    def update_target_graph(self, tau):
        op_holder = []
        for agnt, trgt in zip(self.agent_vars, self.target_vars):
            op = trgt.assign(agnt.value()*tau + (1 - tau)*trgt.value())
            op_holder.append(op)
        return op_holder

    def update_target_weights(self, sess, op_holder):
        for op in op_holder:
            sess.run(op)
            
    def play(self,
             gpu_id=0,
             max_episode_length=2000,
             from_epoch=0):
        
        if (gpu_id == -1):
            config = tf.ConfigProto()
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True        
            
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, self.path+"/model-"+str(from_epoch))
            s = self.train_env.reset()
            R = 0
            for time_step in range(max_episode_length):
                a = self.agent_net.get_q_argmax(sess, [s])[0]
                s, r, done = self.train_env.step(a)
                R += r
                
                self.train_env.plot_state()
                display.clear_output(wait=True)
                display.display(plt.gcf())
                
                if done: break
        return R
    
    def test_agent(self,
                   gpu_id=0,
                   num_episodes=10,
                   max_episode_length=2000,
                   from_epoch=0):
        
        if (gpu_id == -1):
            config = tf.ConfigProto()
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True        
            
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, self.path+"/model-"+str(from_epoch))
            rewards = []
            bellman_errors = []
            for episode_count in range(num_episodes):
                s = self.train_env.reset()
                R = 0
                BE = 0
                for time_step in range(max_episode_length):
                    a = self.agent_net.get_q_argmax(sess, [s])[0]
                    q = self.agent_net.get_q_values(sess, [s])[0][a]
                    
                    s, r, done = self.train_env.step(a)
                    a_ = self.agent_net.get_q_argmax(sess, [s])[0]
                    q_ = self.agent_net.get_q_values(sess, [s])[0][a_]
                    
                    BE += np.abs(q - (r + q_))
                    R += r
                    if done: break
                rewards.append(R)
                bellman_errors.append(BE/(time_step+1))
                
        return rewards, bellman_errors
        