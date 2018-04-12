import os
import time
import random
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers
import t3f

import gym
from gym import spaces
from PIL import Image
from collections import deque, namedtuple

from IPython import display
import matplotlib.pyplot as plt

from environments import Snake
from methods import *


from queue import Queue

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

############################## Snake agent template ##############################

class SnakeAgent:
    
    def __init__(self, state_shape=[4, 4, 3], model_name="baseline_agent"):
        
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
            
        self.path = "snake_models/" + model_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            
    def init_weights(self):
        
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
              exploration="e-greedy",
              agent_update_freq=4,
              target_update_freq=5000,
              tau=1,
              max_num_epochs=50000,
              performance_print_freq=500,
              save_freq=10000, 
              from_epoch=0):
        
        config = self.gpu_config(gpu_id)
        target_ops = self.update_target_graph(tau)
        self.batch_size = batch_size
        
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
                    a = self.choose_action(sess, s, exploration)
                        
                    # make step in the environment    
                    s_, r, end = self.train_env.step(a)
                    
                    # save transition into experience replay
                    self.rep_buffer.push(s, a, r, s_, end)
                    
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
                        self.update_agent_weights(sess, batch)
                        
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
                    
    def choose_action(self, sess, s, exploration="e-greedy"):
        
        if (exploration == "greedy"):
            a = self.agent_net.get_q_argmax(sess, [s])
        elif (exploration == "e-greedy"):
            if np.random.rand(1) < self.eps:
                a = np.random.randint(self.num_actions)
            else:
                a = self.agent_net.get_q_argmax(sess, [s])     
        elif (exploration == "boltzmann"):
            q_values = self.agent_net.get_q_values(sess, [s])
            logits = q_values / self.eps
            probs = softmax(logits).ravel()
            a = np.random.choice(self.num_actions, p=probs)
        else:
            return 0
        return a
            
                    
    def update_agent_weights(self, sess, batch):
        pass

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
        
        config = self.gpu_config(gpu_id)       
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
    
    def play_n_episodes(self,
                        num_episodes=1,
                        gpu_id=0,
                        max_episode_length=2000,
                        exploration='greedy',
                        from_epoch=0):
        
        config = self.gpu_config(gpu_id)
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, self.path+"/model-"+str(from_epoch))
            states_history = []
            reward_history = []
            for episode_count in range(num_episodes):
                s = self.train_env.reset()
                ep_reward = 0
                ep_states = [s]
                for time_step in range(max_episode_length):
                    a = self.choose_action(sess, s, exploration)
                    s, r, done = self.train_env.step(a)
                    ep_reward += r
                    ep_states.append(s)
                    if done: break
                states_history.append(ep_states)
                reward_history.append(ep_reward)
        return states_history, reward_history  
    
    def play_n_episodes_with_gradients(self,
                        num_episodes=1,
                        gpu_id=0,
                        max_episode_length=2000,
                        exploration='greedy',
                        from_epoch=0):
        
        config = self.gpu_config(gpu_id)
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, self.path+"/model-"+str(from_epoch))
            states_history = []
            reward_history = []
            gradients_history = []
            bellmans_history = []
            for episode_count in range(num_episodes):
                s = self.train_env.reset()
                ep_reward = 0
                ep_states = [s]
                ep_grads = []
             
                q_values = []
                rewards = []
                
                for time_step in range(max_episode_length):
                    a = self.choose_action(sess, s, exploration)
                    q = self.agent_net.get_q_values(sess, [s]).ravel()[a]
                    q_values.append(q)
                    
                    grad = self.agent_net.get_gradients(sess, [s], [a])
                    
                    ep_grads.append(grad)
                    
                    s, r, done = self.train_env.step(a)
                    rewards.append(r)
                    ep_reward += r
                    ep_states.append(s)
                    if done: break
                
                qs = np.array(q_values)
                next_qs = np.concatenate((qs[1:], np.zeros(1)))
                rs = np.array(rewards)
                bellman_errors = np.abs(qs - (rs + self.gamma * next_qs))
                
                bellmans_history.append(bellman_errors)    
                states_history.append(ep_states)
                reward_history.append(ep_reward)
                gradients_history.append(ep_grads)
        return states_history, reward_history, gradients_history, bellmans_history
    
    def play_n_episodes_with_actions_and_rewards(self,
                        num_episodes=1,
                        gpu_id=0,
                        max_episode_length=2000,
                        exploration='greedy',
                        from_epoch=0):
        
        config = self.gpu_config(gpu_id)
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, self.path+"/model-"+str(from_epoch))
            states_history = []
            reward_history = []
            action_history = []
            for episode_count in range(num_episodes):
                s = self.train_env.reset()
                #ep_reward = 0
                ep_states = [s]               
                ep_actions = []
                ep_rewards = []
                
                for time_step in range(max_episode_length):
                    a = self.choose_action(sess, s, exploration)
                    s, r, done = self.train_env.step(a)
                    #ep_reward += r
                    ep_states.append(s)
                    ep_actions.append(a)
                    ep_rewards.append(r)
                    if done: break
                states_history.append(ep_states)
                reward_history.append(ep_rewards)
                action_history.append(ep_actions)
        return states_history, reward_history, action_history
    
    
    
    
    
    def get_q_values(self, states, gpu_id=0, from_epoch=0):
        
        config = self.gpu_config(gpu_id)
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, self.path+"/model-"+str(from_epoch))
            q_values = self.agent_net.get_q_values(sess, states)
        return q_values
    
    def gpu_config(self, gpu_id):
        if (gpu_id == -1):
            config = tf.ConfigProto()
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        return config
    
############################## Deep Q-Network agent ##############################

class SnakeDQNAgent(SnakeAgent):
    
    def __init__(self, state_shape=[4, 4, 3], 
                 convs=[[16, 2, 1], [32, 1, 1]], 
                 fully_connected=[128],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 model_name="DQN"):
        
        super(SnakeDQNAgent, self).__init__(state_shape=state_shape,
                                            model_name=model_name)
        
        tf.reset_default_graph()
        self.agent_net = QNetwork(self.num_actions, state_shape=state_shape,
                                  convs=convs, fully_connected=fully_connected, 
                                  optimizer=optimizer, scope="agent")
        self.target_net = QNetwork(self.num_actions, state_shape=state_shape,
                                   convs=convs, fully_connected=fully_connected, 
                                   optimizer=optimizer, scope="target")
        self.init_weights()
        
        
    def update_agent_weights(self, sess, batch):
        
        # estimate the right hand side of Bellman equation
        max_actions = self.agent_net.get_q_argmax(sess, batch.s_)
        q_values = self.target_net.get_q_values(sess, batch.s_)
        double_q = q_values[np.arange(self.batch_size), max_actions]
        targets = batch.r + (self.gamma * double_q * batch.end)

        # update agent network
        self.agent_net.update(sess, batch.s, batch.a, targets)
        
############################## Deep Q-Network agent ##############################

class SnakeDuelDQNAgent(SnakeAgent):
    
    def __init__(self, state_shape=[4, 4, 3], 
                 convs=[[16, 2, 1], [32, 1, 1]], 
                 fully_connected=[128],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 model_name="DQN"):
        
        super(SnakeDuelDQNAgent, self).__init__(state_shape=state_shape,
                                                model_name=model_name)
        
        tf.reset_default_graph()
        self.agent_net = DuelQNetwork(self.num_actions, state_shape=state_shape,
                                      convs=convs, fully_connected=fully_connected,
                                      optimizer=optimizer, scope="agent")
        self.target_net = DuelQNetwork(self.num_actions, state_shape=state_shape,
                                       convs=convs, fully_connected=fully_connected,
                                       optimizer=optimizer, scope="target")
        self.init_weights()
        
        
    def update_agent_weights(self, sess, batch):
        
        # estimate the right hand side of Bellman equation
        max_actions = self.agent_net.get_q_argmax(sess, batch.s_)
        q_values = self.target_net.get_q_values(sess, batch.s_)
        double_q = q_values[np.arange(self.batch_size), max_actions]
        targets = batch.r + (self.gamma * double_q * batch.end)

        # update agent network
        self.agent_net.update(sess, batch.s, batch.a, targets)

############################## Distributional agent ##############################

class SnakeDistDQNAgent(SnakeAgent):
    
    def __init__(self, state_shape=[4, 4, 3],
                 convs=[[16, 2, 1], [32, 1, 1]], 
                 fully_connected=[128],
                 num_atoms=21,
                 v=(-10, 10),
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 model_name="DistDQN"):

        super(SnakeDistDQNAgent, self).__init__(state_shape=state_shape,
                                                model_name=model_name)

        tf.reset_default_graph()
        self.agent_net = DistQNetwork(self.num_actions, state_shape=state_shape,
                                      convs=convs, fully_connected=fully_connected,
                                      num_atoms=num_atoms, v=v,
                                      optimizer=optimizer, scope="agent")
        self.target_net = DistQNetwork(self.num_actions, state_shape=state_shape,
                                       convs=convs, fully_connected=fully_connected,
                                       num_atoms=num_atoms, v=v,
                                       optimizer=optimizer, scope="target")
        self.init_weights()

    def update_agent_weights(self, sess, batch):

        # estimate the projection of the right hand side of Bellman equation
        max_actions = self.agent_net.get_q_argmax(sess, batch.s_)
        target_m = self.target_net.cat_proj(sess, batch.r, batch.s_,
                                            max_actions, batch.end,
                                            gamma=self.gamma)

        # update agent network
        self.agent_net.update(sess, batch.s, batch.a, target_m)
        
#################################### QTT agent ###################################

class SnakeQQTTAgent(SnakeAgent):
    
    def __init__(self, state_shape=[4, 4, 3],
                 num_colors=2, 
                 tt_rank=8,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 model_name="DistDQN"):

        super(SnakeQQTTAgent, self).__init__(state_shape=state_shape,
                                             model_name=model_name)

        tf.reset_default_graph()
        self.agent_net = QQTTTable(self.num_actions, state_shape=state_shape,
                                   num_colors=num_colors,tt_rank=tt_rank, 
                                   optimizer=optimizer, dtype=tf.float32, scope="agent")
        self.target_net = QQTTTable(self.num_actions, state_shape=state_shape,
                                    num_colors=num_colors,tt_rank=tt_rank,
                                    optimizer=optimizer, dtype=tf.float32, scope="target")
        self.init_weights()

    def update_agent_weights(self, sess, batch):

        # estimate the right hand side of Bellman equation
        max_actions = self.agent_net.get_q_argmax(sess, batch.s_)
        q_values = self.target_net.get_q_values(sess, batch.s_)
        double_q = q_values[np.arange(self.batch_size), max_actions]
        targets = batch.r + (self.gamma * double_q * batch.end)

        # update agent network
        self.agent_net.update(sess, batch.s, batch.a, targets)
        
        
        
#################################### SnakeDQNAgentFisherI(SnakeDQNAgent) ###################################   

class SnakeDQNAgentFisherI(SnakeDQNAgent):
    
    def __init__(self, state_shape=[4, 4, 3], 
                 convs=[[16, 2, 1], [32, 1, 1]], 
                 fully_connected=[128],
                 optimizer=tf.train.AdamOptimizer(2.5e-4),
                 model_name="DQN"):
        
        super(SnakeDQNAgentFisherI, self).__init__(state_shape=state_shape, convs=convs,
                                                   fully_connected=fully_connected,
                                                   optimizer=optimizer, model_name=model_name)
        
        print("Constructing symbolic matvecs...")
        self.construct_matvecs()
        print("Done")
    
    def construct_matvecs(self):
        
        var_list = tf.trainable_variables()
        
        n = len(var_list) // 2
        
        self.num_agent_vars = n 
        
        agent_vars = var_list[:n]
        
        self.agent_vars = agent_vars
        
        loss_ = self.agent_net.loss
        
        grads = tf.gradients(loss_, agent_vars)
           
        self.fisher_info = tf.add_n([tf.reduce_sum(g * g) for g in grads])
        
    
    def get_fisher_info(self, sess, s, r, a, s_, end):
        
        max_actions = self.agent_net.get_q_argmax(sess, [s_])
        q_values = self.target_net.get_q_values(sess, [s_])
        double_q = q_values[0, max_actions]
        targets = r + (self.gamma * double_q * end)
         
        

        feed_dict = {}
        feed_dict[self.agent_net.input_states] = [s,]
        feed_dict[self.agent_net.input_actions] = [a,]
        feed_dict[self.agent_net.targets] = targets

        fi = sess.run(self.fisher_info, feed_dict=feed_dict)
  
        return fi
        

    def train(self,
              gpu_id=0,
              batch_size=32,
              exploration="e-greedy",
              agent_update_freq=4,
              target_update_freq=5000,
              tau=1,
              max_num_epochs=50000,
              performance_print_freq=500,
              save_freq=10000, 
              from_epoch=0):
        
        config = self.gpu_config(gpu_id)
        target_ops = self.update_target_graph(tau)
        self.batch_size = batch_size
        
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
            
            running_singular_values = Queue(5000)
            
            while num_epochs < max_num_epochs:
                
                train_ep_reward = 0
                
                # reset the environment / start new game
                s = self.train_env.reset()
                for time_step in range(self.max_ep_length):
                    
                    # choose action e-greedily
                    a = self.choose_action(sess, s, exploration)
                        
                    # make step in the environment    
                    s_, r, end = self.train_env.step(a)
                    
                    fisher_info = self.get_fisher_info(sess, s, r, a, s_, 1.0 - end)
                    
                    if running_singular_values.qsize() == running_singular_values.maxsize:
                        running_singular_values.get()
                                           
                    running_singular_values.put(fisher_info)
                    
                    if fisher_info < np.median(np.array(list(running_singular_values.queue))):
                        self.rep_buffer.push(s, a, r, s_, end)
                     
                              
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
                        self.update_agent_weights(sess, batch)
                        
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