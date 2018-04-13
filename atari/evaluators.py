import os
import time
import random
import numpy as np
import ray

import tensorflow as tf
import tensorflow.contrib.layers as layers

import gym
from gym import spaces
from PIL import Image
from collections import deque, namedtuple

from IPython import display
import matplotlib.pyplot as plt

from atari_wrappers import wrap_deepmind
from agents import *

@ray.remote
class Worker:
    def __init__(self, worker_index, game_id):
        self.worker_index = worker_index
        self.env = wrap_deepmind(gym.make(game_id), clip_rewards=False)
        self.ep_rewards = []
           
    def reset(self):
        self.ep_reward = 0
        return self.env.reset()
    
    def get_ep_rewards(self):
        return self.ep_rewards

    def make_step(self, a):
        s_, r, done, _ = self.env.step(a)
        self.ep_reward += r
        if done:
            if self.env.unwrapped.ale.lives() == 0:
                self.ep_rewards.append(self.ep_reward)
                self.ep_reward = 0
            s_ = self.env.reset()
        return s_
    
    
class AtariEvaluator:
    
    def __init__(self, game_id, agent, num_cpus=8):
        self.agent = agent
        ray.init(num_cpus=num_cpus, num_gpus=1)
        num_workers = num_cpus
        self.workers = [Worker.remote(worker_id, game_id) 
                   for worker_id in range(num_workers)]
        
    def run_n_frames(self, gpu_id, num_frames=1000, from_epoch=0):
        
        init_states = [worker.reset.remote() 
                       for worker in self.workers]
        states = ray.get(init_states)
        
        config = self.agent.gpu_config(gpu_id)
        with tf.Session(config=config) as sess:
            self.agent.saver.restore(sess, self.agent.path+"/model-"+str(from_epoch))
            for i in range(num_frames):
                actions = self.agent.agent_net.get_q_argmax(sess, states)
                ray_states = [worker.make_step.remote(actions[i])
                              for (i, worker) in enumerate(self.workers)]
                states = ray.get(ray_states)
                
        rews = [worker.get_ep_rewards.remote() for worker in self.workers]
        ray_ep_rewards = ray.get(rews)
        rewards = []
        for i in ray_ep_rewards:
            rewards += i
        return rewards