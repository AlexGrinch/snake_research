import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import tensorflow as tf
from methods import DistQNetwork
from environments import Snake

class AgentViz:
    
    def __init__(self, sess, path, agent_net, grid_size=(2, 2), figsize=(15, 15), max_frames=1000):
        self.path = path
        self.sess = sess
        self.h = grid_size[0]
        self.w = grid_size[1]
        self.max_frames = max_frames
        fig, ax = plt.subplots(self.h, self.w, figsize=figsize)
        self.fig = fig
        self.ax = ax
        for i in range(self.h):
            for j in range(self.w):
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
        
        env = Snake()
        saver = tf.train.Saver()
        saver.restore(sess, path)
        
        self.img_lists = []
        for i in range(self.h):
            for j in range(self.w):
                tmp_imgs = []
                s = env.reset()
                for k in range(self.max_frames):
                    a = agent_net.get_q_argmax(sess, [s])[0]
                    s, r, done = env.step(a)
                    im = s[:, :, 0]
                    tmp_imgs.append(im, )
                    if done: break
                self.img_lists.append(tmp_imgs)
        
        
        
        max_len = -1
        for img_list in self.img_lists:
            if len(img_list) > max_len:
                max_len = len(img_list)
        
        self.max_frames = min(max_len, self.max_frames)
        
        new_lists = []
        for img_list in self.img_lists:
            if len(img_list) < self.max_frames:
                while len(img_list) < self.max_frames:
                    img_list.append(img_list[-1])
            new_lists.append(img_list)
        self.img_lists = new_lists
        for l in self.img_lists:
            print (len(l))
        
    
    def init(self):
        self.l = []
        for i in range(self.h):
            for j in range(self.w):
                self.l.append(self.ax[i, j].imshow(self.img_lists[i * self.w + j][0]))
    
    def __call__(self, k):
        
        if k == 0:
            return self.init()
        
        else:
            for i in range(self.h):
                for j in range(self.w):
                    self.l[i * self.w + j].set_data(self.img_lists[i * self.w + j][k])
           

                       
                
                
    
    
