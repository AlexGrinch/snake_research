import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import tensorflow as tf
from methods import DistQNetwork
from environments import Snake

class AgentViz:
    
    def __init__(self, games, grid_size=(2, 2), figsize=(15, 15), state2im=None):

        self.h = grid_size[0]
        self.w = grid_size[1]
        if self.h * self.w != len(games):
            raise ValueError('len(games) must be equal to prod(grid_size), got {} and {}'.format(self.h * self.w, len(games)))
        
        if state2im is None:
            self.state2im = lambda x: sum([x[:, :, i] * (i + 1) for i in range(5)])
            
        else:
            self.stat2im = state2im
                                         
        fig, ax = plt.subplots(self.h, self.w, figsize=figsize)
        self.fig = fig
        self.ax = ax
        for i in range(self.h):
            for j in range(self.w):
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
        
    
       
        
        
        max_len = -1
        for game in games:
            if len(game) > max_len:
                max_len = len(game)
        
        self.max_frames = max_len
        
        new_games = []
        for game in games:
            if len(game) < max_len:
                while len(game) < max_len:
                    game.append(game[-1])
            new_games.append(game)
        self.games = new_games
        
    
    def init(self):
        self.l = []
        for i in range(self.h):
            for j in range(self.w):
                self.l.append(self.ax[i, j].imshow(self.state2im(self.games[i * self.w + j][0])))
    
    def __call__(self, k):
        
        if k == 0:
            return self.init()
        
        else:
            for i in range(self.h):
                for j in range(self.w):
                    self.l[i * self.w + j].set_data(self.state2im(self.games[i * self.w + j][k]))
           

                       
                
                
    
    
