import numpy as np
import matplotlib.pyplot as plt

class Snake:
    
    def __init__(self, grid_size=(8, 8)):
        """
        Classic Snake game implemented as Gym environment.
        
        Parameters
        ----------
        grid_size: tuple
            tuple of two parameters: (height, width)
        """
        
        self.height, self.width = grid_size
        self.state = np.zeros(grid_size)
        self.x, self.y = [], []
        self.dir = None
        
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        Returns
        -------
        observation: numpy.array of size (width, height, 1)
            the initial observation of the space.
        """
        
        self.state = np.zeros((self.height, self.width))
        
        
        
        
        x_tail = np.random.randint(self.height)
        y_tail = np.random.randint(self.width)
        
        xs = [x_tail, ]
        ys = [y_tail, ]
        
        for i in range(2):
            nbrs = self.get_neighbors(xs[-1], ys[-1])
            while 1:
                idx = np.random.randint(0, len(nbrs))
                x0 = nbrs[idx][0]
                y0 = nbrs[idx][1]
                occupied = [list(pt) for pt in zip(xs, ys)]
                if not [x0, y0] in occupied:
                    xs.append(x0)
                    ys.append(y0)
                    break
        
        for x_t, y_t in list(zip(xs, ys)):
            self.state[x_t, y_t] = 1
     
        self.generate_food()
        self.x = xs
        self.y = ys

        return self.get_state()
    
    def step(self, a):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        
        Args
        ----
        action: int from {0, 1, 2, 3}
            an action provided by the environment
            
        Returns
        -------
        observation: numpy.array of size (width, height, 1)
            agent's observation of the current environment
        reward: int from {-1, 0, 1}
            amount of reward returned after previous action
        done: boolean
            whether the episode has ended, in which case further step() 
            calls will return undefined results
        """
        
        self.update_dir()
        x_, y_ = self.next_cell(self.x[-1], self.y[-1], a)

        # snake dies if hitting the walls
        if x_ < 0 or x_ == self.height or y_ < 0 or y_ == self.width:
            return self.get_state(), -1, True
        
        # snake dies if hitting its tail with head
        if self.state[x_, y_] == 1:
            return self.get_state(), -1, True
        
        self.x.append(x_)
        self.y.append(y_)
        
        # snake elongates after eating a food
        if self.state[x_, y_] == 3:
            self.state[x_, y_] = 1
            if self.generate_food():
                return self.get_state(), 1, False
            else:
                return self.get_state(), 1, True

        # snake moves forward if cell ahead is empty
        if self.state[x_, y_] == 0:
            self.state[x_, y_] = 1
            self.state[self.x[0], self.y[0]] = 0
            self.x = self.x[1:]
            self.y = self.y[1:]
            return self.get_state(), 0, False  
        
    def get_state(self):
        state = self.state.copy()
        state[self.x[-1], self.y[-1]] = 2
        return state.reshape((self.height, self.width, 1))
        
    def generate_food(self):
        free = np.where(self.state == 0)
        if free[0].size == 0:
            return False
        else:
            idx = np.random.randint(free[0].size)
            self.state[free[0][idx], free[1][idx]] = 3
            return True
        
    def next_cell(self, i, j, a):
        if a == 0: 
            return i+self.dir[0], j+self.dir[1]
        if a == 1: 
            return i-self.dir[1], j+self.dir[0]
        if a == 2: 
            return i+self.dir[1], j-self.dir[0]
        
    def plot_state(self):
        img = self.get_state()[:,:,0]
        plt.imshow(img, vmin=0, vmax=4, interpolation='nearest')
        
    def get_neighbors(self, i, j):
        """
        Get all the neighbors of the point (i, j)
        (excluding (i, j))
        """
        h = self.height
        w = self.width
        nbrs = [[i + k, j + m] for k in [-1, 0, 1] for m in [-1, 0, 1]
                if i + k >=0 and i + k < h and j + m >= 0 and j + m < w
                and not (k == m) and not (k == -m)]
        return nbrs
    
    def update_dir(self):
        x_dir = self.x[-1] - self.x[-2]
        y_dir = self.y[-1] - self.y[-2]
        self.dir = (x_dir, y_dir)