import numpy as np


class dl_in_gen():
    
    def __init__(self, T, state_size, action_size):
        self.hist_size = state_size+action_size+1
        self.deep_learning_in_size = self.hist_size*T
        self.reset()
        
    def reset(self):
        self.deep_learning_input = np.zeros(self.deep_learning_in_size)
        
    def dl_input(self, states, actions):
        
        for state, action in zip(states, actions):
            state_t = np.concatenate((action, state))
            self.deep_learning_input = np.roll(self.deep_learning_input, -self.hist_size)
            self.deep_learning_input[-self.hist_size:] = state_t
        return self.deep_learning_input