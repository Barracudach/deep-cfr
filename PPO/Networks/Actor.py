import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, output_size):
        super(Actor, self).__init__()
        self.output_size=output_size
        pass

    def forward(self,state):
        return np.random.rand(self.output_size)
    
 
