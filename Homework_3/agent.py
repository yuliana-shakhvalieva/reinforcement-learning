import random
import numpy as np
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        state = torch.tensor(np.array([state]), device=device).float()
        action = self.model.forward(state)
        return action.cpu().numpy()[0]

    def reset(self):
        pass

