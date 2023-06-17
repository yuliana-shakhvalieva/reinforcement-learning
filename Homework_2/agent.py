import random
import numpy as np
import os
import torch

seed = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), device=device).float()
            action, pure_action, distr = self.model.act(state)
        return action.cpu().numpy()[0]

    def reset(self):
        pass
