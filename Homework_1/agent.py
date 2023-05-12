import random
import numpy as np
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.as_tensor(state).to(device)
        action = self.model(state).to(device)
        return np.argmax(action.cpu().detach().numpy())
