import pybullet_envs
from gym import make
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import random
import copy
import pickle

GAMMA = 0.99
TAU = 0.005
CRITIC_LR = 3e-4
ACTOR_LR = 2e-4
DEVICE = "cuda"
BATCH_SIZE = 2048
ITERATIONS = 3000000
RLPC_EPS = 0.5
AWAC_LAMBDA = 1

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim))
        
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        
    def _get_policy(self, state):
        mean = self.model(state)
        log_std = self._log_std.clamp(-20, 2)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state, action):
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state):
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(-1, 1)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class Algo:
    def __init__(self, state_dim, action_dim, data):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=CRITIC_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=CRITIC_LR)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        
        self.replay_buffer = list(data)
        self.critic_1_loss = nn.MSELoss()
        self.critic_2_loss = nn.MSELoss()
        
        self.actor_params_sizes = [p.numel() for p in self.actor.parameters()]
        self.critic_1_params_sizes = [p.numel() for p in self.critic_1.parameters()]
        self.critic_2_params_sizes = [p.numel() for p in self.critic_2.parameters()]
        

    def update(self):
        if len(self.replay_buffer) > BATCH_SIZE * 16:
            
            # Sample batch
            transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer)-1)] for _ in range(BATCH_SIZE)]
            state, action, next_state, reward, done = zip(*transitions)
            state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), device=DEVICE, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
            done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)
            
            eps = torch.empty(action.size()).normal_(mean=0, std=0.2).to(DEVICE)
            next_action = self.target_actor.forward(next_state)[0] + eps
            
            expexted_action = reward + (1 - done) * GAMMA * torch.min(self.target_critic_1.forward(next_state, next_action), 
                                                         self.target_critic_2.forward(next_state, next_action))
            
            action_1 = self.critic_1.forward(state, action)
            action_2 = self.critic_2.forward(state, action)
            
            loss_critic_1 = self.critic_1_loss(action_1, expexted_action)
            self.critic_1_optim.zero_grad()
            loss_critic_1.backward(retain_graph=True)
            self.critic_1_optim.step()
            
            loss_critic_2 = self.critic_2_loss(action_2, expexted_action)
            self.critic_2_optim.zero_grad()
            loss_critic_2.backward(retain_graph=True)
            self.critic_2_optim.step()
            
            new_action, _ = self.actor.forward(state)
            v = torch.min(self.critic_1.forward(state, new_action), self.critic_2.forward(state, new_action))
            q = torch.min(self.critic_1.forward(state, action), self.critic_2.forward(state, action))
            adv = q - v
            weights = torch.clamp_max(torch.exp(adv / AWAC_LAMBDA), 100)
                                                     
            action_log_prob = self.actor.log_prob(state, action)
            actor_loss = (-action_log_prob * weights).mean()
                                                     
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step() 

            # RLPC
            with torch.no_grad():
                actor_params = torch.cat([p.view(-1) for p in self.actor.parameters()])
                target_actor_params = torch.cat([p.view(-1) for p in self.target_actor.parameters()])
                actor_diff = actor_params - target_actor_params
                actor_norm = actor_diff.norm(p=2)
                if actor_norm > RLPC_EPS:
                    actor_ratio = RLPC_EPS / actor_norm
                    actor_params = target_actor_params + actor_ratio * actor_diff
                    for p, new_p in zip(self.actor.parameters(), actor_params.split(self.actor_params_sizes)):
                        p.data.copy_(new_p.view(p.size()))

                critic_1_params = torch.cat([p.view(-1) for p in self.critic_1.parameters()])
                target_critic_1_params = torch.cat([p.view(-1) for p in self.target_critic_1.parameters()])
                critic_diff_1 = critic_1_params - target_critic_1_params
                critic_norm_1 = critic_diff_1.norm(p=2)
                if critic_norm_1 > RLPC_EPS:
                    critic_ratio_1 = RLPC_EPS / critic_norm_1
                    critic_params_1 = target_critic_1_params + critic_ratio_1 * critic_diff_1
                    for p, new_p in zip(self.critic_1.parameters(), critic_params_1.split(self.critic_1_params_sizes)):
                        p.data.copy_(new_p.view(p.size()))
                        
                critic_2_params = torch.cat([p.view(-1) for p in self.critic_2.parameters()])
                target_critic_2_params = torch.cat([p.view(-1) for p in self.target_critic_2.parameters()])
                critic_diff_2 = critic_2_params - target_critic_2_params
                critic_norm_2 = critic_diff_2.norm(p=2)
                if critic_norm_2 > RLPC_EPS:
                    critic_ratio_2 = RLPC_EPS / critic_norm_2
                    critic_params_2 = target_critic_2_params + critic_ratio_2 * critic_diff_2
                    for p, new_p in zip(self.critic_2.parameters(), critic_params_2.split(self.critic_2_params_sizes)):
                        p.data.copy_(new_p.view(p.size()))

            soft_update(self.target_critic_1, self.critic_1)
            soft_update(self.target_critic_2, self.critic_2)
            soft_update(self.target_actor, self.actor)

            return actor_loss.cpu().detach().numpy()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=DEVICE)
            return self.actor(state).cpu().numpy()[0]

    def save(self, loss):
        a = np.round(float(loss), 2)
        torch.save(self.actor, f"./agents/agent_{a}.pkl")


if __name__ == "__main__":
    with open("./offline_data.pkl", "rb") as f:
        data = pickle.load(f)

    algo = Algo(state_dim=32, action_dim=8, data=data)

    for i in trange(ITERATIONS):
        steps = 0
        loss = algo.update()
        if (i + 1) % (ITERATIONS//1000) == 0:
            algo.save(loss)