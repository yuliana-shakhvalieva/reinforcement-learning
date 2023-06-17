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
from tqdm import trange

GAMMA = 0.99
TAU = 0.002
CLIP = 0.2
CRITIC_LR = 2e-4
ACTOR_LR = 2e-4
DEVICE = "cuda"
BATCH_SIZE = 256
ENV_NAME = "AntBulletEnv-v0"
TRANSITIONS = 2000000

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
            nn.Linear(256, action_dim),
            nn.Tanh())
        
    def forward(self, state):
        return self.model(state)
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1))
    
    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=CRITIC_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=CRITIC_LR)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        
        self.replay_buffer = deque(maxlen=300000)
        self.critic_loss = nn.MSELoss()


    def update(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > BATCH_SIZE * 16:
            
            # Sample batch
            transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer)-1)] for _ in range(BATCH_SIZE)]
            state, action, next_state, reward, done = zip(*transitions)
            state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), device=DEVICE, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
            done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)

            eps = torch.empty(action.size()).normal_(mean=0, std=0.5).to(DEVICE)
            next_action = self.target_actor.forward(next_state) + eps
            
            expexted_action = reward + (1 - done) * GAMMA * torch.min(self.target_critic_1.forward(next_state, next_action), 
                                                         self.target_critic_2.forward(next_state, next_action))
            
            action_1 = self.critic_1.forward(state, action)
            action_2 = self.critic_2.forward(state, action)
            
            loss_critic_1 = self.critic_loss(action_1, expexted_action)
            self.critic_1_optim.zero_grad()
            loss_critic_1.backward(retain_graph=True)
            self.critic_1_optim.step()
            
            loss_critic_2 = self.critic_loss(action_2, expexted_action)
            self.critic_2_optim.zero_grad()
            loss_critic_2.backward(retain_graph=True)
            self.critic_2_optim.step()
            
            new_action = self.actor.forward(state)
            pred_action = self.critic_1.forward(state, new_action)
            actor_loss = - torch.mean(pred_action)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step() 
            
            soft_update(self.target_critic_1, self.critic_1)
            soft_update(self.target_critic_2, self.critic_2)
            soft_update(self.target_actor, self.actor)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=DEVICE)
            return self.actor(state).cpu().numpy()[0]

    def save(self, reward):
        torch.save(self.actor, f"./agent/agent_{reward}.pkl")


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns

if __name__ == "__main__":
    env = make(ENV_NAME)
    test_env = make(ENV_NAME)
    td3 = TD3(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    eps = 0.2
    
    for i in trange(TRANSITIONS):
        steps = 0
        
        #Epsilon-greedy policy
        action = td3.act(state)
        action = np.clip(action + eps * np.random.randn(*action.shape), -1, +1)

        next_state, reward, done, _ = env.step(action)
        td3.update((state, action, next_state, reward, done))
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(test_env, td3, 5)
            reward_mean = np.mean(rewards)
            td3.save(round(reward_mean, 2))