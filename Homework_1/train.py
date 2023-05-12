from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy
from collections import deque

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
BUFFER_MAXLEN = 2 ** 14
random.seed(456)

class BatchDQN:
    def __init__(self, batch):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.flags = []
        self._parse_data(batch)
        self._tensor_convert()

    def _parse_data(self, batch):
        for value in batch:
            self.states.append(value[0])
            self.actions.append(value[1])
            self.next_states.append(value[2])
            self.rewards.append(value[3])
            self.flags.append(value[4])
            
    def _tensor_convert(self):
        self.states = torch.from_numpy(np.array(self.states)).float().to(device)
        self.actions = torch.from_numpy(np.array(self.actions)).int().to(device)
        self.next_states = torch.from_numpy(np.array(self.next_states)).float().to(device)
        self.rewards = torch.from_numpy(np.array(self.rewards)).float().to(device)
        self.flags = torch.from_numpy(np.array(self.flags)).to(device)


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0  # Do not change
        self.model = nn.Sequential(nn.Linear(state_dim, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, action_dim)).to(device)
        self.target_network = copy.deepcopy(self.model)
        self.buffer = deque(maxlen=BUFFER_MAXLEN)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.loss = nn.SmoothL1Loss()

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch_array = []
        for _ in range(BATCH_SIZE):
            idx = random.randint(0, len(self.buffer) - 1)
            batch_array.append(self.buffer[idx])
        return BatchDQN(batch_array)

    def train_step(self, batch):
        # Use batch to update DQN's network.
        non_final_mask = torch.tensor(tuple(map(lambda s: s == False, batch.flags)), device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s, f in zip(batch.next_states, batch.flags.unsqueeze(1)) if f == False]).to(device)
        action_values = self.model(batch.states).gather(1, batch.actions.unsqueeze(1).type('torch.LongTensor').to(device)).to(device)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = torch.max(self.target_network(non_final_next_states), axis=1)[0]
        expected_action_values = (next_state_values * GAMMA) + batch.rewards
        loss = self.loss(action_values, expected_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target_network.load_state_dict(self.model.state_dict())

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = torch.as_tensor(state).to(device)
        if not target:
            action = self.model(state).to(device)
        else:
            action = self.target_network(state).to(device)
        return np.argmax(action.cpu().detach().numpy())

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()[0]
        total_reward = 0.

        while not done:
            state, reward, done, _, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()[0]
 
    for _ in tqdm(range(INITIAL_STEPS)):
        action = env.action_space.sample()

        next_state, reward, done, _, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        state = next_state if not done else env.reset()[0]

    for i in range(TRANSITIONS):
        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        state = next_state if not done else env.reset()[0]
        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            dqn.save()