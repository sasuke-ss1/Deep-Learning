import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


env = gym.make('CartPole-v1').unwrapped
plt.ion()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))
print(device)

class ReplayMemory():
    def __init__(self, memory_size):
        self.memory = deque([], maxlen = memory_size)
    
    def push(self, *args):
        self.memory.append(Experience(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return(len(self.memory))
    
class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(observation_space, 164)
        self.l2 = nn.Linear(164, action_space)
         
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.l1(x))
        x =self.l2(x)
        return x
    
    
# Hyperparams
batch_size = 128
lr = 1e-3
episodes = 200
E_start = 0.9
E_end = 0.05
E_decay = 200
gamma = 0.75
Policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
'''
Target_net = DQN()
Traget_net.load_state_dict(Policy_net.state_dict())
'''
memory = ReplayMemory(10000)
optimizer = optim.Adam(Policy_net.parameters(), lr = lr)
steps = 0
episode_duration = []


def select_action(state):
    global steps
    sample = random.random()
    eps = E_end + (E_start - E_end) * math.exp(-1. * steps/ E_decay)
    if sample < eps:
        return Policy_net(state).max(1)[1].view(1, 1)
    else:
        return(torch.tensor([[random.randrange(2)]], dtype = torch.long, device = device))

def plot():
    plt.figure(2)
    plt.clf
    plt.title("Training")
    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    ed = np.array(episode_duration)
    plt.plot(ed)
    plt.pause(0.001)
    
def run(env, episode):
    state = env.reset()
    steps = 0
    while True:
        env.render()
        action = select_action(torch.tensor([state], dtype = torch.float32, device = device))
        next_state, reward, done, _ = env.step(int(action.item()))

        if done:
            if steps < 30 and episode < 50:
                reward -= 10
            elif steps < 200 and episode < 120:
                reward -= 10
            else:
                reward = -1
        if steps > 100:
            reward += 2
        if steps > 200:
            reward += 2
        if steps > 300:
            reward += 4
        if steps > 800:
            reward += 100
            
        memory.push(torch.cuda.FloatTensor([state]), action, torch.cuda.FloatTensor([next_state])\
                    , torch.cuda.FloatTensor([reward]))
            
        opt()
        state = next_state
        steps += 1
        if done or steps > 1000:
            episode_duration.append(steps + 1)
            plot()
            break


def opt():
    if len(memory) < batch_size:
        return
    Experience = memory.sample(batch_size)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*Experience)
    
    batch_state = torch.cat(batch_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward)
    batch_next_state = torch.cat(batch_next_state)
    
    ##############dekh lena#################
    current_q_values = Policy_net(batch_state).gather(1, batch_action)
    max_next_q_values = Policy_net(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (gamma * max_next_q_values)
    expected_q_values = expected_q_values.view(-1, 1)
    loss = F.smooth_l1_loss(current_q_values, expected_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for i in range(episodes):
    run(env, i)
env.close()
plt.ioff()


    
