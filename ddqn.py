import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple
import copy
import gym

EPISODES = 10000
STEPS = 200
EPSILON = 1
EPSILON_MIN = 0.01
EPSILON_DISCOUNT = 0.99
TRAIN_START = 1
LEARNING_RATE = 0.01
NUM_STATES = 2
NUM_ACTIONS = 3
NODES = 32
BATCH_SIZE = 1
CAPACITY = 10000
PATH = '.\ddqn_saved.pth'
DATA = namedtuple('DATA', ('state', 'action', 'reward', 'next_state', 'done'))

class DB:
    def __init__(self):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0
        self.batch_size = BATCH_SIZE

    def __len__(self):
        return len(self.memory)

    def save(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = DATA(state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity

    def sampling(self):
        return random.sample(agent.DB.memory, self.batch_size)

class Brain:
    def __init__(self):
        self.state_space = NUM_STATES
        self.action_space = NUM_ACTIONS
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.nodes = NODES

    def modeling_NN(self):
        model = nn.Sequential()
        model.add_module('L_in', nn.Linear(self.state_space, self.nodes))
        model.add_module("relu1", nn.ReLU())
        model.add_module('L_mid', nn.Linear(self.nodes, self.nodes))
        model.add_module("relu2", nn.ReLU())
        model.add_module('L_out', nn.Linear(self.nodes, self.action_space))
        return model

    def modeling_OPTIM(self, Q):
        optimizer = Adam(Q.parameters(), lr=LEARNING_RATE)
        return optimizer

    def action(self, state):
        self.epsilon = EPSILON * EPSILON_DISCOUNT if self.epsilon > self.epsilon_min else self.epsilon_min
        state = torch.tensor(state).float()

        #이용
        if random.uniform(0, 1) < self.epsilon:
            agent.Q.eval()
            with torch.no_grad():
                action = agent.Q(state)
                action = torch.argmax(action).item()
        #탐색
        else:
            action = random.randrange(0, self.action_space)

        return action

    def update_Q(self, batch):
        batch = DATA(*zip(*batch))
        state_serial = batch.state
        action_serial = batch.action
        reward_serial = batch.reward
        next_serial = batch.next_state
        done_serial = batch.done

        print(state_serial,action_serial)

        exit()

        agent.Q.train()
        loss = F.mse_loss()
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()


    def update_TQ(self):
        self.TQ = copy.deepcopy(self.Q)

class Agent:
    def __init__(self):
        self.Brain = Brain()
        self.DB = DB()
        self.Q = self.Brain.modeling_NN()
        self.TQ = copy.deepcopy(self.Q)
        self.optimizer = self.Brain.modeling_OPTIM(self.Q)

    def action_request(self, state):
        return self.Brain.action(state)

    def save_to_DB(self, state, action, reward, next_state, done):
        self.DB.save(state, action, reward, next_state, done)

    def train(self):
        self.Brain.update_Q(self.DB.sampling())

    def update(self):
        self.Brain.update_TQ()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = Agent()
    for E in range(EPISODES):
        state = env.reset()
        score = 0
        for S in range(STEPS):
            env.render()
            action = agent.action_request(state)
            next_state, reward, done, info = env.step(action)
            agent.save_to_DB(state, action, reward, next_state, done)
            if agent.DB.__len__() > TRAIN_START:
                agent.train()
            if done:
                print('EPISODE NO.', E, '  SCORE : ', score, '  STEPS : ', S, '  EPSILON : ', agent.Brain.epsilon)
                break
            else:
                state = next_state
                score += reward
        agent.update()

    print('모델을 저장하시겠습니까? [Y/N]')
    answer = input()
    if answer == 'y' or answer == 'Y':
        torch.save(agent.Brain.Q.state_dict(), PATH)


