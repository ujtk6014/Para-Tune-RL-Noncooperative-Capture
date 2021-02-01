import numpy as np
import random

from collections import deque


# namedtupleを生成
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state','done'))

class BasicBuffer:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, state, action, state_next, reward,done):
        '''transition = (state, action, state_next, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, state_next, reward, done)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)


# class BasicBuffer:
#     def __init__(self, max_size):
#         self.max_size = max_size
#         self.buffer = deque(maxlen=max_size)

#     def push(self, state, action, reward, next_state, done):
#         experience = (state, action, np.array([reward]), next_state, done)
#         self.buffer.append(experience)

#     def sample(self, batch_size):
#         state_batch = []
#         action_batch = []
#         reward_batch = []
#         next_state_batch = []
#         done_batch = []

#         batch = random.sample(self.buffer, batch_size)

#         for experience in batch:
#             state, action, reward, next_state, done = experience
#             state_batch.append(state)
#             action_batch.append(action)
#             reward_batch.append(reward)
#             next_state_batch.append(next_state)
#             done_batch.append(done)

#         return np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), np.array(done_batch).reshape(-1,1)

#     def __len__(self):
#         return len(self.buffer)