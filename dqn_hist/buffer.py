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

    def push(self, state, action, reward, state_next,done):
        '''transition = (state, action, state_next, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, reward, state_next, done)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)

# TD誤差を格納するメモリクラスを定義します

TD_ERROR_EPSILON = 0.0001  # 誤差に加えるバイアス


class TDerrorMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, td_error):
        '''TD誤差をメモリに保存します'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)

    def get_prioritized_indexes(self, batch_size):
        '''TD誤差に応じた確率でindexを取得'''

        # TD誤差の和を計算
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)  # 微小値を足す

        # batch_size分の乱数を生成して、昇順に並べる
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        # 作成した乱数で串刺しにして、インデックスを求める
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                print(type(self.memory[idx]))
                tmp_sum_absolute_td_error += (
                    abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1

            # 微小値を計算に使用した関係でindexがメモリの長さを超えた場合の補正
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)

        return indexes

    def update_td_error(self, updated_td_errors):
        '''TD誤差の更新'''
        self.memory = updated_td_errors

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