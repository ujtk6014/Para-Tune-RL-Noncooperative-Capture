import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy.random as rd

from buffer import BasicBuffer, TDerrorMemory
from noise import OUNoise
import numpy as np

# namedtupleを生成
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state','done'))


#initialization 2021/1/11
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class QNetDuel(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.net__head = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
        )
        self.net_val = nn.Sequential(  # value
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )
        self.net_adv = nn.Sequential(  # advantage value
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )

    def forward(self, state):
        x = self.net__head(state)
        val = self.net_val(x)
        adv = self.net_adv(x)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

class DDQNAgent:
    def __init__(self, env, gamma, tau, buffer_maxlen, learning_rate, train, decay):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.tau = tau
        self.gamma = gamma
        self.lr = learning_rate
        self.train = train


        self.mid_dim = 512
        self.explore_rate = 0.5
        self.softmax = nn.Softmax(dim=1)

        # initialize actor and critic networks
        self.q_net = QNetDuel(self.obs_dim, self.action_dim,self.mid_dim).to(self.device)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.q_net.train()

        self.q_net_target = QNetDuel(self.obs_dim, self.action_dim,self.mid_dim).to(self.device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.eval()

        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = BasicBuffer(buffer_maxlen)
        # TD誤差のメモリオブジェクトを生成
        self.td_error_memory = TDerrorMemory(buffer_maxlen)
    
    def get_action(self, state, episode=0):
        epsilon = 0.5 *( 1/(0.1*episode + 1) )
        state = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
        actions = self.q_net(state)
        # if self.train == True:
        #     if rd.rand() < self.explore_rate:
        #         a_prob = self.softmax(actions).cpu().data.numpy()[0]
        #         a_int = rd.choice(self.action_dim, p=a_prob)
        #     else:
        #         a_int = actions.argmax(dim=1).cpu().data.numpy()[0]
        # else:
        #     a_int = actions.argmax(dim=1).cpu().data.numpy()[0]
        if self.train == True:
            if epsilon < rd.rand():
                a_int = actions.argmax(dim=1).cpu().data.numpy()[0]
            else:
                a_prob = self.softmax(actions).cpu().data.numpy()[0]
                a_int = np.int64(rd.choice(self.action_dim, p=a_prob))
        else:
            a_int = actions.argmax(dim=1).cpu().data.numpy()[0]
        return a_int

    def update(self,batch_size, episode):
        if episode < 10:
            transitions = self.replay_buffer.sample(batch_size)
        else:
            # TD誤差に応じてミニバッチを取り出すに変更
            indexes = self.td_error_memory.get_prioritized_indexes(batch_size)
            transitions = [self.replay_buffer.memory[n] for n in indexes]

        batch = Transition(*zip(*transitions))
        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.reward
        next_state_batch = batch.next_state
        masks = batch.done
        # state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)

        with torch.no_grad():
            next_Q = self.q_net_target(next_state_batch).max(dim=1, keepdim=True)[0]
            expected_Q = reward_batch + self.gamma * next_Q
        
        self.q_net.train()
        a_ints = action_batch.type(torch.long)
        q_eval = self.q_net(state_batch).gather(1, a_ints)
        critic_obj = self.criterion(q_eval, expected_Q)

        self.q_net_optimizer.zero_grad()
        critic_obj.backward()
        self.q_net_optimizer.step()

        if episode % 2 ==0:
            # update target networks
            for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
                target_param.data.copy_(param.data)

            # target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
    def get_td_error(self, state, action, next_state, reward):
        state = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state.flatten()).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action.flatten()).to(self.device)
        with torch.no_grad():
            next_Q = self.q_net_target(next_state).max(dim=1, keepdim=True)[0]
            expected_Q = reward + self.gamma * next_Q
        a_ints = action.type(torch.long)
        q_eval = self.q_net(state).gather(1, a_ints.unsqueeze(1))
        td_error = expected_Q - q_eval

        return td_error.detach().to('cpu').numpy()

    def update_td_error_memory(self):  # PrioritizedExperienceReplayで追加
        '''TD誤差メモリに格納されているTD誤差を更新する'''

        # ネットワークを推論モードに切り替える
        self.q_net.eval()
        self.q_net_target.eval()

        # 全メモリでミニバッチを作成
        transitions = self.replay_buffer.memory
        batch = Transition(*zip(*transitions))

        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.reward
        next_state_batch = batch.next_state
        masks = batch.done
        # state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)

        with torch.no_grad():
            next_Q = self.q_net_target(next_state_batch).max(dim=1, keepdim=True)[0]
            expected_Q = reward_batch + self.gamma * next_Q

        a_ints = action_batch.type(torch.long)
        q_eval = self.q_net(state_batch).gather(1, a_ints)

        # TD誤差を求める
        td_errors = expected_Q - q_eval

        # TD誤差メモリを更新、Tensorをdetach()で取り出し、NumPyにしてから、Pythonのリストまで変換
        self.td_error_memory.memory = td_errors.squeeze().detach().to('cpu').numpy().tolist()


