import logging
import math

import gym
import numpy as np

from gym import make as gym_make
from gym import spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)


class SatelliteContinuousEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    #----------toolbox----------
    def skew(self,vec): 
        # create a skew symmetric matrix from a vector
        mat = np.array([[0, -vec[2], vec[1]],
                        [vec[2], 0, -vec[0]],
                        [-vec[1], vec[0], 0]])
        return mat

    def euler2dcm(self,euler):
        phi   = euler[2] # Z axis Yaw
        theta = euler[1] # Y axis Pitch
        psi   = euler[0] # X axis Roll
        rotx = np.array([[1, 0, 0],
                        [0, np.cos(psi), np.sin(psi)],
                        [0, -np.sin(psi), np.cos(psi)]])
        roty = np.array([[np.cos(theta), 0, -np.sin(theta)],
                        [0, 1, 0],
                        [np.sin(theta), 0, np.cos(theta)]])
        rotz = np.array([[np.cos(phi), np.sin(phi), 0],
                        [-np.sin(phi), np.cos(phi), 0],
                        [0, 0, 1]])
        dcm = rotx @ roty @ rotz
        return dcm

    def dcm2euler(self,dcm):
        # calculate 321 Euler angles [rad] from DCM
        sin_theta = - dcm[0,2]
        if sin_theta == 1 or sin_theta == -1:
            theta = np.arcsin(sin_theta)
            psi = 0
            sin_phi = -dcm(2,1)
            phi = np.arcsin(sin_phi)
        else:
            theta = np.arcsin(sin_theta)
            phi = np.arctan2(dcm[1,2], dcm[2,2])
            psi = np.arctan2(dcm[0,1], dcm[0,0])
            
        euler = np.array([phi, theta, psi])
        return euler

    def dcm2quaternion(self,dcm):
        # calculate quaternion from DCM
        q = np.zeros(4, dtype=float)

        #q = [q0,q1,q2,q3] version
        C0 = np.trace(dcm)
        C = [C0,dcm[0,0],dcm[1,1],dcm[2,2]]
        Cj = max(C)
        j = C.index(Cj)
        q[j] = 0.5 * np.sqrt(1+2*Cj-C0)

        if j == 0:
            q[1] =(dcm[1,2] - dcm[2,1]) / (4*q[0])
            q[2] =(dcm[2,0] - dcm[0,2]) / (4*q[0])
            q[3] =(dcm[0,1] - dcm[1,0]) / (4*q[0])
            if q[0] < 0:
                q = -q
        elif j==1:# %ε(1)が最大の場合
            q[0]=(dcm[1,2]-dcm[2,1])/(4*q[1])
            q[3]=(dcm[2,0]+dcm[0,2])/(4*q[1])
            q[2]=(dcm[0,1]+dcm[1,0])/(4*q[1])
            if q[0] < 0:
                q = -q           
        elif j==2: # %ε(2)が最大の場合
            q[0]=(dcm[2,0]-dcm[0,2])/(4*q[2])
            q[3]=(dcm[1,2]+dcm[2,1])/(4*q[2])
            q[1]=(dcm[0,1]+dcm[1,0])/(4*q[2])
            if q[0] < 0:
                q = -q
        elif j==3: # %ε(3)が最大の場合
            q[0]=(dcm[0,1]-dcm[1,0])/(4*q[3])
            q[2]=(dcm[1,2]+dcm[2,1])/(4*q[3])
            q[1]=(dcm[2,0]+dcm[0,2])/(4*q[3])
            if q[0] < 0:
                q = -q
        return q

    # calculate DCM from quaternion
    def quaternion2dcm(self,q):
        dcm = np.zeros((3,3), dtype=float)
        #q = [q0,q1,q2,q3] version
        dcm[0,0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]
        dcm[1,0] = 2 * (q[2]*q[1] - q[3]*q[0])
        dcm[2,0] = 2 * (q[3]*q[1] + q[2]*q[0])
        dcm[0,1] = 2 * (q[1]*q[2] + q[3]*q[0])
        dcm[1,1] = q[2]*q[2] - q[3]*q[3] - q[1]*q[1] + q[0]*q[0]
        dcm[2,1] = 2 * (q[3]*q[2] - q[1]*q[0])
        dcm[0,2] = 2 * (q[1]*q[3] - q[2]*q[0])
        dcm[1,2] = 2 * (q[2]*q[3] + q[1]*q[0])
        dcm[2,2] = - q[2]*q[2] - q[1]*q[1] + q[0]*q[0] + q[3]*q[3]
        return dcm
    
    # differntial calculation of quaternion
    def quaternion_differential(self, omega, quaternion):
        mat = np.array([[0,  -omega[0], -omega[1],  -omega[2]],
                        [omega[0], 0,  omega[2],  -omega[1]],
                        [omega[1], -omega[2], 0,  omega[0]],
                        [omega[2], omega[1], -omega[0], 0]])
        ddt_quaternion = 0.5 * mat @ quaternion
        return ddt_quaternion
    
    def omega_differential(self, omega, inertia_inv, inertia, action):
        ddt_omega =  inertia_inv @ (-np.cross(omega, inertia @ omega) + action)
        return ddt_omega
    
    #-------end toolbox, start actual env-------


    def __init__(self):
        # 初期条件　慣性パラメータ
        self.mass = 10.0
        self.inertia = np.array([[2.683, 0.0, 0.0], \
                                [0.0, 2.683, 0.0], \
                                [0.0, 0.0, 1.897]])
        self.max_multi = 5
        self.multi = np.random.randint(100,self.max_multi*100)/100
        self.est_th = np.diag(self.inertia)
        self.tg_inertia = self.inertia*self.multi
        self.inertia_comb = self.inertia + self.tg_inertia
        self.inertia_comb_inv = np.linalg.inv(self.inertia_comb)
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.g = np.array([0,0,0])  # gravity

        #シミュレーションパラメータ　
        self.dt = 0.1 
        self.simutime =50
        self.omega_count = 0
        
        #報酬パラメータ
        self.q_weight =  1
        self.w_weight = 1
        self.action_weight = 1
        
        # 初期状態 角度(deg)　角速度(rad/s)
        # Rest to Rest
        self.startEuler = np.deg2rad(np.array([0,0,0]))
        #Non-cooperative capture
        # self.startEuler = np.deg2rad(np.array([10,0,0]))
        self.startQuate = self.dcm2quaternion(self.euler2dcm(self.startEuler))
        # self.startOmega = np.array([0,0,0])
        self.startOmega =  np.deg2rad(np.array([5,-5,5]) + np.random.uniform(-1, 1, size=3))

        # 目標値(deg)
        # Rest to Rest
        # self.goalEuler = np.deg2rad(np.array([0,0,0])) + 0.2*np.random.uniform(-np.pi, high=np.pi, size=3)
        # while np.array_equal(self.goalEuler, np.array([0, 0, 0])):
            # self.goalEuler = (0.2*np.random.randint(-np.pi, high=np.pi, size=3))
        # Non-cooperative capture
        self.goalEuler = np.deg2rad(np.array([0,0,0]))

        self.goalQuate = self.dcm2quaternion(self.euler2dcm(self.goalEuler))

        #エラークオータニオンマトリックス
        er1 = self.goalQuate[0]
        er2 = self.goalQuate[1]
        er3 = self.goalQuate[2]
        er4 = self.goalQuate[3]
        self.error_Q = np.array([[er1, er2, er3, er4],
                                [-er2, er1, er4, -er3],
                                [-er3, -er4, er1, er2],
                                [-er4, er3, -er2, er1]])
        
        self.errorQuate = self.error_Q@self.startQuate

        #エラークオータニオンの微分
        self.d_errorQuate = self.quaternion_differential(self.startOmega, self.errorQuate)

        #---thresholds for episode-----------------------------------------------------------------------------------
        self.nsteps = 0  # timestep
        self.max_steps = self.simutime/self.dt

        # Angle, angle speed and speed at which to fail the episode
        self.maxOmega = 5
        self.angle_thre = 0.999962
        self.max_torque = 1

        self.max_action = 1
        self.lower_action = -1
        upper_bound = np.array([self.max_action, self.max_action, self.max_action,self.max_action],dtype=np.float32)
        # upper_bound = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max],dtype=np.float32)
        lower_bound = np.array([self.lower_action, self.lower_action, self.lower_action,self.lower_action],dtype=np.float32)
        #------------------------------------------------------------------------------------------------------------

        # 状態量（姿勢角４・姿勢角微分４・角速度３・推定慣性モーメントの対角成分 ３）
        high = np.ones(14,dtype = np.float32)*np.finfo(np.float32).max

        self.action_space = spaces.Box(lower_bound, upper_bound)
        self.observation_space = spaces.Box(-high, high)
        self.pre_state = np.hstack((self.startQuate,self.startOmega))
        self.state = np.hstack((self.errorQuate,self.d_errorQuate, self.startOmega, self.est_th))
        # self.pre_state = [self.startQuate,self.startOmega]
        # self.state = [self.errorQuate,self.d_errorQuate, self.startOmega]

        self.seed()
        self.viewer = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        
        pre_state = self.pre_state
        # q, omega = pre_state
        q = pre_state[:4]
        omega = pre_state[-3:]
        
        state = self.state
        qe = state[:4]

        #ステートアップデート（オイラー法）
        # q_dot = self.quaternion_differential(omega, q)
        # omega_dot = self.omega_differential(omega,self.inertia_inv,action)
        # q = q + q_dot * self.dt
        # omega = omega + omself.steps_beyond_done = Noneferential(omega,self.inertia_comb_inv,self.inertia_comb,action)
        k1 = self.omega_differential(omega,self.inertia_comb_inv,self.inertia_comb,action)
        k2 = self.omega_differential(omega + 0.5*self.dt*k1, self.inertia_comb_inv,self.inertia_comb,action)
        k3 = self.omega_differential(omega + 0.5*self.dt*k2, self.inertia_comb_inv,self.inertia_comb,action)
        k4 = self.omega_differential(omega + self.dt*k3, self.inertia_comb_inv,self.inertia_comb,action)

        l1 = self.quaternion_differential(omega, q) 
        l2 = self.quaternion_differential(omega + 0.5*self.dt*k1, q + 0.5*self.dt*l1)
        l3 = self.quaternion_differential(omega + 0.5*self.dt*k2, q + 0.5*self.dt*l2)
        l4 = self.quaternion_differential(omega + self.dt*k3, q + self.dt*l3)

        q_new = q + 1/6 * (l1 + 2*l2 + 2*l3 + l4) * self.dt
        omega_new = omega + 1/6 * (k1 + 2*k2 + 2*k3 + k4) * self.dt

        qe_new = self.error_Q @ q_new
        qe_dot_new = self.quaternion_differential(omega_new, qe_new)

        self.pre_state = np.hstack((q_new, omega_new))
        self.state = np.hstack((qe_new, qe_dot_new, omega_new, self.est_th))

        # とりまdoneはfalseにしておく
        # done = False

        #ステップ数を更新
        self.nsteps += 1

        # 終了判定　角速度がマックス値を超える or 入力制約超過　or 最大ステップ数に達したら 
        done_1 = abs(omega[0]) > self.maxOmega \
                or abs(omega[1]) > self.maxOmega \
                or abs(omega[2]) > self.maxOmega \
                or abs(action[0]) > self.max_torque \
                or abs(action[1]) > self.max_torque \
                or abs(action[2]) > self.max_torque 
        done_2 = self.nsteps >= self.max_steps
        done_3 = False
        if omega@omega < 1e-5:
            self.omega_count += 1
        if self.omega_count > 5:
            done_3 = True
            self.omega_count = 0
        done = bool(done_1 or done_2 or done_3)

        # 報酬関数
        #--------REWARD---------
        if not done:
            # reward += -0.01
            #状態と入力を抑えたい
            # reward = -(self.q_weight*((1-qe_new[0])**2 + qe[1:]@qe[1:]) + self.w_weight*omega_new@omega_new + self.action_weight*action@action) 
            if qe_new[0] >= self.angle_thre:
                reward = np.array([1,-1,-1,-1])@np.power(qe,2)
            else:
                if qe_new[0] > qe[0]:
                    reward = 0.1
                else:
                    reward = -0.1
        
        elif self.steps_beyond_done is None:
            # epsiode just ended
            self.steps_beyond_done = 0
            if bool(done_1):
                reward = -30
            else:
                if qe_new[0] >= self.angle_thre:
                    reward = 30
                else:
                    reward = 0
        #------------------------

        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, self.pre_state, {}

    def reset(self):
        # 初期条件　慣性パラメータ
        self.mass = 10.0
        self.inertia = np.array([[2.683, 0.0, 0.0], \
                                [0.0, 2.683, 0.0], \
                                [0.0, 0.0, 1.897]])
        self.multi = np.random.randint(100,self.max_multi*100)/100
        self.tg_inertia = self.inertia*self.multi
        self.est_th = np.diag(self.inertia)
        self.inertia_comb = self.inertia + self.tg_inertia
        self.inertia_comb_inv = np.linalg.inv(self.inertia_comb)
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        # 初期状態 角度(deg)　角速度(rad/s)
        self.startEuler = np.deg2rad(np.array([0,0,0]))
        # self.startEuler = np.deg2rad(np.array([10,0,0]))
        self.startQuate = self.dcm2quaternion(self.euler2dcm(self.startEuler))
        # self.startOmega = np.array([0,0,0])
        coef = 2*np.random.randint(0,2,size=3)-1
        self.startOmega = coef* np.deg2rad(np.array([5,-5,5]))#+ np.random.uniform(-1, 1, size=3))

        # 目標値(deg)
        # self.goalEuler = np.deg2rad(np.array([0,0,0])) + 0.2*np.random.uniform(-np.pi, high=np.pi, size=3)
        # while np.array_equal(self.goalEuler, np.array([0, 0, 0])):
            # self.goalEuler = (0.2*np.random.randint(-np.pi, high=np.pi, size=3))
        self.goalEuler = np.deg2rad(np.array([0,0,0]))
        self.goalQuate = self.dcm2quaternion(self.euler2dcm(self.goalEuler))

        #エラークオータニオンマトリックス
        er1 = self.goalQuate[0]
        er2 = self.goalQuate[1]
        er3 = self.goalQuate[2]
        er4 = self.goalQuate[3]
        self.error_Q = np.array([[er1, er2, er3, er4],
                                [-er2, er1, er4, -er3],
                                [-er3, -er4, er1, er2],
                                [-er4, er3, -er2, er1]])
        
        self.errorQuate = self.error_Q@self.startQuate

        #エラークオータニオンの微分
        self.d_errorQuate = self.quaternion_differential(self.startOmega, self.errorQuate)
        self.pre_state = np.hstack((self.startQuate,self.startOmega))
        self.state = np.hstack((self.errorQuate,self.d_errorQuate, self.startOmega,self.est_th))

        obs = self.state
        # タイムスタンプをリセット
        self.nsteps = 0  
        self.steps_beyond_done = None
        return obs

    def reset_capture(self):
        # 初期条件　慣性パラメータ
        self.inertia = np.array([[2.683, 0.0, 0.0], \
                                [0.0, 2.683, 0.0], \
                                [0.0, 0.0, 1.897]])
        self.multi = np.random.uniform(1, high=5)
        self.tg_inertia = self.inertia*self.multi
        self.inertia_comb = self.inertia + self.tg_inertia
        self.inertia_comb_inv = np.linalg.inv(self.inertia_comb)
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        # 初期状態 角度(deg)　角速度(rad/s)
        self.startEuler = np.deg2rad(np.array([10,0,0]))
        self.startQuate = self.dcm2quaternion(self.euler2dcm(self.startEuler))
        self.startOmega = np.array([0,0,0])
        
        # 目標値(deg)
        self.goalEuler = np.deg2rad(np.array([0,0,0]))
        self.goalQuate = self.dcm2quaternion(self.euler2dcm(self.goalEuler))

        #エラークオータニオンマトリックス
        er1 = self.goalQuate[0]
        er2 = self.goalQuate[1]
        er3 = self.goalQuate[2]
        er4 = self.goalQuate[3]
        self.error_Q = np.array([[er1, er2, er3, er4],
                                [-er2, er1, er4, -er3],
                                [-er3, -er4, er1, er2],
                                [-er4, er3, -er2, er1]])
        
        self.errorQuate = self.error_Q@self.startQuate

        #エラークオータニオンの微分
        self.d_errorQuate = self.quaternion_differential(self.startOmega, self.errorQuate)
        self.pre_state = np.hstack((self.startQuate,self.startOmega))
        self.state = np.hstack((self.errorQuate,self.d_errorQuate, self.startOmega))

        obs = self.state
        # タイムスタンプをリセット
        self.nsteps = 0  
        return obs

    def render(self, mode='human'):
        #do nothing
        print('rendering currently not supported.')

    def close(self):
        print('rendering not supported currently.')
