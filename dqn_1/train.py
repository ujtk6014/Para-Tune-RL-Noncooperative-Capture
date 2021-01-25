import os

import math
import matplotlib.pyplot as plt
import torch
from gym import wrappers 
import numpy as np
from tqdm import tqdm
import datetime

from network import DDQNAgent
from utils import *


def train(batch_size=128, critic_lr=1e-3, actor_lr=1e-4, max_episodes=10000, max_steps=300, gamma=0.99, tau=1e-3,
          buffer_maxlen=100000):
    # simulation of the agent solving the spacecraft attitude control problem
    env = make("SatelliteContinuous")

    max_episodes = max_episodes
    max_steps = max_steps
    batch_size = batch_size

    gamma = gamma
    tau = tau
    buffer_maxlen = buffer_maxlen
    critic_lr = critic_lr
    actor_lr = actor_lr
    learning_rate = 1e-3

    agent = DDQNAgent(env, gamma, tau, buffer_maxlen, learning_rate, True, max_episodes * max_steps)
    #学習済みモデルを使うとき
    # curr_dir = os.path.abspath(os.getcwd())
    # agent = torch.load(curr_dir + "/models/spacecraft_control_ddqn.pkl")
    episode_rewards = mini_batch_train_adaptive(env, agent, max_episodes, max_steps, batch_size)

    #-------------------plot settings------------------------------
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 10 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 10 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 10 # 軸だけ変更されます 
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams['axes.grid'] = True # make grid
    #--------------------------------------------------------------  
    plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

    date = datetime.datetime.now()
    date = '{0:%Y%m%d}'.format(date)
    curr_dir = os.path.abspath(os.getcwd())
    plt.savefig(curr_dir + "/results/reward/dqn/plot_reward_"+ date + ".png")
    if not os.path.isdir("models"):
        os.mkdir("models")
    torch.save(agent, curr_dir + "/models/spacecraft_control_ddqn.pkl")


def evaluate():
    # simulation of the agent solving the cartpole swing-up problem
    env = make("SatelliteContinuous")
    # uncomment for recording a video of simulation
    # env = wrappers.Monitor(env, './video', force=True)

    curr_dir = os.path.abspath(os.getcwd())

    agent = torch.load(curr_dir + "/models/spacecraft_control_ddqn.pkl")
    agent.train = False

    state = env.reset()
    print("The goal angle is :" + str(np.rad2deg(env.goalEuler)) +"\n")
    total_r = 0
    qe = np.empty((0,4))
    q = np.empty((0,4))
    w = np.empty((0,3))
    r_hist = np.empty((0,3))
    k_hist = np.empty((0,1))
    D_hist = np.empty((0,3))
    actions = np.empty((0,1))
    inputs = np.empty((0,3))
    
    alpha = 0.5    
    k_delta = 0.01
    D_delta = 1e-5
    alpha = 0.5
    k = 0.8
    D = np.diag([4e-4,1,1,1,5.8e-4,1,1,1,5.2e-4])
    dt = env.dt
    simutime = 30
    max_steps = int(simutime/dt)  # dt is 0.01

    th_e = np.array(env.inertia.flatten())
    with tqdm(range(max_steps),leave=False) as pbar:
        for step, ch in enumerate(pbar):
            if step % 5 == 0:
                action = agent.get_action(state)
                n= str(Base_10_to_n(action,3))
                while len(n) < 4:
                    n ="0"+n
                para = np.empty((4,1))
                for i in range(4):
                    if n[i] == '0':
                        para[i] = 1
                    elif n[i] == '1':
                        para[i] = 0
                    elif n[i] == '2':
                        para[i] = -1

            k += k_delta * para[0]
            D[0,0] += D_delta * para[1]
            D[4,4] += D_delta * para[2]
            D[8,8] += D_delta * para[3]
            if k<0 or D[0,0]<0 or D[4,4]<0 or D[8,8] <0:
                env.neg_param_flag = True
            para = np.empty((4,1))
            W = state[8:11]
            x1 = state[1:4]
            x2 = alpha*x1 + W
            dqe = state[4:8]
            Y = np.array([[alpha*dqe[1], alpha*dqe[2], alpha*dqe[3], W[0]*W[2], W[1]*W[2], W[2]*W[2], -W[0]*W[2], -W[1]*W[1], -W[1]*W[2]],
                [-W[0]*W[2], -W[1]*W[2], -W[2]*W[2], alpha*dqe[1], alpha*dqe[2], alpha*dqe[3], W[0]*W[0], W[0]*W[1], W[0]*W[2]],
                [W[0]*W[1], W[1]*W[1], W[1]*W[2], -W[0]*W[0], -W[0]*W[1], -W[0]*W[2], alpha*dqe[1], alpha*dqe[2], alpha*dqe[3]]])
            input = -0.5*x1 -Y@th_e - k*x2

            dth = np.linalg.inv(D) @ Y.T @ x2
            th_e += env.dt*dth
            # action = np.squeeze(action)
            next_error_state, reward, done, next_state, _ = env.step(input)
            # if step == 20/dt:
            #     env.inertia = env.inertia_comb
            #     env.inertia_inv = np.linalg.inv(env.inertia)
            #     env.pre_state[-3:] += np.deg2rad([5,-5,5])
            #     env.state[-3:] += np.deg2rad([5,-5,5])
            # env.render()        
            q=np.append(q,next_state[:4].reshape(1,-1),axis=0)
            qe=np.append(qe,next_error_state[:4].reshape(1,-1),axis=0)
            w=np.append(w,next_error_state[8:11].reshape(1,-1),axis=0)
            total_r += reward
            d_tmp = np.array([D[0,0],D[4,4],D[8,8]])
            k_hist = np.append(k_hist, k.reshape(1,-1),axis=0)
            actions = np.append(actions, action.reshape(1,-1),axis=0)
            r_hist = np.append(r_hist, np.array([env.r1,env.r2,env.r3]).reshape(1,-1),axis=0)
            inputs = np.append(inputs, input.reshape(1,-1),axis=0)
            D_hist = np.append(D_hist, d_tmp.reshape(1,-1),axis=0)
            state = next_error_state

    env.close()
    #-------------------------------結果のプロット----------------------------------
    #show the total reward
    print("Total Reward is : " + str(total_r))
    # データの形の整理
    q = q.reshape([-1,4])
    qe = qe.reshape([-1,4])
    w = w.reshape([-1,3])
    # angle = [e for i in]

    # plot the angle and action curve
    #-------------------plot settings------------------------------
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 10 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 10 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 10 # 軸だけ変更されます 
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams['axes.grid'] = True # make grid
    tate = 2.0
    yoko = 4.0
    #------------------------------------------------
    
    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps)*dt, q[:,0],label =r"$q_{0}$")
    plt.plot(np.arange(max_steps)*dt, q[:,1],label =r"$q_{1}$")
    plt.plot(np.arange(max_steps)*dt, q[:,2],label =r"$q_{2}$")
    plt.plot(np.arange(max_steps)*dt, q[:,3],label =r"$q_{3}$")
    # plt.title('Quaternion')
    plt.ylabel('quaternion value')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=4)
    plt.tight_layout()
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.grid(True)
    plt.savefig(curr_dir + "/results/dqn_eval/plot_quat.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps)*dt, qe[:,0],label =r"$q_{e0}$")
    plt.plot(np.arange(max_steps)*dt, qe[:,1],label =r"$q_{e1}$")
    plt.plot(np.arange(max_steps)*dt, qe[:,2],label =r"$q_{e2}$")
    plt.plot(np.arange(max_steps)*dt, qe[:,3],label =r"$q_{e3}$")
    # plt.title('Quaternion Error')
    plt.ylabel('quaternion error value')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=4)
    plt.tight_layout()
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/dqn_eval/plot_quate_error.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps)*dt, w[:,0],label =r"$\omega_{x}$")
    plt.plot(np.arange(max_steps)*dt, w[:,1],label =r"$\omega_{y}$")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/dqn_eval/plot_omega.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps)*dt, inputs[:,0],label = r"$\tau_{x}$")
    plt.plot(np.arange(max_steps)*dt, inputs[:,1],label = r"$\tau_{y}$")
    plt.plot(np.arange(max_steps)*dt, inputs[:,2],label = r"$\tau_{z}$")
    # plt.title('Action')
    plt.ylabel('Input torque [Nm]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    # plt.ylim(-0.3, 0.25)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/dqn_eval/plot_torque.png")

    angle = np.array([np.rad2deg(env.dcm2euler(env.quaternion2dcm(q[i,:]))).tolist() for i in range(max_steps)])
    angle = angle.reshape([-1,3])
    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps)*dt, angle[:,0],label = r"$\phi$")
    plt.plot(np.arange(max_steps)*dt, angle[:,1],label = r"$\theta$")
    plt.plot(np.arange(max_steps)*dt, angle[:,2],label = r"$\psi$")
    # plt.title('Action')
    plt.ylabel('angle [deg]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    # plt.title('Action')

    plt.plot(np.arange(max_steps)*dt, k_hist[:,0],label = r"$\gain$")
    plt.ylabel('k gain')
    plt.xlabel(r'time [s]')
    plt.tight_layout()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/dqn_eval/plot_k.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps)*dt, D_hist[:,0],label = r"$D_{1}$")
    plt.plot(np.arange(max_steps)*dt, D_hist[:,1],label = r"$D_{2}$")
    plt.plot(np.arange(max_steps)*dt, D_hist[:,2],label = r"$D_{3}$")
    # plt.title('Action')
    plt.ylabel('d [deg]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/dqn_eval/plot_d.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps)*dt, actions[:,0],label = r"$actions$")
    # plt.title('Action')
    plt.ylabel('Selected action')
    plt.xlabel(r'time [s]')
    plt.tight_layout()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/dqn_eval/plot_action.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,0],label = r"$q$ pnlty")
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,1],label = r"$\omega$ pnlty")
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,2],label = r"$\tau$ pnlty")
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,0]+r_hist[:,1]+r_hist[:,2],label = r"$toal$",linestyle='dotted')
    # plt.title('Action')
    plt.ylabel('reward')
    plt.xlabel(r'time [s]')
    plt.tight_layout()
    plt.legend()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/dqn_eval/plot_reward.png")
    

    plt.show()
    # -------------------------結果プロット終わり--------------------------------
def env_pd():

    # simulation of the agent solving the cartpole swing-up problem
    env = make("SatelliteContinuous")
    # uncomment for recording a video of simulation
    # env = wrappers.Monitor(env, './video', force=True)

    curr_dir = os.path.abspath(os.getcwd())
    env.reset()
    print("The goal angle is :" + str(np.rad2deg(env.goalEuler)))
    r = 0
    qe = np.empty((0,4))
    q = np.empty((0,4))
    w = np.empty((0,3))
    actions = np.empty((0,3))

    kp = 0.7
    kd = 1.9
    Kp = np.array([[0,kp,0,0],
                  [0,0,kp,0],
                  [0,0,0,kp]])
    Kd = np.array([[kd,0,0],
                  [0,kd,0],
                  [0,0,kd]])
    action = np.array([0,0,0]).reshape(1,3)
    actions = np.append(actions, action,axis=0)

    dt = 0.01
    max_steps = int(50/0.01) -1 # dt is 0.01

    for i in range(1, max_steps):
        action = np.squeeze(action)
        next_error_state, reward, done, next_state, _ = env.step(action)
        # env.render()
        # q=np.append(q,next_state[0].reshape(1,-1),axis=0)
        # qe=np.append(qe,next_error_state[0].reshape(1,-1),axis=0)
        # w=np.append(w,next_error_state[2].reshape(1,-1),axis=0)
        q=np.append(q,next_state[:4].reshape(1,-1),axis=0)
        qe=np.append(qe,next_error_state[:4].reshape(1,-1),axis=0)
        w=np.append(w,next_error_state[-3:].reshape(1,-1),axis=0)
        r += reward
        # state = next_state
        #----------------control law (PID controller)-----------------------
        action = -Kp@next_error_state[:4].reshape(-1,1)-Kd@next_error_state[-3:].reshape(-1,1)
        actions = np.append(actions, action.reshape(1,-1),axis=0)
        #--------------------------------------------------------------------

    # env.close()
    #show the total reward
    print("Total Reward is : " + str(r))
    # データの形の整理
    q = q.reshape([-1,4])
    qe = qe.reshape([-1,4])
    w = w.reshape([-1,3])
    # angle = [e for i in]

    # plot the angle and action curve
    curr_dir = os.path.abspath(os.getcwd())
    if not os.path.isdir("results"):
        os.mkdir("results")
    # plot the angle and action curve
    #-------------------plot settings------------------------------
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 15 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 15 # 軸だけ変更されます 
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams['axes.grid'] = True # make grid
    tate = 4.0
    yoko = 6.0
    #------------------------------------------------
    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps-1)*dt, q[:,0],label =r"$q_{0}$")
    plt.plot(np.arange(max_steps-1)*dt, q[:,1],label =r"$q_{1}$")
    plt.plot(np.arange(max_steps-1)*dt, q[:,2],label =r"$q_{2}$")
    plt.plot(np.arange(max_steps-1)*dt, q[:,3],label =r"$q_{3}$")
    # plt.title('Quaternion')
    plt.ylabel('quaternion value')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=4)
    plt.tight_layout()
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.grid(True)
    plt.savefig(curr_dir + "/results/pd_test/plot_quat.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps-1)*dt, qe[:,0],label =r"$q_{e0}$")
    plt.plot(np.arange(max_steps-1)*dt, qe[:,1],label =r"$q_{e1}$")
    plt.plot(np.arange(max_steps-1)*dt, qe[:,2],label =r"$q_{e2}$")
    plt.plot(np.arange(max_steps-1)*dt, qe[:,3],label =r"$q_{e3}$")
    # plt.title('Quaternion Error')
    plt.ylabel('quaternion error value')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=4)
    plt.tight_layout()
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/pd_test/plot_quate_error.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps-1)*dt, w[:,0],label =r"$\omega_{x}$")
    plt.plot(np.arange(max_steps-1)*dt, w[:,1],label =r"$\omega_{y}$")
    plt.plot(np.arange(max_steps-1)*dt, w[:,2],label =r"$\omega_{z}$")
    plt.savefig(curr_dir + "/results/pd_test/plot_omega.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps)*dt, actions[:,0],label = r"$\tau_{x}$")
    plt.plot(np.arange(max_steps)*dt, actions[:,1],label = r"$\tau_{y}$")
    plt.plot(np.arange(max_steps)*dt, actions[:,2],label = r"$\tau_{z}$")
    # plt.title('Action')
    plt.ylabel('Input torque [Nm]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    # plt.ylim(-0.3, 0.25)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/pd_test/plot_torque.png")

    angle = np.array([np.rad2deg(env.dcm2euler(env.quaternion2dcm(q[i,:]))).tolist() for i in range(max_steps-1)])
    angle = angle.reshape([-1,3])
    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps-1)*dt, angle[:,0],label = r"$\phi$")
    plt.plot(np.arange(max_steps-1)*dt, angle[:,1],label = r"$\theta$")
    plt.plot(np.arange(max_steps-1)*dt, angle[:,2],label = r"$\psi$")
    # plt.title('Action')
    plt.ylabel('angle [deg]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/pd_test/plot_angle.png")
    plt.show()    

def env_adaptive():

    # simulation of the agent solving the cartpole swing-up problem
    env = make("SatelliteContinuous")
    # uncomment for recording a video of simulation
    # env = wrappers.Monitor(env, './video', force=True)

    curr_dir = os.path.abspath(os.getcwd())
    env.reset()
    total_r = 0
    qe = np.empty((0,4))
    q = np.empty((0,4))
    w = np.empty((0,3))
    actions = np.empty((0,3))
    r_hist = np.empty((0,3))

    #----------------------control parameters----------------------------
    alpha = 0.5
    k = 0.8
    zeta = 0.2
    D = np.diag([4e-4,1,1,1,5.8e-4,1,1,1,5.2e-4])
    th = env.inertia_comb.flatten()
    th_e = np.array(env.inertia.flatten())
    #--------------------------------------------------------------------
    action = np.array([0,0,0]).reshape(1,3)
    actions = np.append(actions, action,axis=0)

    dt = env.dt
    max_steps = int(30/dt) -1 # dt is 0.01

    for i in range(1, max_steps):
        action = np.squeeze(action)

        # if i == 20/dt:
        #     env.inertia = env.inertia_comb
        #     env.inertia_inv = np.linalg.inv(env.inertia)
        #     env.pre_state[-3:] += np.deg2rad([5,-5,5])
        #     env.state[-3:] += np.deg2rad([5,-5,5])
        
        next_error_state, reward, done, next_state, _ = env.step(action)
        q=np.append(q,next_state[:4].reshape(1,-1),axis=0)
        qe=np.append(qe,next_error_state[:4].reshape(1,-1),axis=0)
        w=np.append(w,next_error_state[8:11].reshape(1,-1),axis=0)
        r_hist = np.append(r_hist, np.array([env.r1,env.r2,env.r3]).reshape(1,-1),axis=0)
        total_r += reward
    #----------------control law (Adaptive controller)-----------------------
        W = next_error_state[8:11]
        x1 = next_error_state[1:4]
        x2 = alpha*x1 + W
        dqe = next_error_state[4:8]
        Y = np.array([[alpha*dqe[1], alpha*dqe[2], alpha*dqe[3], W[0]*W[2], W[1]*W[2], W[2]*W[2], -W[0]*W[2], -W[1]*W[1], -W[1]*W[2]],
             [-W[0]*W[2], -W[1]*W[2], -W[2]*W[2], alpha*dqe[1], alpha*dqe[2], alpha*dqe[3], W[0]*W[0], W[0]*W[1], W[0]*W[2]],
              [W[0]*W[1], W[1]*W[1], W[1]*W[2], -W[0]*W[0], -W[0]*W[1], -W[0]*W[2], alpha*dqe[1], alpha*dqe[2], alpha*dqe[3]]])
        action = -0.5*x1 -Y@th_e - k*x2
        actions = np.append(actions, action.reshape(1,-1),axis=0)

        dth = np.linalg.inv(D) @ Y.T @ x2
        th_e += dt*dth
    #---------------------------------------------------------------------

    # env.close()
    #show the total reward
    print("Total Reward is : " + str(total_r))
    # データの形の整理
    q = q.reshape([-1,4])
    qe = qe.reshape([-1,4])
    w = w.reshape([-1,3])
    # angle = [e for i in]

    # plot the angle and action curve
    curr_dir = os.path.abspath(os.getcwd())
    if not os.path.isdir("results"):
        os.mkdir("results")

    # plot the angle and action curve
    #-------------------plot settings------------------------------
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 10 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 10 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 10 # 軸だけ変更されます 
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams['axes.grid'] = True # make grid
    tate = 2.0
    yoko = 4.0
    #------------------------------------------------
    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps-1)*dt, q[:,0],label =r"$q_{0}$")
    plt.plot(np.arange(max_steps-1)*dt, q[:,1],label =r"$q_{1}$")
    plt.plot(np.arange(max_steps-1)*dt, q[:,2],label =r"$q_{2}$")
    plt.plot(np.arange(max_steps-1)*dt, q[:,3],label =r"$q_{3}$")
    # plt.title('Quaternion')
    plt.ylabel('quaternion value')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=4)
    plt.tight_layout()
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.grid(True)
    plt.savefig(curr_dir + "/results/adap_test/plot_quat.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps-1)*dt, qe[:,0],label =r"$q_{e0}$")
    plt.plot(np.arange(max_steps-1)*dt, qe[:,1],label =r"$q_{e1}$")
    plt.plot(np.arange(max_steps-1)*dt, qe[:,2],label =r"$q_{e2}$")
    plt.plot(np.arange(max_steps-1)*dt, qe[:,3],label =r"$q_{e3}$")
    # plt.title('Quaternion Error')
    plt.ylabel('quaternion error value')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=4)
    plt.tight_layout()
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/adap_test/plot_quate_error.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps-1)*dt, w[:,0],label =r"$\omega_{x}$")
    plt.plot(np.arange(max_steps-1)*dt, w[:,1],label =r"$\omega_{y}$")
    plt.plot(np.arange(max_steps-1)*dt, w[:,2],label =r"$\omega_{z}$")
    # plt.title('Angular velocity')
    plt.ylabel('angular velocity [rad/s]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/adap_test/plot_omega.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps)*dt, actions[:,0],label = r"$\tau_{x}$")
    plt.plot(np.arange(max_steps)*dt, actions[:,1],label = r"$\tau_{y}$")
    plt.plot(np.arange(max_steps)*dt, actions[:,2],label = r"$\tau_{z}$")
    # plt.title('Action')
    plt.ylabel('Input torque [Nm]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    # plt.ylim(-0.3, 0.25)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/adap_test/plot_torque.png")

    angle = np.array([np.rad2deg(env.dcm2euler(env.quaternion2dcm(q[i,:]))).tolist() for i in range(max_steps-1)])
    angle = angle.reshape([-1,3])
    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps-1)*dt, angle[:,0],label = r"$\phi$")
    plt.plot(np.arange(max_steps-1)*dt, angle[:,1],label = r"$\theta$")
    plt.plot(np.arange(max_steps-1)*dt, angle[:,2],label = r"$\psi$")
    # plt.title('Action')
    plt.ylabel('angle [deg]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/adap_test/plot_angle.png")
    
    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,0],label = r"$q$ pnlty")
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,1],label = r"$\omega$ pnlty")
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,2],label = r"$\tau$ pnlty")
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,0]+r_hist[:,1]+r_hist[:,2],label = r"$toal$",linestyle='dotted')
    # plt.title('Action')
    plt.ylabel('reward')
    plt.xlabel(r'time [s]')
    plt.tight_layout()
    plt.legend()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/dqn_eval/plot_reward.png")
    plt.show()    

if __name__ == '__main__':
    plt.close()
    val = input('Enter the number 1:train 2:evaluate 3:env_pd  4:env_adaptive  > ')
    if val == '1':
        train()
    elif val == '2':
        evaluate()
    elif val == '3':
        env_pd()
    elif val == '4':
        env_adaptive()
    else:
        print("You entered the wrong number, run again and choose from 1 or 2 or 3.")
