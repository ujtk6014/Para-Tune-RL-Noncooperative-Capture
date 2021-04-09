import os

import math
import matplotlib.pyplot as plt
import torch
from gym import wrappers 
import numpy as np
import wandb


from network import TD3Agent
from utils import *

def train():
    # simulation of the agent solving the spacecraft attitude control problem
    env = make("SatelliteContinuous")
    #logger
    wandb.init(project='Para-Tune-RL-TD3',
        config={
        "batch_size": 128,
        "critic_lr": 1e-3,
        "actor_lr": 1e-4,
        "max_episodes": 3000,
        "max_steps": 500,
        "gamma": 0.99,
        "tau" : 1e-3,
        "buffer_maxlen": 100000,
        "policy_noise": 0.2,
        "policy_freq": 2,
        "noise_clip": 0.5,
        "prioritized_on": False,
        "State": 'angle:4, ang_rate:4, ang_vel:3, th_e:9',}
    )
    config = wandb.config

    max_episodes = config.max_episodes
    max_steps = config.max_steps
    batch_size = config.batch_size

    policy_noise = config.policy_noise
    policy_freq = config.policy_freq
    noise_clip = config.noise_clip

    gamma = config.gamma
    buffer_maxlen = config.buffer_maxlen
    tau = config.tau
    critic_lr = config.critic_lr
    actor_lr = config.actor_lr

    agent = TD3Agent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr, True, max_episodes * max_steps,
                    policy_freq, policy_noise, noise_clip)
    # wandb.watch([agent.critic,agent.actor], log="all")
    #学習済みモデルを使うとき
    # curr_dir = os.path.abspath(os.getcwd())
    # agent = torch.load(curr_dir + "/models/spacecraft_control_td3_home.pkl")

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

    curr_dir = os.path.abspath(os.getcwd())
    plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    # plt.show()
    # plt.savefig(curr_dir + "/results/td3_eval/plot_training_reward.png")

    if not os.path.isdir("models"):
        os.mkdir("models")
    torch.save(agent, curr_dir + "/models/spacecraft_control_td3_home.pkl")

def evaluate():
    # simulation of the agent solving the cartpole swing-up problem
    env = make("SatelliteContinuous")
    # uncomment for recording a video of simulation
    # env = wrappers.Monitor(env, './video', force=True)

    curr_dir = os.path.abspath(os.getcwd())

    # agent = torch.load(curr_dir + "/models/spacecraft_control_td3_home.pkl")
    agent = torch.load(curr_dir + "/models/spacecraft_control_td3_home.pkl",map_location='cpu')
    agent.device = torch.device('cpu')
    agent.train = False

    state = env.reset()

    print("The target multi :" + str(env.multi) +"\n")
    r = 0
    R = np.empty((0,1))
    qe = np.empty((0,4))
    q = np.empty((0,4))
    w = np.empty((0,3))
    actions = np.empty((0,3))
    r_hist = np.empty((0,4))
    k_hist = []
    alpha_hist = []
    d_hist = np.empty((0,9))

    dt = 0.1
    simutime = 50
    # env.simutime = simutime
        
    max_steps = int(simutime/dt) -1 # dt is 0.1
    alpha = 0.5
    k_max = 7
    alpha_max = 1

    th_e = np.array(env.inertia.flatten()*env.multi)

    with tqdm(range(max_steps),leave=False) as pbar:
        for i, ch in enumerate(pbar):
            action = agent.get_action(state)
            para_candi = (action + 1)/2
            #----------------control law (Adaptive controller)-----------------------
            k = para_candi[0]*k_max
            alpha = para_candi[1]*alpha_max
            d_tmp = [para_candi[i+2]*2500 +500 for i in range(len(para_candi)-2)]
            D = np.diag([1/d_tmp[0],1/d_tmp[1],1/d_tmp[2],1/d_tmp[3],1/d_tmp[4],1/d_tmp[5],1/d_tmp[6],1/d_tmp[7],1/d_tmp[8]])

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
            # env.est_th = ([th_e[0],th_e[4],th_e[8]]/np.diag(env.inertia) -1)/(env.max_multi-1)
            env.est_th = th_e.flatten()/25
            next_error_state, reward, done, next_state, _ = env.step(input)
            # if i == 20/dt:
            #     env.inertia = env.inertia_comb
            #     env.inertia_inv = np.linalg.inv(env.inertia)
            #     env.pre_state[-3:] += np.deg2rad([5,-5,5])
            #     env.state[-3:] += np.deg2rad([5,-5,5])
            # env.render()        
            q=np.append(q,next_state[:4].reshape(1,-1),axis=0)
            qe=np.append(qe,next_error_state[:4].reshape(1,-1),axis=0)
            w=np.append(w,next_error_state[8:11].reshape(1,-1),axis=0)
            r += reward
            R = np.append(R,reward)
            actions = np.append(actions, input.reshape(1,-1),axis=0)
            k_hist.append(k)
            alpha_hist.append(alpha)
            r_hist = np.append(r_hist, np.array([-env.r1,-env.r2,-env.r3,-env.r4]).reshape(1,-1),axis=0)
            d_hist = np.append(d_hist, np.array(d_tmp).reshape(1,-1),axis=0)
            state = next_error_state

    env.close()
    #-------------------------------結果のプロット----------------------------------#
    #region
    #show the total reward
    print("Total Reward is : " + str(r))
    # データの形の整理
    q = q.reshape([-1,4])
    qe = qe.reshape([-1,4])
    w = w.reshape([-1,3])
    # angle = [e for i in]
 
    # plot the angle and action curve
    #-------------------plot settings------------------------------
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 8 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 8 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 8 # 軸だけ変更されます 
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams['axes.grid'] = True # make grid
    plt.rcParams["legend.loc"] = "best"         # 凡例の位置、"best"でいい感じのところ
    plt.rcParams["legend.frameon"] = True       # 凡例を囲うかどうか、Trueで囲う、Falseで囲わない
    plt.rcParams["legend.framealpha"] = 1.0     # 透過度、0.0から1.0の値を入れる
    plt.rcParams["legend.facecolor"] = "white"  # 背景色
    # plt.rcParams["legend.edgecolor"] = "black"  # 囲いの色
    plt.rcParams["legend.fancybox"] = True     # Trueにすると囲いの四隅が丸くなる
    tate = 4.0
    yoko = 8.0
    #------------------------------------------------

    # plt.figure(figsize=(yoko,tate),dpi=100)
    # plt.plot(np.arange(max_steps)*dt, q[:,0],label =r"$q_{0}$")
    # plt.plot(np.arange(max_steps)*dt, q[:,1],label =r"$q_{1}$")
    # plt.plot(np.arange(max_steps)*dt, q[:,2],label =r"$q_{2}$")
    # plt.plot(np.arange(max_steps)*dt, q[:,3],label =r"$q_{3}$")
    # # plt.title('Quaternion')
    # plt.ylabel('quaternion value')
    # plt.xlabel(r'time [s]')
    # plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=4)
    # plt.tight_layout()
    # plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # # plt.grid(True)
    # plt.savefig(curr_dir + "/results/td3_eval/plot_quat.png")

    # plt.figure(figsize=(yoko,tate),dpi=100)
    # plt.plot(np.arange(max_steps)*dt, qe[:,0],label =r"$q_{e0}$")
    # plt.plot(np.arange(max_steps)*dt, qe[:,1],label =r"$q_{e1}$")
    # plt.plot(np.arange(max_steps)*dt, qe[:,2],label =r"$q_{e2}$")
    # plt.plot(np.arange(max_steps)*dt, qe[:,3],label =r"$q_{e3}$")
    # # plt.title('Quaternion Error')
    # plt.ylabel('quaternion error value')
    # plt.xlabel(r'time [s]')
    # plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=4)
    # plt.tight_layout()
    # plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.savefig(curr_dir + "/results/td3_eval/plot_quate_error.png")

    plt.figure(figsize=(12,5),dpi=100)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    # plt.figure(figsize=(yoko,tate),dpi=100)
    plt.subplot(231)
    plt.plot(np.arange(max_steps)*dt, w[:,0],label =r"$\omega_{x}$")
    plt.plot(np.arange(max_steps)*dt, w[:,1],label =r"$\omega_{y}$")
    plt.plot(np.arange(max_steps)*dt, w[:,2],label =r"$\omega_{z}$")
    # plt.title('Angular velocity')
    plt.ylabel('angular velocity [rad/s]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.savefig(curr_dir + "/results/td3_eval/plot_omega.png")

    # plt.figure(figsize=(yoko,tate),dpi=100)
    plt.subplot(232)
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
    # plt.savefig(curr_dir + "/results/td3_eval/plot_torque.png")

    angle = np.array([np.rad2deg(env.dcm2euler(env.quaternion2dcm(q[i,:]))).tolist() for i in range(max_steps)])
    angle = angle.reshape([-1,3])
    # plt.figure(figsize=(yoko,tate),dpi=100)
    plt.subplot(233)
    plt.plot(np.arange(max_steps)*dt, angle[:,0],label = r"$\phi$")
    plt.plot(np.arange(max_steps)*dt, angle[:,1],label = r"$\theta$")
    plt.plot(np.arange(max_steps)*dt, angle[:,2],label = r"$\psi$")
    # plt.title('Action')
    plt.ylabel('angle [deg]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.savefig(curr_dir + "/results/td3_eval/plot_angle.png")

    # # plt.figure(figsize=(yoko,tate),dpi=100)
    plt.subplot(234)
    plt.plot(np.arange(max_steps)*dt, k_hist,label = r"$\gain$")
    plt.ylabel('k gain')
    plt.xlabel(r'time [s]')
    plt.tight_layout()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.savefig(curr_dir + "/results/td3_eval/plot_k.png")

    plt.subplot(235)
    plt.plot(np.arange(max_steps)*dt, alpha_hist,label = r"$alpha$")
    plt.ylabel(r'$\alpha$')
    plt.xlabel(r'time [s]')
    plt.tight_layout()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.savefig(curr_dir + "/results/td3_eval/plot_k.png")

    # # plt.figure(figsize=(yoko,tate),dpi=100)
    plt.subplot(236)
    plt.plot(np.arange(max_steps)*dt, d_hist[:,0],label = r"$D_{1}$")
    # plt.plot(np.arange(max_steps)*dt, d_hist[:,1],label = r"$D_{2}$")
    # plt.plot(np.arange(max_steps)*dt, d_hist[:,2],label = r"$D_{3}$")
    # plt.plot(np.arange(max_steps)*dt, d_hist[:,3],label = r"$D_{4}$")
    plt.plot(np.arange(max_steps)*dt, d_hist[:,4],label = r"$D_{5}$")
    # plt.plot(np.arange(max_steps)*dt, d_hist[:,5],label = r"$D_{6}$")
    # plt.plot(np.arange(max_steps)*dt, d_hist[:,6],label = r"$D_{7}$")
    # plt.plot(np.arange(max_steps)*dt, d_hist[:,7],label = r"$D_{8}$")
    plt.plot(np.arange(max_steps)*dt, d_hist[:,8],label = r"$D_{9}$")
    # plt.title('Action')
    plt.ylabel('d')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.savefig(curr_dir + "/results/td3_eval/plot_d.png")
    plt.savefig(curr_dir + "/results/td3_eval/results.png")

    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps)*dt, r_hist[:,0],label = r"$q$ pnlty")
    plt.plot(np.arange(max_steps)*dt, r_hist[:,1],label = r"$\omega$ pnlty")
    plt.plot(np.arange(max_steps)*dt, r_hist[:,2],label = r"$\tau$ pnlty")
    plt.plot(np.arange(max_steps)*dt, r_hist[:,3],label = r"$\Delta\tau$ pnlty")
    plt.plot(np.arange(max_steps)*dt, np.sum(r_hist,axis = 1),label = r"$R_{total}$",linestyle='dotted')
    plt.plot(np.arange(max_steps)*dt, R,label = r"$total$")
    # plt.title('Action')
    plt.ylabel('reward')
    plt.xlabel(r'time [s]')
    plt.tight_layout()
    plt.legend()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/td3_eval/plot_reward.png")
    plt.show()
    #endregion
    # ----------------------------結果プロット終わり--------------------------------#
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

    dt = 0.1
    simutime = 50
    max_steps = int(simutime/dt) -1 # dt is 0.01

    for i in range(1, max_steps):
        action = np.squeeze(action)
        next_error_state, reward, done, next_state, _ = env.step(action)
        # env.render()
        # q=np.append(q,next_state[0].reshape(1,-1),axis=0)
        # qe=np.append(qe,next_error_state[0].reshape(1,-1),axis=0)
        # w=np.append(w,next_error_state[2].reshape(1,-1),axis=0)
        q=np.append(q,next_state[:4].reshape(1,-1),axis=0)
        qe=np.append(qe,next_error_state[:4].reshape(1,-1),axis=0)
        w=np.append(w,next_error_state[8:11].reshape(1,-1),axis=0)
        r += reward
        # state = next_state
        #----------------control law (PID controller)-----------------------
        action = -Kp@next_error_state[:4].reshape(-1,1)-Kd@next_error_state[8:11].reshape(-1,1)
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

    #region
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
    # plt.title('Angular velocity')
    plt.ylabel('angular velocity [rad/s]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
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
    #endregion

def env_adaptive():

    # simulation of the agent solving the cartpole swing-up problem
    env = make("SatelliteContinuous")
    # uncomment for recording a video of simulation
    # env = wrappers.Monitor(env, './video', force=True)

    curr_dir = os.path.abspath(os.getcwd())
    env.reset()
    print( "the multi is :" + str(env.multi))
    r = 0
    qe = np.empty((0,4))
    q = np.empty((0,4))
    w = np.empty((0,3))
    actions = np.empty((0,3))
    r_hist = np.empty((0,4))

    #----------------------control parameters----------------------------
    alpha = 0.5
    k = 2
    zeta = 0.2
    D = np.diag([4e-4,1,1,1,5.8e-4,1,1,1,5.2e-4])
    th = env.inertia_comb.flatten()
    th_e = np.array(env.inertia.flatten())
    #--------------------------------------------------------------------
    action = np.array([0,0,0]).reshape(1,3)
    actions = np.append(actions, action,axis=0)

    dt = 0.1
    simutime = 50

    max_steps = int(simutime/dt) -1 # dt is 0.01

    for i in range(1, max_steps):
        action = np.squeeze(action)
        next_error_state, reward, done, next_state, _ = env.step(action)
        # if i == 20/dt:
        #     env.inertia = env.inertia_comb   #region
        q=np.append(q,next_state[:4].reshape(1,-1),axis=0)
        qe=np.append(qe,next_error_state[:4].reshape(1,-1),axis=0)
        w=np.append(w,next_error_state[8:11].reshape(1,-1),axis=0)
        r_hist = np.append(r_hist, np.array([env.r1,env.r2,env.r3,env.r4]).reshape(1,-1),axis=0)
        r += reward
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

    #---------------------結果のプロット開始-------------------------#
    #region
    #-------------------plot settings------------------------------
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 8 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 8 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 8 # 軸だけ変更されます 
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams['axes.grid'] = True # make grid
    plt.rcParams["legend.loc"] = "best"         # 凡例の位置、"best"でいい感じのところ
    plt.rcParams["legend.frameon"] = True       # 凡例を囲うかどうか、Trueで囲う、Falseで囲わない
    plt.rcParams["legend.framealpha"] = 1.0     # 透過度、0.0から1.0の値を入れる
    plt.rcParams["legend.facecolor"] = "white"  # 背景色
    # plt.rcParams["legend.edgecolor"] = "black"  # 囲いの色
    plt.rcParams["legend.fancybox"] = True     # Trueにすると囲いの四隅が丸くなる
    tate = 4.0
    yoko = 8.0
    #------------------------------------------------
    plt.figure(figsize=(12,5),dpi=100)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    # plt.figure(figsize=(yoko,tate),dpi=100)
    plt.subplot(231)
    # plt.figure(figsize=(yoko,tate),dpi=100)
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
    # plt.savefig(curr_dir + "/results/adap_test/plot_quat.png")

    plt.subplot(232)
    # plt.figure(figsize=(yoko,tate),dpi=100)
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
    # plt.savefig(curr_dir + "/results/adap_test/plot_quate_error.png")

    # plt.figure(figsize=(yoko,tate),dpi=100)
    plt.subplot(233)
    plt.plot(np.arange(max_steps-1)*dt, w[:,0],label =r"$\omega_{x}$")
    plt.plot(np.arange(max_steps-1)*dt, w[:,1],label =r"$\omega_{y}$")
    plt.plot(np.arange(max_steps-1)*dt, w[:,2],label =r"$\omega_{z}$")
    # plt.title('Angular velocity')
    plt.ylabel('angular velocity [rad/s]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.savefig(curr_dir + "/results/adap_test/plot_omega.png")

    plt.subplot(234)
    # plt.figure(figsize=(yoko,tate),dpi=100)
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
    plt.subplot(235)
    # plt.figure(figsize=(yoko,tate),dpi=100)
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
    # plt.savefig(curr_dir + "/results/adap_test/plot_angle.png")

    plt.subplot(236)
    # plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,0],label = r"$q$ pnlty")
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,1],label = r"$\omega$ pnlty")
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,2],label = r"$\tau$ pnlty")
    plt.plot(np.arange(max_steps-1)*dt, r_hist[:,3],label = r"$\Delta\tau$ pnlty")
    plt.plot(np.arange(max_steps-1)*dt, np.sum(r_hist,axis = 1),label = r"$toal$",linestyle='dotted')
    # plt.title('Action')
    plt.ylabel('reward')
    plt.xlabel(r'time [s]')
    plt.tight_layout()
    plt.legend()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.savefig(curr_dir + "/results/adap_test/plot_reward.png")
    plt.show()    
    #endregion
    #---------------------結果のプロット終わり-----------------------#

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

    #----------------------control parameters----------------------------
    alpha = 0.5
    k = 2