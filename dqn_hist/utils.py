from environment import SatelliteContinuousEnv
from gym import make as gym_make
from tqdm import tqdm
from collections import OrderedDict
import numpy as np


def make(env_name, *make_args, **make_kwargs):
    if env_name == "SatelliteContinuous":
        return SatelliteContinuousEnv()
    else:
        return gym_make(env_name, *make_args, **make_kwargs)

#10進数をn進数に変換する
def Base_10_to_n(X, n):
    if (int(X/n)):
        return Base_10_to_n(int(X/n), n)+str(X%n)
    return str(X%n)

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    counter = 0
    try:
        with tqdm(range(max_episodes),leave=False) as pbar:
            for episode, ch in enumerate(pbar):
                pbar.set_description("[Train] Episode %d" % episode)
                state = env.reset()
                episode_reward = 0    
                
                for step in range(max_steps):
                    pbar.set_postfix(OrderedDict(goal= env.goalEuler, steps = step))#OrderedDict(loss=1-episode/5, acc=episode/10))
                    action = agent.get_action(state, (episode + 1) * (step + 1))
                    next_error_state, reward, done, next_state, _ = env.step(action)
                    agent.replay_buffer.push(state, action, reward, next_error_state, done)
                    episode_reward += reward

                    # update the agent if enough transitions are stored in replay buffer
                    if len(agent.replay_buffer) > batch_size:
                        agent.update(batch_size)

                    if done or step == max_steps - 1:
                        episode_rewards.append(episode_reward)
                        # Count number of consecutive games with cumulative rewards >-55 for early stopping
                        if episode_reward > -55:
                            counter += 1
                        else:   
                            counter = 0
                        print("\nEpisode " + str(episode) + " total reward : " + str(episode_reward)+"\n")
                        break
                    # if counter == 10:
                        # break
    except KeyboardInterrupt:
        print('Training stopped manually!!!')
        pass

    return episode_rewards

def mini_batch_train_adaptive(env, agent, max_episodes, max_steps, batch_size, time_window):
    episode_rewards = []
    counter = 0
    k_delta = 0.01
    D_delta = 1e-5
    state_num = 7 #姿勢角４・角速度３
    try:
        with tqdm(range(max_episodes),leave=False) as pbar:
            for episode, ch in enumerate(pbar):
                pbar.set_description("[Train] Episode %d" % episode)
                state_hist = np.zeros(state_num*time_window)
                next_state_hist = np.zeros(state_num*time_window)
                obs = env.reset()
                th_e = np.array(env.inertia.flatten())
                episode_reward = 0    
                alpha = 0.5
                k = 0.8
                D = np.diag([4e-4,1,1,1,5.8e-4,1,1,1,5.2e-4])
                for step in range(max_steps):
                    pbar.set_postfix(OrderedDict(multi = env.multi, w_0= np.rad2deg(env.startOmega), steps = step))#OrderedDict(loss=1-episode/5, acc=episode/10))
                    
                    if step > time_window:
                        action = agent.get_action(state_hist, episode)
                        action.flatten()
                    #----------------control law (Adaptive controller)-----------------------
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
                            para = np.empty((4,1))
                            if k<0 or D[0,0]<0 or D[4,4]<0 or D[8,8] <0:
                                env.neg_param_flag = False
                    W = obs[4:7]
                    x1 = obs[1:4]
                    x2 = alpha*x1 + W
                    dqe = env.quaternion_differential(W,obs[0:4])
                    Y = np.array([[alpha*dqe[1], alpha*dqe[2], alpha*dqe[3], W[0]*W[2], W[1]*W[2], W[2]*W[2], -W[0]*W[2], -W[1]*W[1], -W[1]*W[2]],
                        [-W[0]*W[2], -W[1]*W[2], -W[2]*W[2], alpha*dqe[1], alpha*dqe[2], alpha*dqe[3], W[0]*W[0], W[0]*W[1], W[0]*W[2]],
                        [W[0]*W[1], W[1]*W[1], W[1]*W[2], -W[0]*W[0], -W[0]*W[1], -W[0]*W[2], alpha*dqe[1], alpha*dqe[2], alpha*dqe[3]]])
                    input = -0.5*x1 -Y@th_e - k*x2

                    dth = np.linalg.inv(D) @ Y.T @ x2
                    th_e += env.dt*dth
                    #---------------------------------------------------------------------
                    next_error_state, reward, done, next_state, _ = env.step(input)
                    next_state_hist[1:] = next_state_hist[0:-1]
                    next_state_hist[:state_num] = next_error_state
                    
                    if step > time_window:
                        agent.replay_buffer.push(state_hist, action, reward, next_state_hist, done)
                        td_error = agent.get_td_error(state_hist, action, next_state_hist, reward)
                        agent.td_error_memory.push(td_error)
                    episode_reward += reward

                    # update the agent if enough transitions are stored in replay buffer
                    if len(agent.replay_buffer) > batch_size:
                        agent.update(batch_size,episode)

                    if done or step == max_steps - 1:
                        episode_rewards.append(episode_reward)
                        agent.update_td_error_memory()
                        # Count number of consecutive games with cumulative rewards >-55 for early stopping
                        if episode_reward > -55:
                            counter += 1
                        else:   
                            counter = 0
                        print("\nEpisode " + str(episode) + " total reward : " + str(episode_reward)+"\n")
                        break
                    
                    #状態量の更新
                    obs = next_error_state
                    state_hist = next_state_hist

    except KeyboardInterrupt:
        print('Training stopped manually!!!')
        pass

    return episode_rewards