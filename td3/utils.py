from environment import SatelliteContinuousEnv
from gym import make as gym_make
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import wandb


def make(env_name, *make_args, **make_kwargs):
    if env_name == "SatelliteContinuous":
        return SatelliteContinuousEnv()
    else:
        return gym_make(env_name, *make_args, **make_kwargs)


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    counter = 0
    try:
        with tqdm(range(max_episodes),leave=False) as pbar:
            for episode, ch in enumerate(pbar):
                pbar.set_description("[Train] Episode %d" % episode)
                # print("\nThe goal angle is: %d" + str(env.goalEuler))
            # for episode in range(max_episodes):
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

                        print("\nEpisode " + str(episode) + " total reward : " + str(episode_reward)+"\n")
                        break

    except KeyboardInterrupt:
        print('Training stopped manually!!!')
        pass

    return episode_rewards

def mini_batch_train_adaptive(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    counter = 0
    k_max = 10
    d_grad = 2500
    try:
        # with tqdm(range(max_episodes),leave=False) as pbar:
        #     for episode, ch in enumerate(pbar):
                # pbar.set_description("[Train] Episode %d" % episode)
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0    
            th_e = np.array(env.inertia.flatten())
            alpha = 0.5

            for step in range(max_steps):
                # pbar.set_postfix(OrderedDict(multi = env.multi, w_0= np.rad2deg(env.startOmega), steps = step))#OrderedDict(loss=1-episode/5, acc=episode/10))
                action = agent.get_action(state, (episode + 1) * (step + 1))
                action = (action + 1)/2
                #----------------control law (Adaptive controller)-----------------------
                # alpha = action[0]
                # k = action[1]
                k = action[0]*k_max
                d_tmp = [action[i+1]*2500 +500 for i in range(len(action)-1)]
                D = np.diag([1/d_tmp[0],1,1,1,1/d_tmp[1],1,1,1,1/d_tmp[2]])
                
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
                env.est_th = [th_e[0],th_e[4],th_e[8]]/(env.max_multi*np.diag(env.inertia))
                #---------------------------------------------------------------------
                next_error_state, reward, done, next_state, _ = env.step(input)
                # if done:
                #     reward = -1/500*step + 1#1/np.sqrt(2*np.pi*500**2)*np.exp(-steps**2/(2*500**2))
                # else:
                #     reward = 0
                agent.replay_buffer.push(state, action, reward, next_error_state, done)
                episode_reward += reward

                # update the agent if enough transitions are stored in replay buffer
                if len(agent.replay_buffer) > batch_size:
                    agent.update(batch_size)

                if done or step == max_steps - 1:
                    episode_rewards.append(episode_reward)
                    wandb.log({ "episode reward": episode_reward,
                                "critic_loss": agent.critic_loss_for_log,
                                "actor_loss": agent.actor_loss_for_log,
                                "target_multi": env.multi,
                                "number of steps": step})

                    print("\nEpisode " + str(episode) + " total reward:" + str(episode_reward)+ " steps:" + str(step) +  " target_multi:" + str(env.multi) + "\n")
                    break
                state = next_error_state

    except KeyboardInterrupt:
        print('Training stopped manually!!!')
        pass

    return episode_rewards