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
    k_max = 7
    alpha_max = 1
    d_grad = 2500
    try:
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0    
            th_e = np.array(env.inertia.flatten()*env.multi)

            for step in range(max_steps):
                action = agent.get_action(state)
                para_candi = (action + 1)/2
                #----------------control law (Adaptive controller)-----------------------
                # alpha = action[0]
                # k = action[1]
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
                env.est_th = th_e.flatten()/25
                # env.est_th = ([th_e[0],th_e[4],th_e[8]])/((env.max_multi+1)*np.diag(env.inertia))
                #---------------------------------------------------------------------
                next_error_state, reward, done, next_state, _ = env.step(input)
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