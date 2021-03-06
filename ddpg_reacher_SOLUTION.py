from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
from collections import deque
import torch
import pickle
import matplotlib.pyplot as plt
import time
from continuous_gym_env import ContinuousGymEnv
from reacher1_env import Reacher1Env

def show_info(env):
    # size of each action
    print('Size of each action:', env.get_action_space_size())
    print('There are {} agents. Each observes a state with length: {}'.format(1, env.get_state_space_size()))
    print('The state for the first agent looks like:', env.reset())


def random_run(env,seed=None):
    # Taking Random Actions in the Environment

    # env.reset(train_mode=False)[env.brain_names[0]] will enable me to read env_info.vector_observations
    # but seems not to completely reset the environment. Without a new env.reset() call the environment would
    # be stuck on a second run.

    state=env.reset()
    np.random.seed(seed)
    action_size = env.get_action_space_size()
    score = 0                                               # initialize the score (for each agent)
    while True:
        action = np.random.randn(action_size)               # select an action (for each agent)
        action = np.clip(action, -1, 1)
        [next_state, reward, done, x ] = env.step(action)   # all actions between -1 and 1
        score += reward                                     # update the score (for each agent)
        state = next_state                                  # roll over states to next time step
        time.sleep(0.02)
        if np.any(done):                                    # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(score))


def train_ddpg(env, seed=None, agent_pars=None, starting_episode=0):

    env.reset()
    np.random.seed(seed)
    agent = Agent(random_seed=seed, action_size= env.get_action_space_size(), state_size=env.get_state_space_size(), parameter_dict=agent_pars)
    pickle.dump(agent_pars, open("checkpoints/last_run/agent_pars.p", "wb"))




    train_episodes=1000
    #max_t_low=300 #the episode will end by itself
    #max_t_high=1000

    max_t_low=2000
    max_t_high=2000

    expected_t = 1000

    max_t=max_t_low


    use_noise_p=0.3
    scores_deque = deque(maxlen=100)
    scores = []

    if starting_episode>0:
        agent.actor_local.load_state_dict(torch.load("checkpoints/last_run/actor"+str(starting_episode)+".pth"))
        agent.actor_target.load_state_dict(torch.load("checkpoints/last_run/actor_tg"+str(starting_episode)+".pth"))
        agent.critic_local.load_state_dict(torch.load("checkpoints/last_run/critic"+str(starting_episode)+".pth"))
        agent.critic_target.load_state_dict(torch.load("checkpoints/last_run/critic_tg"+str(starting_episode)+".pth"))
        scores=pickle.load(open("checkpoints/last_run/score.p", "rb"))
        agent.noise_factor = max(agent.noise_min_factor, agent.noise_factor*pow(agent.noise_decay, expected_t*starting_episode))

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i_episode in range(1, train_episodes + 1):
        state=env.reset()
        agent.reset()
        score=0
        natural_end=False
        noise_episode = np.random.rand()<use_noise_p
        for t in range(max_t):
            action = agent.act(state, add_noise=noise_episode)
            [next_state, reward, done, x ] = env.step(action)
            if i_episode%50 ==0:
                env.render()
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                natural_end = True
                break

        if   natural_end is False:
            print("timeout end")
            max_t=min(max_t_high, int(max_t*1.05))
        else:
            max_t=max(max_t_low, int(max_t*0.99))

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score),
              end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoints/last_run/actor'+str(i_episode)+'.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoints/last_run/critic'+str(i_episode)+'.pth')
            torch.save(agent.actor_target.state_dict(), 'checkpoints/last_run/actor_tg'+str(i_episode)+'.pth')
            torch.save(agent.critic_target.state_dict(), 'checkpoints/last_run/critic_tg'+str(i_episode)+'.pth')
            pickle.dump(scores, open("checkpoints/last_run/score.p", "wb"))

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if i_episode % 20 == 0:
            ax.clear()
            ax.plot(np.arange(1, len(scores) + 1), scores)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.draw()
            plt.pause(.001)

def show_ddpg_on_reacher(env, seed=None, starting_episode=1000, agent_pars=None):
    state=env.reset()
    selected_i = starting_episode
    agent = Agent(random_seed=seed, action_size= env.get_action_space_size(), state_size=env.get_state_space_size(), parameter_dict=agent_pars)
    agent.actor_local.load_state_dict(torch.load('checkpoints/final/actor'+str(selected_i)+'.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoints/final/critic' + str(selected_i) + '.pth'))
    agent.actor_target.load_state_dict(torch.load('checkpoints/final/actor_tg' + str(selected_i) + '.pth'))
    agent.critic_target.load_state_dict(torch.load('checkpoints/final/critic_tg' + str(selected_i) + '.pth'))

    score = pickle.load(open('checkpoints/final/score.p', 'rb'))
    plt.ion()
    plt.figure()
    plt.plot(score)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.pause(0.01)
    plt.waitforbuttonpress()

    for i in range(10):
        score =0;
        while True:
            action = agent.act(state, False)            # select an action (for each agent)
            [next_state, reward, done, x ] = env.step(action)   # all actions between -1 and 1
            score += reward                                     # update the score (for each agent)
            state = next_state                                  # roll over states to next time step
            time.sleep(0.001)
            if np.any(done):                                    # exit loop if episode finished
                break
        print('Total score (averaged over agents) this episode: {}'.format(score))


def ddpg_reacher_solution(mode="random", starting_episode=0):

    # Reacher environment: move a double-jointed arm to target locations.
    # A reward of `+0.1` is provided for each step that the agent's hand is in the goal location.
    # The goal of the agent is to maintain its position at the target location for as many time steps as possible.
    env = Reacher1Env(seed=0)
    #env = ContinuousGymEnv('BipedalWalker-v2', seed=0)

    #env = ContinuousGymEnv('LunarLanderContinuous-v2', seed=0)

    show_info(env)

    if mode is "random":
        print("Taking random actions in the Reacher Environment")
        random_run(env, seed=0)
        random_run(env, seed=0)

    elif mode is "train":
        ag_pars = {}
        ag_pars["BUFFER_SIZE"] = int(1e5)  # replay buffer size
        ag_pars["BATCH_SIZE"] = 64  # minibatch size
        ag_pars["GAMMA"] = 0.92  # discount factor
        ag_pars["TAU"] = 1e-2  # for soft update of target parameters
        ag_pars["LR_ACTOR"] = 1e-5#1e-5#  # learning rate of the actor
        ag_pars["LR_CRITIC"] = 1e-4#1e-4#  # learning rate of the critic
        ag_pars["WEIGHT_DECAY"] = 0  # L2 weight decay
        ag_pars["LEARN_EVERY"] = 1  # learn only once every LEARN_EVERY actions
        ag_pars["ACTOR_FC1"] = 400#100#
        ag_pars["ACTOR_FC2"] = 300#100#
        ag_pars["CRITIC_FC1"] = 400#100#
        ag_pars["CRITIC_FC2"] = 300#100#
        ag_pars["NOISE_DECAY"] = 0.99999
        ag_pars["NOISE_START"] = 0.02
        ag_pars["NOISE_MIN"] = 0.0002
        print("Training a ddpg Agent in the Reacher Environment")
        train_ddpg(env, seed=0, agent_pars=ag_pars, starting_episode=starting_episode)
    elif mode is "show":
        ag_pars = {}
        ag_pars["BUFFER_SIZE"] = int(1e5)  # replay buffer size
        ag_pars["BATCH_SIZE"] = 64  # minibatch size
        ag_pars["GAMMA"] = 0.99  # discount factor
        ag_pars["TAU"] = 1e-2  # for soft update of target parameters
        ag_pars["LR_ACTOR"] = 1e-5#1e-5#  # learning rate of the actor
        ag_pars["LR_CRITIC"] = 1e-4#1e-4#  # learning rate of the critic
        ag_pars["WEIGHT_DECAY"] = 0  # L2 weight decay
        ag_pars["LEARN_EVERY"] = 1  # learn only once every LEARN_EVERY actions
        ag_pars["ACTOR_FC1"] = 400#100#
        ag_pars["ACTOR_FC2"] = 300#100#
        ag_pars["CRITIC_FC1"] = 400#100#
        ag_pars["CRITIC_FC2"] = 300#100#
        ag_pars["NOISE_DECAY"] = 0.99999
        ag_pars["NOISE_START"] = 0.02
        ag_pars["NOISE_MIN"] = 0.0002
        print("Showing the behavior of a trained ddpg Agent in the Reacher Environment")
        show_ddpg_on_reacher(env, seed=0, starting_episode=starting_episode)

    else:
        print("INVALID MODE: please select a valid mode of operation among 'random', 'train' and 'show'")

    env.close()
    return


if __name__ == "__main__":
    # show agent in action after 100, 300, 500, 600 or 1000 iterations (environment solved at 500)
    ddpg_reacher_solution("show", 500)
    # retrain agent
    #ddpg_reacher_solution("train", 0)