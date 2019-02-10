import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from reacher1_env import Reacher1Env
from ddpg_agent import Agent

import pickle
import torch

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

env = Reacher1Env(seed=0)


pars_sel={}
pars_sel["BUFFER_SIZE"]  = [int(1e5)]  # replay buffer size
pars_sel["BATCH_SIZE"]   = [32, 128]  # minibatch size
pars_sel["GAMMA"]        = [0.99, 0.995]  # discount factor
pars_sel["TAU"]          = [1e-2, 1e-3]  # for soft update of target parameters
pars_sel["LR_ACTOR"]     = [1e-4, 1e-5]  # learning rate of the actor
pars_sel["LR_CRITIC"]    = [1e-3, 1e-4]  # learning rate of the critic
pars_sel["WEIGHT_DECAY"] = [0, 0.01, 0.0001]  # L2 weight decay
pars_sel["LEARN_EVERY"]  = [1, 4, 16] # learn only once every LEARN_EVERY actions
pars_sel["ACTOR_FC1"]    = [400]
pars_sel["ACTOR_FC2"]    = [300]
pars_sel["CRITIC_FC1"]   = [400, 800]
pars_sel["CRITIC_FC2"]   = [300, 600]


num_tests = 5000

# number of actions
print('Number of actions:', env.get_action_space_size())
# examine the state space
print('States look like:', env.get_state())
print('States have length:', env.get_state_space_size())
# [50,20,10]
results = []
result_pars = []
for test_i in range(num_tests):
    pars = {}

    pars["BUFFER_SIZE"]  =random.choice(pars_sel["BUFFER_SIZE"]  )
    pars["BATCH_SIZE"]   =random.choice(pars_sel["BATCH_SIZE"]   )
    pars["GAMMA"]        =random.choice(pars_sel["GAMMA"]        )
    pars["TAU"]          =random.choice(pars_sel["TAU"]          )
    pars["LR_ACTOR"]     =random.choice(pars_sel["LR_ACTOR"]     )
    pars["LR_CRITIC"]    =random.choice(pars_sel["LR_CRITIC"]    )
    pars["WEIGHT_DECAY"] =random.choice(pars_sel["WEIGHT_DECAY"] )
    pars["LEARN_EVERY"]  =random.choice(pars_sel["LEARN_EVERY"]  )
    pars["ACTOR_FC1"]    =random.choice(pars_sel["ACTOR_FC1"]    )
    pars["ACTOR_FC2"]    =random.choice(pars_sel["ACTOR_FC2"]    )
    pars["CRITIC_FC1"]   =random.choice(pars_sel["CRITIC_FC1"]   )
    pars["CRITIC_FC2"]   =random.choice(pars_sel["CRITIC_FC2"]   )

    print(">>> test "+str(test_i))
    print(">>> parameters:")
    print(pars)

    agent = Agent(state_size=env.get_state_space_size(),
                   action_size=env.get_action_space_size(),
                   random_seed=test_i,
                   parameter_dict=pars)
    env.reset()

    curr_score = 0
    score_window = deque(maxlen=100)  # last 100 scores
    score_list = []
    mean_score_list = []
    running_score = 0
    max_ep_len = 10000
    train_episodes = 301

    for i_episode in range(1, train_episodes + 1):
        state=env.reset()
        agent.reset()
        score=0
        for t in range(max_ep_len):
            action = agent.act(state)
            [next_state, reward, done, x ] = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        score_window.append(score)
        score_list.append(score)
        if i_episode % 10 == 0:
            print("episode " + str(i_episode) + ", mean score: " + str(np.mean(score_window)))
        if i_episode % 50 == 0:
            mean_score_list.append(np.mean(score_window))
    print("test completed with scores: " + str(mean_score_list))
    torch.save(agent.actor_local.state_dict(), "test_out_2/weights_test_{:03d}_actor.pth".format(test_i))
    torch.save(agent.critic_local.state_dict(), "test_out_2/weights_test_{:03d}_critic.pth".format(test_i))
    pickle.dump((score_list, mean_score_list, pars), open("test_out_2/scores_and_pars_test_{:03d}.p".format(test_i), "wb"))

