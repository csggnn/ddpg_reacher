from unityagents import UnityEnvironment
import numpy as np



def show_reacher_info(env):
    # reset the environment
    env_info = env.reset(train_mode=True)[env.brain_names[0]]
    brain = env.brains[env.brain_names[0]]
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


def random_reacher_run(env,seed=None):
    # Taking Random Actions in the Environment

    # env.reset(train_mode=False)[env.brain_names[0]] will enable me to read env_info.vector_observations
    # but seems not to completely reset the environment. Without a new env.reset() call the environment would
    # be stuck on a second run.

    env.reset()
    env_info = env.reset(train_mode=False)[env.brain_names[0]] # reset the environment
    brain = env.brains[env.brain_names[0]]
    np.random.seed(seed)
    states = env_info.vector_observations                  # get the current state (for each agent)
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[env.brain_names[0]]   # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


def train_ddpg_on_reacher(env, seed=None):
    env.reset()
    env_info = env.reset(train_mode=True)[env.brain_names[0]]
    brain = env.brains[env.brain_names[0]]
    np.random.seed(seed)

    print("not yet implemented")

def show_ddpg_on_reacher(env, seed=None):
    env.reset()
    env_info = env.reset(train_mode=True)[env.brain_names[0]]
    brain = env.brains[env.brain_names[0]]
    print("not yet implemented")


def ddpg_reacher_solution(mode="random"):

    # Reacher environment: move a double-jointed arm to target locations.
    # A reward of `+0.1` is provided for each step that the agent's hand is in the goal location.
    # The goal of the agent is to maintain its position at the target location for as many time steps as possible.
    env = UnityEnvironment(file_name='../Reacher_Linux/Reacher.x86_64', seed=0)

    show_reacher_info(env)

    if mode is "random":
        print("Taking random actions in the Reacher Environment")
        random_reacher_run(env, seed=0)
        random_reacher_run(env, seed=0)

    elif mode is "train":
        print("Training a ddpg Agent in the Reacher Environment")
        train_ddpg_on_reacher(env, seed=0)
    elif mode is "show":
        print("Showing the behavior of a trained ddpg Agent in the Reacher Environment")
        show_ddpg_on_reacher(env, seed=0)

    else:
        print("INVALID MODE: please select a valid mode of operation among 'random', 'train' and 'show'")

    env.close()
    return


if __name__ == "__main__":
    ddpg_reacher_solution("random")
