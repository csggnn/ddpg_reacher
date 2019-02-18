# DDPG Reacher

**DDPG Reacher** is a PyTorch project implementing a reinforcement learning 
agent controlling a ouble-jointed arm in a *Reacher Environment*. 

In a **Reacher Environment**, a targets moves around a robotic arm, and the reinforcement learning agent must be trained to 
maintain its hand near the target location for as many time steps as possible.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of
the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints of the robotic arm.
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The environment is considered 
solved when an averge reward of 30 is achieved in 100 subsequent episodes.

The agent in this project implements [DDPG](https://arxiv.org/abs/1509.02971) and was developed as
 a solution to  the continuous contol project for the 
[Udacity Deep Reinforcement Learning Nanodegree](https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Installation

In order to run the code in this project, you will need a python3.6 environment with the 
following packages:
- numpy
- matplotlib
- pickle
- torch

You will also need to install the Unity Reacher Envronment: 
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- [Win_32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- [Win_64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)


If you want to run *param_optim_analysis.py*, also install the *pandas* package

The project also allows for training a DDPG agent for continuous action space **gym** enviroments. 
- More information on **gym** and installation instructions at this [link](https://github.com/openai/gym)
- If you decide not to install **gym**, just comment line 9  in **ddpg_reacher_SOLUTION.py**


## Usage

Simply run ddpg_reacher_SOLUTION to a trained DDPG agent solving the Reacher environment.

The ddpg_reacher_SOULUTION simply calls *ddpg_reacher_solution("show", 500)*, which loads a DDPG solving the Reacher 
Environment after 500 training episodes.

Alternatively, you cn edit the main routine to call ddpg_reacher_solution("learn") and train a new DDPG agent. 
 
More details on the file structure, implementation choices and parameters in Report.md