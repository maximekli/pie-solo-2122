#!/usr/bin/env python
import gym
import yaml
import pathlib
import numpy as np

from spinup.algos.pytorch.sac import core
from .sac import sac, MLPActorCriticSym
from spinup.utils.run_utils import setup_logger_kwargs

def training():

    last_time_steps = np.ndarray(0)

    outdir = './training/training_results/'
    config_dir = './training/config/training_config_sac.yaml'

    with open(config_dir) as file:
        data = yaml.safe_load(file)

    task_and_robot_environment_name = data['task_and_robot_environment_name']


    env = gym.make(task_and_robot_environment_name)

    # Network size
    hid = data['hid']
    l = data['l']
    ac_kwargs=dict(hidden_sizes=hid*l)
    # Random seed
    seed = data['seed']
    # An epoch consists of a fixed amount of steps.
    steps_per_epoch = data['steps_per_epoch']
    # We train for a fixed amount of epochs
    n_epochs = data['n_epochs']
    # Size of replay buffer
    replay_size = data['replay_size']
    # Discount factor. (Always between 0 and 1.)
    gamma = data['gamma']
    # polyak (float): Interpolation factor in polyak averaging for target networks.
    polyak = data['polyak']
    # learning rate
    lr = data['lr']
    # Entropy regularization coefficient.
    alpha = data['alpha']
    # Batch size
    batch_size = data['batch_size']
    # Number of steps for uniform-random action selection,
    # before running real policy. Helps exploration.
    start_steps = data['start_steps']
    # Number of env interactions to collect before starting to do gradient descent updates. 
    # Ensures replay buffer is full enough for useful updates.
    update_after = data['update_after']
    # Number of env interactions that should elapse between gradient descent updates. Note: Regardless of how long 
    #  you wait between updates, the ratio of env steps to gradient steps is locked to 1.
    update_every = data['update_every']
    # Number of episodes to test the deterministic policy at the end of each epoch.
    num_test_episodes = data['num_test_episodes']
    # maximum length of episode
    max_ep_len = data['max_ep_len']
    # Number of epochs between each policy/value function save
    save_freq = data['save_freq']

    logger_kwargs = setup_logger_kwargs(task_and_robot_environment_name,seed,outdir)
    log_file_dir = outdir+task_and_robot_environment_name+'/'+task_and_robot_environment_name+'_s'+str(seed)+'/'
    logger_obs_ac_args = {'output_dir':log_file_dir, 'output_fname':'obs_ac.csv'}

    # Set max timestep
    env.spec.timestep_limit = max_ep_len

    sac(env=env, 
        test_env = env,
        actor_critic=MLPActorCriticSym, #core.MLPActorCritic
        ac_kwargs=dict(hidden_sizes=hid*l),
        seed=seed, 
        steps_per_epoch=steps_per_epoch, 
        epochs=n_epochs, 
        replay_size=replay_size, 
        gamma=gamma, 
        polyak=polyak, 
        lr=lr, 
        alpha=alpha, 
        batch_size=batch_size, 
        start_steps=start_steps, 
        update_after=update_after, 
        update_every=update_every, 
        num_test_episodes=num_test_episodes, 
        max_ep_len=max_ep_len, 
        logger_kwargs=logger_kwargs,
        logger_obs_ac_args=logger_obs_ac_args, 
        save_freq=save_freq,
        symmetry=True,
        load=False)
    env.close()