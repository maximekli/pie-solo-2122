from training.src.start_training import training as train
from gym.envs.registration import register
import numpy as np


if __name__ == '__main__':
    register(id='hopper-v3', entry_point='envs:HopperEnv')
    register(id='solo-v1', entry_point='envs:SoloEnv')

    train()
