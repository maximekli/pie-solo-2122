from training.src.start_training import training as train
from gym.envs.registration import register

if __name__ == '__main__':
    register(id='solo-v1', entry_point='envs:SoloEnv')

    train()
