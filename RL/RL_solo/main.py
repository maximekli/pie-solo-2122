from training.src.start_training import training as train
from gym.envs.registration import register
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    register(id='hopper-v3', entry_point='envs:HopperEnv')
    register(id='solo-v1', entry_point='envs:SoloEnv')

    train()

    data = pd.read_csv('training/training_results/solo-v1/solo-v1_s1/progress.txt',sep='\s+',header=0)
    data = pd.DataFrame(data)
    plt.plot(data['Epoch'], data['AverageEpRet'],'r--',label='Average')
    plt.plot(data['Epoch'], data['MaxEpRet'],'b--',label='Max')
    plt.plot(data['Epoch'], data['MinEpRet'],'b--',label='Min')
    plt.xlabel('Epoch')
    plt.ylabel('EpRet')
    plt.legend()
    plt.show()
    plt.savefig('training/training_results/solo-v1/solo-v1_s1/rewards.png')
