import gym
from RL.RL_solo.envs.solo_v1 import SoloEnv
from RL.RL_solo.envs.hopper_v3 import HopperEnv
from time import sleep
import numpy as np
#from IPython.display import clear_output

env = SoloEnv(terminate_when_unhealthy=True) #HopperEnv() #SoloEnv()

for i_episode in range(100):
    observation = env.reset()

    for t in range(200*5):
        env.render()
        #print(observation)

        a = [0, 0, 0]
        # action for actuators of type PD
        # action = np.concatenate([a,a,a,a ,np.zeros(12)]) #env.action_space.sample()
        # action for actuators of type motor
        action = np.concatenate([a,a,a,a])
        #print(action)
        
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
