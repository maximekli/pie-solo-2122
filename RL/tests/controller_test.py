import gym
from RL.RL_solo.envs.solo_v1 import SoloEnv
import numpy as np

env = SoloEnv(terminate_when_unhealthy=False)

for i_episode in range(100):
    observation = env.reset()

    for t in range(1000):
        env.render()
        action = np.zeros(env.action_space.shape[0])
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
