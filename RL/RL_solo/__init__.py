from gym.envs.registration import register

register(id='hopper-v3',entry_point='RL_solo.envs:HopperEnv',) 
register(id='solo-v1', entry_point='RL_solo.envs:SoloEnv')
