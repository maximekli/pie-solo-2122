from gym.envs.registration import register

register(id='hopper_v3',entry_point='RL_solo.envs:HopperEnv',) 
register(id='solo_v1', entry_point='RL_solo.envs:SoloEnv')