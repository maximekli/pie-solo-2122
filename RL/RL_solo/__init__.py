from gym.envs.registration import register

register(id='hopper-v3',entry_point='envs:HopperEnv',) 
register(id='solo-v1', entry_point='envs:SoloEnv')