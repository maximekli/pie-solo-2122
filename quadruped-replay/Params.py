'''This class stores parameters about the replay'''
import numpy as np

class Params():
    def __init__(self):
        # Replay parameters
        # self.replay_path = "demo_replay.npz"
        # self.replay_path = 'converted.simple_jumping.npz'
        # self.replay_path = 'converted.yaw_jumping.npz'
        # self.replay_path = 'converted.half_backflip.npz'
        # self.replay_path = 'with_gains.trimmed_padded.converted.simple_jumping.npz'
        # self.replay_path = 'with_gains.trimmed_padded.converted.yaw_jumping.npz'
        self.replay_path = 'with_gains.trimmed_padded.converted.half_backflip.npz'

        self.SIMULATION = True  #  Run the replay in simulation if True
        self.LOGGING = False  #  Save the logs of the experiments if True
        self.PLOTTING = False  #  Plot the logs of the experiments if True

        # Control parameters
        self.dt = 0.001  # Time step of the replay
        self.Kp = np.array([6.0, 6.0, 6.0])  # Proportional gains for the PD+
        self.Kd = np.array([0.3, 0.3, 0.3])  # Derivative gains for the PD+
        self.Kff = 1.0  # Feedforward torques multiplier for the PD+

        # Other parameters
        self.config_file = "config_solo12.yaml"  #  Name of the yaml file containing hardware information

        # End position goal
        solo_q0_flipped = np.array([0., 0., 0.235, 0., 0., 0., 1.,
                                    -0.1, np.pi-0.8, np.pi-1.6,
                                    0.1, np.pi-0.8, np.pi-1.6,
                                    -0.1, np.pi-0.8, 1.6,
                                    0.1, np.pi-0.8, 1.6])

        solo_q0 = np.array([0., 0., 0.235, 0., 0., 0., 1.,
                            0.1, 0.8, -1.6,
                            -0.1, 0.8, -1.6,
                            0.1, -0.8, 1.6,
                            -0.1, -0.8, 1.6])

        self.q_end = solo_q0_flipped if 'half_backflip' in self.replay_path else solo_q0

        # Torque saturation
        self.tau_sat = 2.5 # N.m
