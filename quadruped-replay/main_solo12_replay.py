# coding: utf8

import os
import numpy as np
from Params import Params
from utils import *

params = Params()  # Object that holds all controller parameters
np.set_printoptions(precision=3, linewidth=400)

class Replay():
    """
    Feel free to modify this class if your replay file has a different format
    """
    def __init__(self, params):
        """
        Args:
            params: store control parameters
        """
        # Read replay data
        replay = np.load(params.replay_path)

        # Retrieve actuator commands
        self.q = replay['q'][7:, 1:].transpose().copy()  # Should be N by 12
        self.v = replay['v'][6:, 1:].transpose().copy()  # Should be N by 12
        self.tau = replay['tau'][:, 1:].transpose().copy()  # Should be N by 12

        # Number of samples
        self.N = self.q.shape[0]

        # Control gains
        self.P = replay['Kp'][:, 1:].transpose().copy() if 'Kp' in replay.files else np.tile(params.Kp, (self.N, 4))  # N by 12
        self.D = replay['Kd'][:, 1:].transpose().copy() if 'Kd' in replay.files else np.tile(params.Kd, (self.N, 4))  # N by 12
        self.FF = np.tile(params.Kff, (self.N, 12))  # N by 12
        self.tau_sat = params.tau_sat

        # Initial positions and torques
        self.q0 = self.q[0, :]
        self.tau0 = self.FF[0, :] * self.tau[0, :]

        self.q_end = params.q_end[7:]
        self.tau_end = np.zeros(12)


def replay_loop():
    """
    Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key
    """

    # Load replay data
    replay = Replay(params)

    # INITIALIZATION ***************************************************
    device, logger, qc = initialize(params, replay.q0, replay.tau0, replay.N)

    # PUT ON THE FLOOR
    put_on_the_floor(device, params, replay.q0, replay.tau0)

    # REPLAY LOOP ***************************************************
    k = 0
    while ((not device.is_timeout) and (k < replay.N)):

        # Update sensor data (IMU, encoders, Motion capture)
        device.parse_sensor_data()

        # Check that the initial position of actuators is not too far from the
        # desired position of actuators to avoid breaking the robot
        if (k <= 10):
            if np.max(np.abs(replay.q[k, :] - device.joints.positions)) > 0.15:
                print("DIFFERENCE: ", replay.q[k, :] - device.joints.positions)
                print("q_des: ", replay.q[k, :])
                print("q_mes: ", device.joints.positions)
                break

        # Set desired quantities for the actuators
        device.joints.set_position_gains(replay.P[k, :])
        device.joints.set_velocity_gains(replay.D[k, :])
        device.joints.set_desired_positions(replay.q[k, :])
        device.joints.set_desired_velocities(replay.v[k, :])
        device.joints.set_torques(replay.FF[k, :] * replay.tau[k, :])
        device.joints.set_tau_sat(replay.tau_sat)

        # Call logger if necessary
        if params.LOGGING or params.PLOTTING:
            logger.sample(device, qc)

        # Send command to the robot
        device.send_command_and_wait_end_of_cycle(params.dt)

        # Increment counter
        k += 1

    # WAIT IN POSITION
    wait_in_position(device, params, replay.q_end, replay.tau_end)

    # DAMPING TO GET ON THE GROUND PROGRESSIVELY *********************
    damping(device, params)

    # FINAL SHUTDOWN *************************************************
    shutdown(device, params, replay, logger)

    return 0


def main():
    """
    Main function
    """

    if not params.SIMULATION:  # When running on the real robot
        os.nice(-20)  #  Set the process to highest priority (from -20 highest to +20 lowest)
    replay_loop()
    quit()


if __name__ == "__main__":
    main()
