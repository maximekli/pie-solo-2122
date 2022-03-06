# coding: utf8

#####################
#  LOADING MODULES ##
#####################

import time
import numpy as np
import pybullet as p  # PyBullet simulator
import example_robot_data

from PD import PD
from initialization_simulation import configure_simulation, getPosVelJoints # Functions to initialize the simulation and retrieve joints positions/velocities
# from inverse_kinematics import *

robot   = example_robot_data.load('solo12')
q0      = robot.q0.copy()
q0[7]   = 0
q0[10]   = 0
q0[7:13]= q0[13:]
dt      = 1e-3

####################################
#  LOAD CONFIGURATION TRAJECTORY  ##
####################################

try:
    Q = np.load('Q.npy', allow_pickle=True)
except OSError:
    import inverse_kinematics
    Q = np.load('Q.npy', allow_pickle=True)
Q = Q[:-4]

################################
#  INITIALIZATION SIMULATION  ##
################################
    
# If True then we will sleep in the main loop to have a 1:1 ratio of (elapsed real time / elapsed time in the
# simulation)
realTimeSimulation                                  = True  # If True then we will sleep in the main loop to have a frequency of 1/dt
enableGUI                                           = True  # enable PyBullet GUI or not
robotId, solo, revoluteJointIndices, torques_ref    = configure_simulation(dt, enableGUI)
torques_sat                                         = 3 # N.m
# t_simu                                              = 0

for i in range(len(Q)+1000):
    # Time at the start of the loop
    if realTimeSimulation:
        t0 = time.perf_counter()

    if i < len(Q) :
        # Parameters for the PD controller
        Kp      = 8
        Kd      = 0.06

        # Joints configuration
        vq  = np.zeros(q0.size)
        q  = Q[i]
        
        # Get position and velocity of all joints in PyBullet (free flying base + motors)
        qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
        qa          = qa[7:]
        qa_dot      = qa_dot[6:]

        # Target position and velocity for all joints
        qa_ref      = np.array([q[7:]]).T  # target angular positions for the motors
        qa_dot_ref  = np.array([vq[7:]]).T  # target angular velocities for the motors

        # Call controller to get torques for all joints
        jointTorques = PD(qa_ref, qa_dot_ref, qa, qa_dot, dt, Kp, Kd, torques_sat, torques_ref)
        # jointTorques = c_salto_IK(qa, qa_dot, dt, robot, t_simu)
        # t_simu += dt

    # ''' STAY IN POSITION '''
    elif i < len(Q) + 500 :
        # Parameters for the PD controller
        Kp      = 8
        Kd      = 0.06

        # Get position and velocity of all joints in PyBullet (free flying base + motors)
        qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
        qa          = qa[7:]
        qa_dot      = qa_dot[6:]

        # Call controller to get torques for all joints
        jointTorques = PD(qa_ref, qa_dot_ref, qa, qa_dot, dt, Kp, Kd, torques_sat, torques_ref)

    # ''' GO BACK TO INITIALIZED POSITION '''
    else : 
        # Parameters for the PD controller
        Kp      = 8
        Kd      = 0.06

        # Joints configuration
        vq      = np.zeros(q0.size)
        q       = q0

        # Get position and velocity of all joints in PyBullet (free flying base + motors)
        qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
        qa          = qa[7:]
        qa_dot      = qa_dot[6:]

        # Target position and velocity for all joints
        qa_ref      = np.array([q[7:]]).T  # target angular positions for the motors
        qa_dot_ref  = np.array([vq[7:]]).T  # target angular velocities for the motors

        # Call controller to get torques for all joints
        jointTorques = PD(qa_ref, qa_dot_ref, qa, qa_dot, dt, Kp, Kd, torques_sat, torques_ref)

    # Set control torques for all joints in PyBullet
    p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)
    # Compute one step of simulation
    p.stepSimulation()

    # Sleep to get a real time simulation
    if realTimeSimulation:
        t_sleep = dt - (time.perf_counter() - t0)
        if t_sleep > 0:
            time.sleep(t_sleep)


# Shut down the PyBullet client
p.disconnect()
