# coding: utf8

#####################
#  LOADING MODULES ##
#####################

import imp
import time
import numpy as np
from numpy.linalg import norm, pinv
from scipy.integrate import quad
import matplotlib.pyplot as plt

import pybullet as p  # PyBullet simulator
import pybullet_data
import pinocchio as pin
import example_robot_data
# Functions to initialize the simulation and retrieve joints positions/velocities
from PD import PD
from initialization_simulation import configure_simulation, getPosVelJoints
from inverse_kinematics import *



################################
#  INITIALIZATION SIMULATION  ##
################################

Q = computeTrajectory()
    
# If True then we will sleep in the main loop to have a 1:1 ratio of (elapsed real time / elapsed time in the
# simulation)
realTimeSimulation                               = True  # If True then we will sleep in the main loop to have a frequency of 1/dt
enableGUI                                        = True  # enable PyBullet GUI or not
robotId, solo, revoluteJointIndices, torques_ref = configure_simulation(dt, enableGUI)
torques_sat = 3 # N.m


for i in range(1000+len(T)+1000):
    # Time at the start of the loop
    if realTimeSimulation:
        t0 = time.perf_counter()

    # ''' INITIALIZED POSITION '''
    if i < 1000 :
        # Parameters for the PD controller
        Kp      = 8
        Kd      = 0.06

        # Joints configuration
        dq      = np.zeros(q_0.size)
        q       = q_0

        # Get position and velocity of all joints in PyBullet (free flying base + motors)
        qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
        qa          = qa[7:]
        qa_dot      = qa_dot[6:]

        # Target position and velocity for all joints
        qa_ref      = np.array([q[7:]]).T  # target angular positions for the motors
        qa_dot_ref  = np.array([dq[7:]]).T  # target angular velocities for the motors

        # Call controller to get torques for all joints
        jointTorques = PD(qa_ref, qa_dot_ref, qa, qa_dot, dt, Kp, Kd, torques_sat, torques_ref)
        
    # ''' JUMP TRAJECTORY '''
    elif (1000 <= i) and (i < 1000+len(T)) :
        # Parameters for the PD controller
        Kp      = 8
        Kd      = 0.06

        # Joints configuration
        dq      = np.zeros(q_0.size)
        msg, q  = Q[i-1000]
        print(f"{i} : {msg}")

        # Get position and velocity of all joints in PyBullet (free flying base + motors)
        qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
        qa          = qa[7:]
        qa_dot      = qa_dot[6:]

        # Target position and velocity for all joints
        qa_ref      = np.array([q[7:]]).T  # target angular positions for the motors
        qa_dot_ref  = np.array([dq[7:]]).T  # target angular velocities for the motors

        # Call controller to get torques for all joints
        jointTorques = PD(qa_ref, qa_dot_ref, qa, qa_dot, dt, Kp, Kd, torques_sat, torques_ref)

    # ''' GO BACK TO INITIALIZED POSITION '''
    else : 
        # Parameters for the PD controller
        Kp      = 8
        Kd      = 0.06

        # Joints configuration
        dq      = np.zeros(q_0.size)
        q       = q_0

        # Get position and velocity of all joints in PyBullet (free flying base + motors)
        qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
        qa          = qa[7:]
        qa_dot      = qa_dot[6:]

        # Target position and velocity for all joints
        qa_ref      = np.array([q[7:]]).T  # target angular positions for the motors
        qa_dot_ref  = np.array([dq[7:]]).T  # target angular velocities for the motors

        # Call controller to get torques for all joints
        jointTorques = PD(qa_ref, qa_dot_ref, qa, qa_dot, dt, Kp, Kd, torques_sat, torques_ref)
    
    
    '''
    revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    JOINTS NAMES
        0:  'FL_HAA'     1: 'FL_HFE'    2:  'FL_KFE'    3:  'FL_ANKLE'
        4:  'FR_HAA'     5: 'FR_HFE'    6:  'FR_KFE'    7:  'FR_ANKLE'
        8:  'HL_HAA'     9: 'HL_HFE'   10:  'HL_KFE'   11:  'HL_ANKLE'
    12:  'HR_HAA'    13: 'HR_HFE'   14:  'HR_KFE'   15:  'HR_ANKLE'
    model.nqs.tolist()
    >> [0, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    model.nvs.tolist()
    >> [0, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    '''
    if i==1000 : print("JUMP BEGIN !")
    if i==1000+id_Ts : print("JUMP EFFECT !")

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