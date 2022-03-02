# coding: utf8

#####################
#  LOADING MODULES ##
#####################

import time
import numpy as np
import pybullet as p  # PyBullet simulator

from PD import PD
from initialization_simulation import configure_simulation, getPosVelJoints # Functions to initialize the simulation and retrieve joints positions/velocities
from inverse_kinematics import *



################################
#  INITIALIZATION SIMULATION  ##
################################

# Q, vQ = computeTrajectory()
    
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
        vq      = np.zeros(q_0.size)
        q       = q_0

        # Get position and velocity of all joints in PyBullet (free flying base + motors)
        qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
        qa          = qa[7:]
        qa_dot      = qa_dot[6:]

        # Target position and velocity for all joints
        qa_ref      = np.array([q[7:]]).T  # target angular positions for the motors
        qa_dot_ref  = np.array([vq[7:]]).T  # target angular velocities for the motors

        # Call controller to get torques for all joints
        jointTorques = PD(qa_ref, qa_dot_ref, qa, qa_dot, dt, Kp, Kd, torques_sat, torques_ref)
        
    # ''' JUMP TRAJECTORY '''
    elif (1000 <= i) and (i < 1000+len(T)) :
        # Parameters for the PD controller
        # Kp      = 8
        # Kd      = 0.06

        # # Joints configuration
        # vq  = vQ[i-1000]
        # q  = Q[i-1000]
        
        # # Get position and velocity of all joints in PyBullet (free flying base + motors)
        # qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
        # qa          = qa[7:]
        # qa_dot      = qa_dot[6:]

        # # Target position and velocity for all joints
        # qa_ref      = np.array([q[7:]]).T  # target angular positions for the motors
        # qa_dot_ref  = np.array([vq[6:]]).T  # target angular velocities for the motors

        # # Call controller to get torques for all joints
        # jointTorques = PD(qa_ref, qa_dot_ref, qa, qa_dot, dt, Kp, Kd, torques_sat, torques_ref)
        
        qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
        qa_dot_dot = 0*qa_dot if i==1000 else pinv(M) @ (tauq - b).T
        Xs = [robot.framePlacement(qa,id).translation for id in FRAME_IDs]
        Fb = np.array([f_x, 0, f_z])
        Mb = model.inertias[1].inertia@np.array([0,2*np.pi/Tt,0])
        D1 = np.concatenate([np.eye(3)]*4, axis=1)
        D2 = np.concatenate([np.array([[0,Xi[2],-Xi[1]],[Xi[2],0,-Xi[0]],[Xi[1],0,-Xi[0]]]) for Xi in Xs], axis=1)
        D = np.concatenate((D1,D2), axis=0)
        Fe = pinv(D)@np.concatenate((Fb,Mb), axis=0)

        M = pin.crba(model, data, qa)
        b = pin.rnea(model, data, qa, qa_dot, np.zeros(robot.model.nv))

        L_J = computeJacobians(qa, IDX_TOOL, FRAME_IDs)
        J = np.concatenate(L_J[0:-1], axis=0)
        tauq = (M@qa_dot_dot).T + b -(J.T@Fe)
        jointTorques = tauq[0,6:]

    # ''' GO BACK TO INITIALIZED POSITION '''
    else : 
        # Parameters for the PD controller
        Kp      = 8
        Kd      = 0.06

        # Joints configuration
        vq      = np.zeros(q_0.size)
        q       = q_0

        # Get position and velocity of all joints in PyBullet (free flying base + motors)
        qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
        qa          = qa[7:]
        qa_dot      = qa_dot[6:]

        # Target position and velocity for all joints
        qa_ref      = np.array([q[7:]]).T  # target angular positions for the motors
        qa_dot_ref  = np.array([vq[7:]]).T  # target angular velocities for the motors

        # Call controller to get torques for all joints
        jointTorques = PD(qa_ref, qa_dot_ref, qa, qa_dot, dt, Kp, Kd, torques_sat, torques_ref)

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