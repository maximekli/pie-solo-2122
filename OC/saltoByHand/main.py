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

robot   = example_robot_data.load('solo12')


# Initial position
# The same as in inverse_kinematics.py
q0      = robot.q0.copy()
q0[7:13]= q0[13:]
# Simulation dt
dt      = 1e-3

# Force symmetry for configurations: no change in the simulation
SYMMETRIC = False

####################################
#  LOAD CONFIGURATION TRAJECTORY  ##
####################################
try:
    Q = np.load('trajectory_npy/ik_Q.npy', allow_pickle=True)
    vQ = np.load('trajectory_npy/ik_vQ.npy', allow_pickle=True)
except OSError:
    import inverse_kinematics
    Q = np.load('trajectory_npy/ik_Q.npy', allow_pickle=True)
    vQ = np.load('trajectory_npy/ik_vQ.npy', allow_pickle=True)
Q   = Q[:-4]
vQ  = vQ[:-4]

try:
    torques = np.load('trajectory_npy/salto_torques.npy', allow_pickle=True)
    isTorquesRef = True # can do feed-forward (not successfully implemented)
except OSError:
    isTorquesRef = False
    
salto_Q         = []
salto_vQ        = []
salto_torques   = []

################################
#  INITIALIZATION SIMULATION  ##
################################
    
# If True then we will sleep in the main loop to have a 1:1 ratio of (elapsed real time / elapsed time in the
# simulation)
realTimeSimulation                                      = True  # If True then we will sleep in the main loop to have a frequency of 1/dt
enableGUI                                               = True  # enable PyBullet GUI or not
robotId, solo, revoluteJointIndices, zero_torques_ref   = configure_simulation(dt, enableGUI)
torques_sat                                             = 3 # N.m

print("Click on the link and press any key to begin simulation")
input()

# Parameters for the PD controller
Kp  = 0 if isTorquesRef else 8
Kd  = 0 if isTorquesRef else 0.06

for i in range(len(Q)+2000):
    # Time at the start of the loop
    if realTimeSimulation:
        t0 = time.perf_counter()
    
    # Get position and velocity of all joints in PyBullet (free flying base + motors)
    qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
    qa          = qa[7:]
    qa_dot      = qa_dot[6:]

    # Joints configuration
    if i < len(Q) :
        ''' JUMP '''
        vq          = vQ[i]
        q           = Q[i]
        torques_ref = torques[i] if isTorquesRef else zero_torques_ref
    elif i < len(Q) + 500 :
        ''' STAY IN POSITION '''
        vq          = np.zeros(vq.shape)
        q           = q
        torques_ref = torques[i] if isTorquesRef else zero_torques_ref
    else : 
        ''' GO BACK TO INITIALIZED POSITION '''
        # Parameters for the PD controller
        Kp          = 0 if isTorquesRef else 3
        Kd          = 0 if isTorquesRef else 0.3
        vq          = np.zeros(vq.shape)
        q           = q0
        torques_ref = torques[i] if isTorquesRef else zero_torques_ref

    # Target position and velocity for all joints
    qa_ref      = np.array([q[7:]]).T  # target angular positions for the motors
    qa_dot_ref  = np.array([vq[6:]]).T  # target angular velocities for the motors

    if SYMMETRIC :
        # Force symmetry: no change in the simulation
        '''
        qa[0:6,0]   -> gauche
        qa[6:12,0]  -> droite
        '''
        # qa_ref[0:6]         = qa_ref[6:12]
        # qa_dot_ref[0:6]     = qa_dot_ref[6:12]
        qa_ref[6:12]        = qa_ref[0:6]
        qa_dot_ref[6:12]    = qa_dot_ref[0:6]

    # Call controller to get torques for all joints
    jointTorques = PD(qa_ref, qa_dot_ref, qa, qa_dot, dt, Kp, Kd, torques_sat, torques_ref)
    
    # Store control parameters
    salto_vQ.append(vq)
    salto_Q.append(q)
    salto_torques.append(jointTorques)

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

np.save('trajectory_npy/salto_Q.npy', salto_Q, allow_pickle=True)
np.save('trajectory_npy/salto_vQ.npy', salto_vQ, allow_pickle=True)
if not isTorquesRef : np.save('trajectory_npy/salto_torques.npy', salto_torques, allow_pickle=True)