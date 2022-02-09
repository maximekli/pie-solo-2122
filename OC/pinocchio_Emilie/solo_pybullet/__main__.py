# coding: utf8

#####################
#  LOADING MODULES ##
#####################

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
from controller import c


#################
#  ENVIRONMENT ##
#################

robot   = example_robot_data.load('solo12')
NQ, NV  = robot.model.nq, robot.model.nv
model   = robot.model
data    = robot.data

jointNames = robot.model.names
JOINT_IDs = range(len(jointNames))
jointMasses = [i.mass for i in model.inertias]
jointLevers = [i.lever for i in model.inertias]
jointInertias = [i.inertia for i in model.inertias]
torques_sat = 3  # torque saturation in N.m

M       = np.array(jointMasses)
m       = sum(jointMasses)
g       = model.gravity.linear

q_0     = robot.q0
v_0     = robot.v0
CoM_0   = robot.com(q_0)
v_CoM_0 = robot.vcom(q_0,v_0)

dt      = 1e-3
Ts      = 0.1
L       = 1
h       = 1

alpha   = np.arctan(4*h/L)
Vs      = np.sqrt(2*abs(g[2])*h)/np.sin(alpha)
Tt      = 2*np.sqrt(2*h/abs(g[2])) + Ts
f_x     = L*np.sqrt(abs(g[2])/(8*h))/Ts
f_z     = np.sqrt(2*abs(g[2])*h)/Ts+abs(g[2])
T       = np.arange(0, Tt, dt)
id_Ts   = T.tolist().index(Ts)


ddx_CoM = lambda t : f_x if t<Ts else 0
# ddz_CoM = lambda t : f_z if t<Ts else 0 # accélération de poussée (donc pas gravité dans cette accélération là)
ddz_CoM = lambda t : (f_z if t<Ts else 0)+g[2]

dx_CoM  = lambda t : quad(ddx_CoM, 0, t)[0]
dy_CoM  = lambda t : 0
dz_CoM  = lambda t : quad(ddz_CoM, 0, t)[0]
v_CoM   = lambda t :  np.array([dx_CoM(t),dy_CoM(t),dz_CoM(t)])
w_CoM   = lambda t : np.array([np.pi/Tt,0,0])

x_CoM  = lambda t : quad(dx_CoM, 0, t)[0]
y_CoM  = lambda t : 0
z_CoM  = lambda t : quad(dz_CoM, 0, t)[0]
X_CoM  = lambda t :  np.array([x_CoM(t),y_CoM(t),z_CoM(t)])



def flight(T=T):
    dQ  = [np.zeros(q_0.size)]
    Q   = [q_0]
    q   = q_0
    for t in T:
        if t==0: continue
        if t<=Ts:
            J       = robot.Jcom(q)
            v       = v_CoM(t)
            w       = w_CoM(t)
            Jpinv   = pinv(J)
            dq      = np.concatenate([np.array([0]), Jpinv.dot(v)])
            q       = q + dt*dq
            dQ.append(dq)
            Q.append(q)
        else:
            dQ.append(np.zeros(q_0.size))
            Q.append(q_0)
    return Q, dQ
Q, dQ = flight()

def test_flight(Q=Q, T=T, id_Ts=id_Ts):
    Q_Ts = Q[id_Ts]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)
    Q = np.array(Q)
    ax1.plot(T, Q[:,3])
    ax1.plot(Ts, Q[id_Ts,3], 'ro')
    ax1.set_xlabel('time')
    ax1.set_ylabel('joint 3: FL_HFE')
    ax1.grid(True)
    ax2.plot(T, Q[:,10])
    ax2.plot(Ts, Q[id_Ts,10], 'ro')
    ax2.set_xlabel('time')
    ax2.set_ylabel('joint 10: HL_KFE')
    ax2.grid(True)
    plt.show(block=False)
# test_flight()




################################
#  INITIALIZATION SIMULATION  ##
################################

def configure_simulation(dt, enableGUI):
    global jointTorques
    # Load the robot for Pinocchio
    solo = example_robot_data.loadSolo(False)
    solo.initDisplay(loadModel=True)

    # Start the client for PyBullet
    if enableGUI:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)  # noqa
    # p.GUI for graphical version
    # p.DIRECT for non-graphical version

    # Set gravity (disabled by default)
    p.setGravity(0, 0, g[2])

    # Load horizontal plane for PyBullet
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # Load the robot for PyBullet
    robotStartPos = [0, 0, 0.35]
    robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    p.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/robots/solo_description/robots")
    robotId = p.loadURDF("solo12.urdf", robotStartPos, robotStartOrientation)

    # Set time step of the simulation
    p.setTimeStep(dt)

    revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    torques_ref = np.zeros((len(revoluteJointIndices), 1)) # feedforward torques
    
    # Disable default motor control for revolute joints
    p.setJointMotorControlArray(robotId,
                                jointIndices=revoluteJointIndices,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocities=[0.0 for m in revoluteJointIndices],
                                forces=torques_ref)

    # Enable torque control for revolute joints
    p.setJointMotorControlArray(robotId,
                                jointIndices=revoluteJointIndices,
                                controlMode=p.TORQUE_CONTROL,
                                forces=torques_ref)

    # Compute one step of simulation for initialization
    p.stepSimulation()

    return robotId, solo, revoluteJointIndices, torques_ref


# Function to get the position/velocity of the base and the angular position/velocity of all joints
def getPosVelJoints(robotId, revoluteJointIndices):

    jointStates = p.getJointStates(robotId, revoluteJointIndices)  # State of all joints
    # print([jointStates[i_joint][0] for i_joint in range(len(jointStates))])
    baseState = p.getBasePositionAndOrientation(robotId)  # Position of the free flying base
    # print( np.array([baseState[1]]).transpose())
    baseVel = p.getBaseVelocity(robotId)  # Velocity of the free flying base

    # Reshaping data into q and qdot
    q = np.vstack((np.array([baseState[0]]).transpose(), np.array([baseState[1]]).transpose(),
                   np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).transpose()))
    qdot = np.vstack((np.array([baseVel[0]]).transpose(), np.array([baseVel[1]]).transpose(),
                      np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).transpose()))

    return q, qdot

    


# If True then we will sleep in the main loop to have a 1:1 ratio of (elapsed real time / elapsed time in the
# simulation)
realTimeSimulation                               = True  # If True then we will sleep in the main loop to have a frequency of 1/dt
enableGUI                                        = True  # enable PyBullet GUI or not
robotId, solo, revoluteJointIndices, torques_ref = configure_simulation(dt, enableGUI)


for i in range(1000+len(Q)+1000):
    # Time at the start of the loop
    if realTimeSimulation:
        t0 = time.perf_counter()
    if i == 0 :
        # Parameters for the PD controller
        Kp      = 8
        Kd      = 0.06
        dq      = np.zeros(q_0.size)
        q       = q_0
        print("\n\n\nINITIALIZATION")   
    if i == 1000 :
        # Parameters for the PD controller
        Kp      = 8
        Kd      = 0.06
        J       = robot.Jcom(q_0)
        v       = 500*v_CoM(Ts)
        w       = w_CoM(Ts)
        Jpinv   = pinv(J)
        dq      = np.concatenate([np.array([0]), Jpinv.dot(v)])
        q       = q_0 + Ts*dq
        print("BEGIN JUMP")    
    if i == 1000+len(Q) : 
        # Parameters for the PD controller
        Kp      = 8
        Kd      = 0.06
        dq      = np.zeros(q_0.size)
        q       = q_0
        print("END JUMP\n\n\n")
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
    # Get position and velocity of all joints in PyBullet (free flying base + motors)
    qa, qa_dot  = getPosVelJoints(robotId, revoluteJointIndices)
    qa          = qa[7:]
    qa_dot      = qa_dot[6:]

    # Target position and velocity for all joints
    qa_ref      = np.array([q[7:]]).T  # target angular positions for the motors
    qa_dot_ref  = np.array([dq[7:]]).T  # target angular velocities for the motors
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