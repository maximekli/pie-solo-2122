# coding: utf8

import numpy as np  # Numpy library
import pybullet_data
import pybullet as p  # PyBullet simulator
from example_robot_data import loadSolo  # Functions to load the SOLO quadruped


def configure_simulation(dt, enableGUI):
    # Load the robot for Pinocchio
    solo = loadSolo(False) # Solo12
    solo.initDisplay(loadModel=True)

    # Start the client for PyBullet
    if enableGUI:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)  # noqa
    # p.GUI for graphical version
    # p.DIRECT for non-graphical version

    # Set gravity (disabled by default)
    p.setGravity(0, 0, solo.model.gravity.linear[2])

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