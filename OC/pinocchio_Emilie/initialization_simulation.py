# coding: utf8

import numpy as np  # Numpy library
import pybullet_data
import pybullet as p  # PyBullet simulator
from example_robot_data import loadSolo  # Functions to load the SOLO quadruped
from model_com import g


def configure_simulation(dt, enableGUI):
    global jointTorques
    # Load the robot for Pinocchio
    solo = loadSolo(False)
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

    
# p.ACTIVATION_STATE_DISABLE_SLEEPING            p.GRAPHICS_SERVER
# p.ACTIVATION_STATE_DISABLE_WAKEUP              p.GRAPHICS_SERVER_MAIN_THREAD
# p.ACTIVATION_STATE_ENABLE_SLEEPING             p.GRAPHICS_SERVER_TCP
# p.ACTIVATION_STATE_ENABLE_WAKEUP               p.GUI
# p.ACTIVATION_STATE_SLEEP                       p.GUI_MAIN_THREAD
# p.ACTIVATION_STATE_WAKE_UP                     p.GUI_SERVER
# p.AddFileIOAction                              p.IK_DLS
# p.addUserData(                                 p.IK_HAS_JOINT_DAMPING
# p.addUserDebugLine(                            p.IK_HAS_NULL_SPACE_VELOCITY
# p.addUserDebugParameter(                       p.IK_HAS_TARGET_ORIENTATION
# p.addUserDebugText(                            p.IK_HAS_TARGET_POSITION
# p.applyExternalForce(                          p.IK_SDLS
# p.applyExternalTorque(                         p.invertTransform(
# p.B3G_ALT                                      p.isConnected(
# p.B3G_BACKSPACE                                p.isNumpyEnabled(
# p.B3G_CONTROL                                  p.JOINT_FEEDBACK_IN_JOINT_FRAME
# p.B3G_DELETE                                   p.JOINT_FEEDBACK_IN_WORLD_SPACE
# p.B3G_DOWN_ARROW                               p.JOINT_FIXED
# p.B3G_END                                      p.JOINT_GEAR
# p.B3G_F1                                       p.JOINT_PLANAR
# p.B3G_F10                                      p.JOINT_POINT2POINT
# p.B3G_F11                                      p.JOINT_PRISMATIC
# p.B3G_F12                                      p.JOINT_REVOLUTE
# p.B3G_F13                                      p.JOINT_SPHERICAL
# p.B3G_F14                                      p.KEY_IS_DOWN
# p.B3G_F15                                      p.KEY_WAS_RELEASED
# p.B3G_F2                                       p.KEY_WAS_TRIGGERED
# p.B3G_F3                                       p.LINK_FRAME
# p.B3G_F4                                       p.loadBullet(
# p.B3G_F5                                       p.loadMJCF(
# p.B3G_F6                                       p.loadPlugin(
# p.B3G_F7                                       p.loadSDF(
# p.B3G_F8                                       p.loadSoftBody(
# p.B3G_F9                                       p.loadTexture(
# p.B3G_HOME                                     p.loadURDF(
# p.B3G_INSERT                                   p.MAX_RAY_INTERSECTION_BATCH_SIZE
# p.B3G_LEFT_ARROW                               p.MESH_DATA_SIMULATION_MESH
# p.B3G_PAGE_DOWN                                p.MJCF_COLORS_FROM_FILE
# p.B3G_PAGE_UP                                  p.multiplyTransforms(
# p.B3G_RETURN                                   p.PD_CONTROL
# p.B3G_RIGHT_ARROW                              p.performCollisionDetection(
# p.B3G_SHIFT                                    p.POSITION_CONTROL
# p.B3G_SPACE                                    p.PosixFileIO
# p.B3G_UP_ARROW                                 p.rayTest(
# p.calculateInverseDynamics(                    p.rayTestBatch(
# p.calculateInverseKinematics(                  p.readUserDebugParameter(
# p.calculateInverseKinematics2(                 p.removeAllUserDebugItems(
# p.calculateJacobian(                           p.removeAllUserParameters(
# p.calculateMassMatrix(                         p.removeBody(
# p.calculateVelocityQuaternion(                 p.removeCollisionShape(
# p.changeConstraint(                            p.removeConstraint(
# p.changeDynamics(                              p.RemoveFileIOAction
# p.changeTexture(                               p.removeState(
# p.changeVisualShape(                           p.removeUserData(
# p.CNSFileIO                                    p.removeUserDebugItem(
# p.computeDofCount(                             p.renderImage(
# p.computeProjectionMatrix(                     p.resetBasePositionAndOrientation(
# p.computeProjectionMatrixFOV(                  p.resetBaseVelocity(
# p.computeViewMatrix(                           p.resetDebugVisualizerCamera(
# p.computeViewMatrixFromYawPitchRoll(           p.resetJointState(
# p.configureDebugVisualizer(                    p.resetJointStateMultiDof(
# p.connect(                                     p.resetJointStatesMultiDof(
# p.CONSTRAINT_SOLVER_LCP_DANTZIG                p.resetMeshData(
# p.CONSTRAINT_SOLVER_LCP_PGS                    p.resetSimulation(
# p.CONSTRAINT_SOLVER_LCP_SI                     p.RESET_USE_DEFORMABLE_WORLD
# p.CONTACT_RECOMPUTE_CLOSEST                    p.RESET_USE_DISCRETE_DYNAMICS_WORLD
# p.CONTACT_REPORT_EXISTING                      p.RESET_USE_SIMPLE_BROADPHASE
# p.COV_ENABLE_DEPTH_BUFFER_PREVIEW              p.resetVisualShapeData(
# p.COV_ENABLE_GUI                               p.restoreState(
# p.COV_ENABLE_KEYBOARD_SHORTCUTS                p.rotateVector(
# p.COV_ENABLE_MOUSE_PICKING                     p.saveBullet(
# p.COV_ENABLE_PLANAR_REFLECTION                 p.saveState(
# p.COV_ENABLE_RENDERING                         p.saveWorld(
# p.COV_ENABLE_RGB_BUFFER_PREVIEW                p.SENSOR_FORCE_TORQUE
# p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW         p.setAdditionalSearchPath(
# p.COV_ENABLE_SHADOWS                           p.setCollisionFilterGroupMask(
# p.COV_ENABLE_SINGLE_STEP_RENDERING             p.setCollisionFilterPair(
# p.COV_ENABLE_TINY_RENDERER                     p.setDebugObjectColor(
# p.COV_ENABLE_VR_PICKING                        p.setDefaultContactERP(
# p.COV_ENABLE_VR_RENDER_CONTROLLERS             p.setGravity(
# p.COV_ENABLE_VR_TELEPORTING                    p.setInternalSimFlags(
# p.COV_ENABLE_WIREFRAME                         p.setJointMotorControl(
# p.COV_ENABLE_Y_AXIS_UP                         p.setJointMotorControl2(
# p.createCollisionShape(                        p.setJointMotorControlArray(
# p.createCollisionShapeArray(                   p.setJointMotorControlMultiDof(
# p.createConstraint(                            p.setJointMotorControlMultiDofArray(
# p.createMultiBody(                             p.setPhysicsEngineParameter(
# p.createSoftBodyAnchor(                        p.setRealTimeSimulation(
# p.createVisualShape(                           p.setTimeOut(
# p.createVisualShapeArray(                      p.setTimeStep(
# p.DIRECT                                       p.setVRCameraState(
# p.disconnect(                                  p.SHARED_MEMORY
# p.enableJointForceTorqueSensor(                p.SHARED_MEMORY_GUI
# p.ER_BULLET_HARDWARE_OPENGL                    p.SHARED_MEMORY_KEY
# p.ER_NO_SEGMENTATION_MASK                      p.SHARED_MEMORY_KEY2
# p.error(                                       p.SHARED_MEMORY_SERVER
# p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX    p.STABLE_PD_CONTROL
# p.ER_TINY_RENDERER                             p.startStateLogging(
# p.ER_USE_PROJECTIVE_TEXTURE                    p.STATE_LOGGING_ALL_COMMANDS
# p.executePluginCommand(                        p.STATE_LOGGING_CONTACT_POINTS
# p.GEOM_BOX                                     p.STATE_LOGGING_CUSTOM_TIMER
# p.GEOM_CAPSULE                                 p.STATE_LOGGING_GENERIC_ROBOT
# p.GEOM_CONCAVE_INTERNAL_EDGE                   p.STATE_LOGGING_MINITAUR
# p.GEOM_CYLINDER                                p.STATE_LOGGING_PROFILE_TIMINGS
# p.GEOM_FORCE_CONCAVE_TRIMESH                   p.STATE_LOGGING_VIDEO_MP4
# p.GEOM_HEIGHTFIELD                             p.STATE_LOGGING_VR_CONTROLLERS
# p.GEOM_MESH                                    p.STATE_LOG_JOINT_MOTOR_TORQUES
# p.GEOM_PLANE                                   p.STATE_LOG_JOINT_TORQUES
# p.GEOM_SPHERE                                  p.STATE_LOG_JOINT_USER_TORQUES
# p.getAABB(                                     p.STATE_REPLAY_ALL_COMMANDS
# p.getAPIVersion(                               p.stepSimulation(
# p.getAxisAngleFromQuaternion(                  p.stopStateLogging(
# p.getAxisDifferenceQuaternion(                 p.submitProfileTiming(
# p.getBasePositionAndOrientation(               p.syncBodyInfo(
# p.getBaseVelocity(                             p.syncUserData(
# p.getBodyInfo(                                 p.TCP
# p.getBodyUniqueId(                             p.TORQUE_CONTROL
# p.getCameraImage(                              p.UDP
# p.getClosestPoints(                            p.unloadPlugin(
# p.getCollisionShapeData(                       p.unsupportedChangeScaling(
# p.getConnectionInfo(                           p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
# p.getConstraintInfo(                           p.URDF_ENABLE_SLEEPING
# p.getConstraintState(                          p.URDF_ENABLE_WAKEUP
# p.getConstraintUniqueId(                       p.URDF_GLOBAL_VELOCITIES_MB
# p.getContactPoints(                            p.URDF_GOOGLEY_UNDEFINED_COLORS
# p.getDebugVisualizerCamera(                    p.URDF_IGNORE_COLLISION_SHAPES
# p.getDifferenceQuaternion(                     p.URDF_IGNORE_VISUAL_SHAPES
# p.getDynamicsInfo(                             p.URDF_INITIALIZE_SAT_FEATURES
# p.getEulerFromQuaternion(                      p.URDF_MAINTAIN_LINK_ORDER
# p.getJointInfo(                                p.URDF_MERGE_FIXED_LINKS
# p.getJointState(                               p.URDF_PRINT_URDF_INFO
# p.getJointStateMultiDof(                       p.URDF_USE_IMPLICIT_CYLINDER
# p.getJointStates(                              p.URDF_USE_INERTIA_FROM_FILE
# p.getJointStatesMultiDof(                      p.URDF_USE_MATERIAL_COLORS_FROM_MTL
# p.getKeyboardEvents(                           p.URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL
# p.getLinkState(                                p.URDF_USE_SELF_COLLISION
# p.getLinkStates(                               p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
# p.getMatrixFromQuaternion(                     p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
# p.getMeshData(                                 p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
# p.getMouseEvents(                              p.VELOCITY_CONTROL
# p.getNumBodies(                                p.vhacd(
# p.getNumConstraints(                           p.VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS
# p.getNumJoints(                                p.VISUAL_SHAPE_DOUBLE_SIDED
# p.getNumUserData(                              p.VR_BUTTON_IS_DOWN
# p.getOverlappingObjects(                       p.VR_BUTTON_WAS_RELEASED
# p.getPhysicsEngineParameters(                  p.VR_BUTTON_WAS_TRIGGERED
# p.getQuaternionFromAxisAngle(                  p.VR_CAMERA_TRACK_OBJECT_ORIENTATION
# p.getQuaternionFromEuler(                      p.VR_DEVICE_CONTROLLER
# p.getQuaternionSlerp(                          p.VR_DEVICE_GENERIC_TRACKER
# p.getUserData(                                 p.VR_DEVICE_HMD
# p.getUserDataId(                               p.VR_MAX_BUTTONS
# p.getUserDataInfo(                             p.VR_MAX_CONTROLLERS
# p.getVisualShapeData(                          p.WORLD_FRAME
# p.getVREvents(                                 p.ZipFileIO
# p.GRAPHICS_CLIENT                              