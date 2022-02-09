
import numpy as np
import pinocchio as pin
import example_robot_data

robot   = example_robot_data.load('solo12')
NQ, NV  = robot.model.nq, robot.model.nv
model   = robot.model
data    = robot.data

def update_state(q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

def get_state(q):
    CoM             = robot.com(q)
    JCoM            = robot.Jcom(q)
    J               = robot.computeJointJacobians(q)
    jointPlacements = [robot.placement(q,JOINT_ID) for JOINT_ID in JOINT_IDs]
    jointCoMs       = np.array([p.translation for p in jointPlacements])
    jointRots       = np.array([p.rotation for p in jointPlacements])
    return locals()

def test_CoM():
    q = pin.randomConfiguration(model)
    dict = get_state(q)
    print(M.dot(dict['jointCoMs'])/m)
    print(dict['CoM'])
    print(dict['jointCoMs'][1])

''' JOINTS NAMES
    0:  'universe'   1: 'root_joint' 
    2:  'FL_HAA'     3: 'FL_HFE'    4:  'FL_KFE' 
    5:  'FR_HAA'     6: 'FR_HFE'    7:  'FR_KFE' 
    8:  'HL_HAA'     9: 'HL_HFE'   10:  'HL_KFE' 
   11:  'HR_HAA'    12: 'HR_HFE'   13:  'HR_KFE'
'''
''' FRAMES NAMES
joint 0 <- 0: 'universe'
joint 1 <- 1: 'root_joint' <- 2: 'base_link'

épaule joint 2 <- 3: 'FL_HAA' <- 4: 'FL_SHOULDER'     c'est la cheville
épaule joint 3 <- 5: 'FL_HFE' <- 6: 'FL_UPPER_LEG'       VVVVVVVVVV
coude  joint 4 <- 7: 'FL_KFE' <- 8: 'FL_LOWER_LEG' <- 9: 'FL_ANKLE' <- 10: 'FL_FOOT'

   11:  'FR_HAA'    12: 'FR_SHOULDER'  13:  'FR_HFE'    14: 'FR_UPPER_LEG'
   15:  'FR_KFE'    16: 'FR_LOWER_LEG' 17:  'FR_ANKLE'  18: 'FR_FOOT'
   
   19:  'HL_HAA'    20: 'HL_SHOULDER'  21:  'HL_HFE'    22: 'HL_UPPER_LEG'
   23:  'HL_KFE'    24: 'HL_LOWER_LEG' 25:  'HL_ANKLE'  26: 'HL_FOOT'
   
   27:  'HR_HAA'    28: 'HR_SHOULDER'  29:  'HR_HFE'    30: 'HR_UPPER_LEG'
   31:  'HR_KFE'    32: 'HR_LOWER_LEG' 33:  'HR_ANKLE'  34: 'HR_FOOT'
'''

def test_position_joints():
    q = pin.randomConfiguration(model)
    dict = get_state(q)
    # print(dict['jointRots'])
    print("###################")
    FL_HAA = dict['jointCoMs'][2]
    FL_HFE = dict['jointCoMs'][3]
    FL_KFE = dict['jointCoMs'][4]






# robot.getJointJacobian(JOINT_ID)
# robot.jointJacobian(q,JOINT_ID)
# robot.computeFrameJacobian(q,FRAME_ID)
# robot.computeJointJacobian(q,JOINT_ID)
# robot.computeJointJacobians(q)

# robot.centroidal(q,v)
# robot.centroidalMap(q)
# robot.centroidalMomentum(q,v)
# robot.centroidalMomentumVariation(q,v,a)

# robot.model
# robot.data

# robot.com()
# robot.vcom(q,v)
# robot.acom(q,v,a)
# robot.Jcom(q)

# robot.index(JOINT_NAME)
# robot.nq
# robot.nv

''' tau = M(q)ddq + C(q,dq) + g(q) '''
# robot.gravity(q)
# robot.mass(q)

# robot.q0
# robot.v0
# robot.velocity(q,v,JOINT_ID)
# robot.classicalAcceleration(q,v,a,JOINT_ID)
# robot.acceleration(q,v,a,JOINT_ID)
# robot.nle(q,v)


# robot.forwardKinematics(q)
# robot.placement(q,JOINT_ID)
# robot.updateGeometryPlacements()


# robot.frameAcceleration(q,v,a,FRAME_ID)
# robot.frameClassicAcceleration(FRAME_ID)
# robot.frameClassicalAcceleration(q,v,a,FRAME_ID)
# robot.frameJacobian(q,FRAME_ID)
# robot.framePlacement(q,FRAME_ID)
# robot.frameVelocity(q,v,FRAME_ID)
# robot.framesForwardKinematics(q)
# robot.getFrameJacobian(FRAME_ID)