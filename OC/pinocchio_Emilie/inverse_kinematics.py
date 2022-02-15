from re import I
from termios import FF1
import numpy as np
from numpy.linalg import norm, solve, pinv
from scipy.integrate import quad
import pinocchio as pin
import matplotlib.pyplot as plt
from model_com import *

def computeJointsConfiguration(Rot_ref, CoM_ref, q_init, K=1, q_0=robot.q0, epsilon=1e-4, DT=1e-3):
    IDX_TOOL    = robot.model.getFrameId('base_link')
    """
        ['universe', 'root_joint', 'base_link', 'FL_HAA', 'FL_SHOULDER',
        'FL_HFE', 'FL_UPPER_LEG', 'FL_KFE', 'FL_LOWER_LEG', 'FL_ANKLE',
        'FL_FOOT', 'FR_HAA', 'FR_SHOULDER', 'FR_HFE', 'FR_UPPER_LEG',
        'FR_KFE', 'FR_LOWER_LEG', 'FR_ANKLE', 'FR_FOOT', 'HL_HAA',
        'HL_SHOULDER', 'HL_HFE', 'HL_UPPER_LEG', 'HL_KFE', 'HL_LOWER_LEG',
        'HL_ANKLE', 'HL_FOOT', 'HR_HAA', 'HR_SHOULDER', 'HR_HFE', 'HR_UPPER_LEG',
        'HR_KFE', 'HR_LOWER_LEG', 'HR_ANKLE', 'HR_FOOT']
    """
    IT_MAX      = 1000*5
    i           = 0
    # q           = q_init.copy()
    # dq          = np.ones(q_0.shape)
    q           = pin.randomConfiguration(robot.model)
    dq          = np.random.rand(robot.model.nv)*2-1
    
    ## REFERENCE FOOTS PLACEMENTS
    oMgoal      = pin.SE3(Rot_ref,CoM_ref)
    F1_ref      = robot.framePlacement(q_0, robot.model.getFrameId('FR_FOOT')).translation.copy()
    F2_ref      = robot.framePlacement(q_0, robot.model.getFrameId('HR_FOOT')).translation.copy()
    F3_ref      = robot.framePlacement(q_0, robot.model.getFrameId('FL_FOOT')).translation.copy()
    F4_ref      = robot.framePlacement(q_0, robot.model.getFrameId('HL_FOOT')).translation.copy()
    while True :
        # Run the algorithms that outputs values in robot.data
        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)
        
        ## CURRENT CONFIGURATION
        # CoM         = robot.com(q).copy()
        oMtool      = robot.framePlacement(q,IDX_TOOL)
        F1          = robot.framePlacement(q, robot.model.getFrameId('FR_FOOT')).translation.copy()
        F2          = robot.framePlacement(q, robot.model.getFrameId('HR_FOOT')).translation.copy()
        F3          = robot.framePlacement(q, robot.model.getFrameId('FL_FOOT')).translation.copy()
        F4          = robot.framePlacement(q, robot.model.getFrameId('HL_FOOT')).translation.copy()

        ## ERROR
        # err_CoM     = CoM - CoM_ref
        err_CoM     = pin.log(oMtool.inverse()*oMgoal).vector
        err_F1      = F1 - F1_ref
        err_F2      = F2 - F2_ref
        err_F3      = F3 - F3_ref
        err_F4      = F4 - F4_ref
        err         = np.concatenate((err_CoM, err_F1, err_F2, err_F3, err_F4), axis=0)

        ## JACOBIAN FROM CURRENT CONFIGURATION
        # J_CoM       = robot.Jcom(q).copy()
        J_CoM       = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_TOOL)
        J_F1        = robot.computeFrameJacobian(q, robot.model.getFrameId('FR_FOOT'))[:3].copy()
        J_F2        = robot.computeFrameJacobian(q, robot.model.getFrameId('HR_FOOT'))[:3].copy()
        J_F3        = robot.computeFrameJacobian(q, robot.model.getFrameId('FL_FOOT'))[:3].copy()
        J_F4        = robot.computeFrameJacobian(q, robot.model.getFrameId('HL_FOOT'))[:3].copy()
        # J_Theta     = TODO
        J           = np.concatenate((J_CoM, J_F1, J_F2, J_F3, J_F4), axis=0)

        ## UPDATE CONFIGURATION
        vq          = pinv(J)@err
        q           = pin.integrate(robot.model,q, vq * DT)
        
        i+=1
        if not i%100: print(f"norm(err) = {norm(err)}")
        if norm(err) < epsilon:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
    if success:
        print("Convergence achieved!")
    else:
        print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
    return q

def computeTrajectory(q_0=robot.q0, T=T):
    Q = []
    q = q_0.copy()
    for t in T:
        if t<=Ts:
            CoM = 0.5*X_CoM(t)
            Rot = Rot_CoM(t)
            Q.append(("JUMP", q.copy()))
            q = computeJointsConfiguration(Rot, CoM, q)
        else:
            Q.append(("OHH", q_0.copy()))
    return Q