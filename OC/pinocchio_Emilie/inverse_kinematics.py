from re import I
from termios import FF1
import numpy as np
from numpy.linalg import norm, solve, pinv
from scipy.integrate import quad
import pinocchio as pin
import matplotlib.pyplot as plt
from model_com import *

def computeJointsConfiguration(Rot_ref, CoM_ref, q_init, K=1, q_0=robot.q0, epsilon=1e-4, DT=1e-2):
    IDX_TOOL    = robot.model.getFrameId('base_link')
    IT_MAX      = 1000
    i           = 0
    q           = q_init.copy()
    dq          = np.ones(q_0.shape)
    
    ## REFERENCE FOOTS PLACEMENTS
    # F1_ref      = robot.framePlacement(q_0, robot.model.getFrameId('FR_FOOT')).translation.copy()
    # F2_ref      = robot.framePlacement(q_0, robot.model.getFrameId('HR_FOOT')).translation.copy()
    # F3_ref      = robot.framePlacement(q_0, robot.model.getFrameId('FL_FOOT')).translation.copy()
    # F4_ref      = robot.framePlacement(q_0, robot.model.getFrameId('HL_FOOT')).translation.copy()
    # # Theta_ref   = TODO
    oMgoal = pin.SE3(Rot_ref,CoM_ref)
    print(Rot_ref)
    while True :
        # Run the algorithms that outputs values in robot.data
        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)
        
        ## CURRENT CONFIGURATION
        # CoM         = robot.com(q).copy()
        # F1          = robot.framePlacement(q, robot.model.getFrameId('FR_FOOT')).translation.copy()
        # F2          = robot.framePlacement(q, robot.model.getFrameId('HR_FOOT')).translation.copy()
        # F3          = robot.framePlacement(q, robot.model.getFrameId('FL_FOOT')).translation.copy()
        # F4          = robot.framePlacement(q, robot.model.getFrameId('HL_FOOT')).translation.copy()
        # # Theta       = TODO
        oMtool = robot.data.oMf[IDX_TOOL]

        ## ERROR
        # err_CoM     = CoM - CoM_ref
        # err_F1      = F1 - F1_ref
        # err_F2      = F2 - F2_ref
        # err_F3      = F3 - F3_ref
        # err_F4      = F4 - F4_ref
        # # err_Theta   = TODO
        # err         = np.concatenate((err_CoM, err_F1, err_F2, err_F3, err_F4), axis=0)

        ## JACOBIAN FROM CURRENT CONFIGURATION
        # J_CoM       = robot.Jcom(q).copy()
        # J_F1        = robot.computeFrameJacobian(q, robot.model.getFrameId('FR_FOOT'))[:3].copy()
        # J_F2        = robot.computeFrameJacobian(q, robot.model.getFrameId('HR_FOOT'))[:3].copy()
        # J_F3        = robot.computeFrameJacobian(q, robot.model.getFrameId('FL_FOOT'))[:3].copy()
        # J_F4        = robot.computeFrameJacobian(q, robot.model.getFrameId('HL_FOOT'))[:3].copy()
        # # J_Theta     = TODO
        # J           = np.concatenate((J_CoM, J_F1, J_F2, J_F3, J_F4), axis=0)

        # o_Jtool3 = pin.computeFrameJacobian(robot.model,robot.data,q,IDX_TOOL,pin.LOCAL_WORLD_ALIGNED)[:3,:]
        # err = oMtool.translation-oMgoal.translation
        # vq = -pinv(o_Jtool3)@err
        err = pin.log(oMtool.inverse()*oMgoal).vector
        tool_Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_TOOL)
        vq = pinv(tool_Jtool)@err
        q = pin.integrate(robot.model,q, vq * DT)

        # Jpinv       = pinv(J)
        # dq          = Jpinv.dot(-K*err)
        # q           = q = pin.integrate(model,q,dq)
        
        i+=1
        # if not i%100: print(f"norm(err)={norm(err)}")
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
        if t<Ts:
            Q.append(q.copy())
            q = computeJointsConfiguration(Rot_CoM(t), X_CoM(t), q)
        else:
            Q.append(q_0.copy())
    return Q