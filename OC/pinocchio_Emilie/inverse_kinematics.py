import numpy as np
from numpy.linalg import norm, pinv
import pinocchio as pin
import matplotlib.pyplot as plt
from model_com import *

""" FRAMES SOLO12
    ['universe', 'root_joint', 'base_link', 'FL_HAA', 'FL_SHOULDER',
    'FL_HFE', 'FL_UPPER_LEG', 'FL_KFE', 'FL_LOWER_LEG', 'FL_ANKLE',
    'FL_FOOT', 'FR_HAA', 'FR_SHOULDER', 'FR_HFE', 'FR_UPPER_LEG',
    'FR_KFE', 'FR_LOWER_LEG', 'FR_ANKLE', 'FR_FOOT', 'HL_HAA',
    'HL_SHOULDER', 'HL_HFE', 'HL_UPPER_LEG', 'HL_KFE', 'HL_LOWER_LEG',
    'HL_ANKLE', 'HL_FOOT', 'HR_HAA', 'HR_SHOULDER', 'HR_HFE', 'HR_UPPER_LEG',
    'HR_KFE', 'HR_LOWER_LEG', 'HR_ANKLE', 'HR_FOOT']
"""
frameNames  = ['FR_FOOT', 'FL_FOOT', 'HR_FOOT', 'HL_FOOT']
FRAME_IDs   = [model.getFrameId(f) for f in frameNames]
IDX_TOOL    = model.getFrameId('base_link')


def computeJacobian(q, IDX_TOOL, FRAME_IDs):
    # J_CoM       = robot.Jcom(q).copy()
    J_CoM       = pin.computeFrameJacobian(model, data, q, IDX_TOOL).copy()
    J_Frames    = np.concatenate([robot.computeFrameJacobian(q, FRAME_ID)[:3].copy() for FRAME_ID in FRAME_IDs], axis=0)
    return np.concatenate((J_CoM, J_Frames), axis=0)

def computeError(oMgoal, oMtool, placements_ref, placements):
    # err_CoM     = CoM - CoM_ref
    err_CoM     = pin.log(oMtool.inverse()*oMgoal).vector
    err_Frames  = np.concatenate([Fi-Fi_ref for Fi, Fi_ref in zip(placements,placements_ref)], axis=0)
    return np.concatenate((err_CoM, err_Frames), axis=0)

def computeJointsConfiguration(Rot_ref, CoM_ref, q_init, K=1, q_0=robot.q0, epsilon=1e-4, DT=1e-2):
    IT_MAX      = 5000
    ## REFERENCE PLACEMENTS
    oMgoal              = pin.SE3(Rot_ref,CoM_ref)
    framePlacements_ref = [robot.framePlacement(q_0, FRAME_ID).translation.copy() for FRAME_ID in FRAME_IDs]
    ## INITIALIZATION LOOP
    i           = 0
    q           = q_init.copy()
    # q           = pin.randomConfiguration(model)
    while True :
        # Run the algorithms that outputs values in data
        pin.framesForwardKinematics(model,data,q)
        pin.computeJointJacobians(model,data,q)
        ## CURRENT CONFIGURATION
        # CoM         = robot.com(q).copy()
        # oMtool          = robot.framePlacement(q,IDX_TOOL)
        # framePlacements = [robot.framePlacement(q, FRAME_ID).translation.copy() for FRAME_ID in FRAME_IDs]
        oMtool          = data.oMf[IDX_TOOL].copy()
        framePlacements = [data.oMf[FRAME_ID].translation.copy() for FRAME_ID in FRAME_IDs]
        ## ERROR
        err         = computeError(oMgoal, oMtool, framePlacements_ref, framePlacements)
        ## JACOBIAN FROM CURRENT CONFIGURATION
        J           = computeJacobian(q, IDX_TOOL, FRAME_IDs)
        ## UPDATE CONFIGURATION
        vq          = pinv(J)@(-K*err)
        q           = pin.integrate(model,q, vq * DT)
        ## UPDATE LOOP
        i+=1
        if not i%1000: print(f"norm(err) = {norm(err)}")
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
    return q, vq

def computeTrajectory(q_0=robot.q0, T=T):
    Q = []
    vQ = []
    q = q_0.copy()
    vq = v_0.copy()
    for t in T:
        if t<=Ts:
            CoM = X_CoM(t)
            Rot = Rot_CoM(t)
            Q.append(("JUMP", q.copy()))
            vQ.append(("JUMP", vq.copy()))
            q, vq = computeJointsConfiguration(Rot, CoM, q)
        else:
            Q.append(("OHH", q_0.copy()))
            vQ.append(("OHH", v_0.copy()))
    return Q, vQ


def plot_controlledTrajectory(q_0=robot.q0, T=T, dt=dt):
    Q, vQ = computeTrajectory(q_0=q_0, T=T)
    L_X_CoM = []
    L_V_CoM = []
    prev_X_CoM = CoM_0.copy()
    for (_, q), (_, vq), t in zip(*[Q,vQ,T]):
        if t<=Ts:
            L_X_CoM.append(robot.com(q).copy())
            L_V_CoM.append(robot.vcom(q, vq).copy())
            prev_X_CoM  = L_X_CoM[-1].copy()
            prev_V_CoM  = L_V_CoM[-1].copy()
        else:
            L_X_CoM.append(prev_X_CoM + prev_V_CoM*(t-Ts) + g*(t-Ts)**2/2)
            L_V_CoM.append(prev_V_CoM + g*(t-Ts))
            
    fig1, ax1 = plt.subplots()
    L_CoM = np.array(L_X_CoM)
    ax1.plot(T,L_CoM)
    ax1.plot(T[id_Ts],L_CoM[id_Ts][0],'ro')
    ax1.plot(T[id_Ts],L_CoM[id_Ts][1],'ro')
    ax1.plot(T[id_Ts],L_CoM[id_Ts][2],'ro')
    plt.show(block=False)
    fig2, ax2 = plt.subplots()
    ax2.plot(L_CoM[:,0],L_CoM[:,2])
    plt.show(block=False)
    fig3, ax3 = plt.subplots()
    ax3.plot(T,Q)
    plt.show()