import numpy as np
from numpy.linalg import norm, pinv
from scipy.optimize import fmin_bfgs
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



def computeJacobians(q, IDX_TOOL, FRAME_IDs):
    # J_CoM       = robot.Jcom(q).copy()
    J_CoM       = robot.computeFrameJacobian(q, IDX_TOOL).copy()
    J_Frames    = [robot.computeFrameJacobian(q, FRAME_ID)[:3].copy() for FRAME_ID in FRAME_IDs]
    return J_Frames + [J_CoM]
def computeErrors(oMgoal, oMtool, placements_ref, placements):
    # err_CoM     = CoM - CoM_ref
    err_CoM     = pin.log(oMtool.inverse()*oMgoal).vector
    err_Frames  = [Fi-Fi_ref for Fi, Fi_ref in zip(placements,placements_ref)]
    return err_Frames + [err_CoM]
def stepJacobianIK(q, oMgoal, framePlacements_ref, K, DT):
    ## Run the algorithms that outputs values in data
    robot.framesForwardKinematics(q)
    robot.computeJointJacobians(q)
    ## CURRENT CONFIGURATION
    oMtool          = pin.SE3(computeRot(q),computeCoM(q))
    framePlacements = [data.oMf[FRAME_ID].translation.copy() for FRAME_ID in FRAME_IDs]
    ## ERROR
    L_err           = computeErrors(oMgoal, oMtool, framePlacements_ref, framePlacements)
    ## JACOBIAN FROM CURRENT CONFIGURATION
    L_J             = computeJacobians(q, IDX_TOOL, FRAME_IDs)
    ## UPDATE CONFIGURATION
    for k, err in enumerate(L_err):
        Ptool = np.eye(robot.nv) if k==0 else Ptool-pinv(L_J[k-1] @ Ptool) @ L_J[k-1] @ Ptool
        vq =  pinv(L_J[k]) @ (-K*err) if k==0 else vq+pinv(L_J[k] @ Ptool) @ (-K*err - L_J[k] @ vq)
    q = pin.integrate(model,q, vq * DT)
    return q, vq

def JacobianIK(Rot_ref, CoM_ref, q_init, K=-1, q_0=robot.q0, epsilon=1e-3, DT=dt, IT_MAX = 10000):
    ## REFERENCE PLACEMENTS
    oMgoal              = pin.SE3(Rot_ref,CoM_ref)
    framePlacements_ref = [robot.framePlacement(q_0, FRAME_ID).translation.copy() for FRAME_ID in FRAME_IDs]
    ## INITIALIZATION LOOP
    q       = q_init.copy()
    herr    = []
    while True :
        ## IK STEP
        # q, vq = stepJacobianIK(q, oMgoal, framePlacements_ref, K, DT)
        vq = pinv(robot.Jcom(q))@(computeCoM(q)-CoM_ref)
        q = pin.integrate(model,q, -10*K*vq)
        ## UPDATE LOOP
        if herr : K = -K if herr[-1]-norm(vq)< 0 else K
        herr.append(norm(vq))
        if not len(herr)%1000: print(f"norm(vq) = {norm(vq)}")
        if (norm(vq) < epsilon)or(len(herr) >= IT_MAX): break
    if len(herr) < IT_MAX:
        print(f"Convergence achieved in {len(herr)} turns with norm(err)={norm(computeCoM(q)-CoM_ref)}!")
    else:
        plt.title("Errors")
        plt.ylabel("err")
        plt.xlabel("iterations")
        plt.plot(herr)
        plt.show()
        print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
    return q, vq









'''
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
# 7+(0 mod 3) -> shoulders (not on x)
# 7+1 -> FL_shoulder (pi/2 -> bras vers l'arrière)
# 7+2 -> FL_knee (pi/2 -> avant bras vers l'arrière)
# 7+4 -> FR_shoulder (pi/2 -> bras vers l'arrière)
# 7+5 -> FR_knee (pi/2 -> avant bras vers l'arrière)
# 7+7 -> HL_shoulder (pi/2 -> bras vers l'arrière)
# 7+8 -> HL_knee (pi/2 -> avant bras vers l'arrière)
# 7+10 -> HR_shoulder (pi/2 -> bras vers l'arrière)
# 7+11 -> HR_knee (pi/2 -> avant bras vers l'arrière)

down    = lambda t :np.diag([1]*7   +   [0  ,1+t/Ts ,1+t/Ts ]*2 +   [0  ,1+0.4*t/Ts ,1+0.4*t/Ts ]*2 )
down_up = lambda t :np.diag([1]*7   +   [0  ,1-0.4*t/Ts ,1-0.4*t/Ts ]*2 +   [0  ,1-t/Ts     ,1-t/Ts     ]*2 )
up      = lambda t :np.diag([1]*7   +   [0  ,1-t/Ts ,1-t/Ts ]*2 +   [0  ,1-t/Ts     ,1-t/Ts     ]*2 )

def computeTrajectory(q_0=robot.q0, T=T):
    Q = []
    vQ = []
    q = q_0.copy()
    vq = v_0.copy()
    CoM = CoM_0
    for t in T:
        if t<=Ts:
            prev_CoM = CoM.copy()
            CoM = X_CoM(t)
            Rot = Rot_CoM(t)
            Q.append(q.copy())
            vQ.append(vq.copy())
            q, vq = JacobianIK(Rot, CoM, q)
            # q = down(t)@q_0 if t<0.3*Ts else (down_up(t)@q_0 if t<0.7*Ts else up(t)@q_0)
            print(f"Time : t={t}")
        else:
            Q.append((1.5*q_0).copy())
            vQ.append(v_0.copy())
    return Q, vQ


def plot_controlledTrajectory(q_0=robot.q0, T=T, dt=dt):
    Q, vQ = computeTrajectory(q_0=q_0, T=T)
    L_X_CoM = []
    prev_X_CoM = CoM_0.copy()
    prev_V_CoM = v_CoM_0.copy()
    for q, t in zip(*[Q,T]):
        if t<=Ts:
            L_X_CoM.append(computeCoM(q))
            prev_X_CoM  = L_X_CoM[-1].copy()
            if t : prev_V_CoM  = (L_X_CoM[-1].copy()-L_X_CoM[-2].copy())/dt
        else:
            L_X_CoM.append(prev_X_CoM + prev_V_CoM*(t-Ts) + g*(t-Ts)**2/2)
            
    fig1, ax1 = plt.subplots()
    L_CoM = np.array(L_X_CoM)
    ax1.set_title("CoM")
    ax1.set_ylabel("x,y and z")
    ax1.set_xlabel("time")
    ax1.plot(T,L_CoM)
    ax1.plot(T[id_Ts],L_CoM[id_Ts][0],'ro')
    ax1.plot(T[id_Ts],L_CoM[id_Ts][1],'ro')
    ax1.plot(T[id_Ts],L_CoM[id_Ts][2],'ro')
    plt.show(block=False)
    fig2, ax2 = plt.subplots()
    ax2.set_title("CoM")
    ax2.set_ylabel("z")
    ax2.set_xlabel("x")
    ax2.plot(L_CoM[:,0],L_CoM[:,2])
    plt.show(block=False)
    fig3, ax3 = plt.subplots()
    ax3.plot(T,Q)
    ax3.set_title("Q")
    ax3.set_ylabel("configurations")
    ax3.set_xlabel("time")
    plt.show()