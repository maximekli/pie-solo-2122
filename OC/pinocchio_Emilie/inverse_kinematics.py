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
    fJ = pin.computeFrameJacobian(model, data, q, IDX_TOOL)[:, -12:]  # Take all terms
    oA = robot.data.oMf[IDX_TOOL].action
    oJ = oA @ fJ  # Transformation from local frame to world frame
    J = [oJ[:, -12:]]
    for FRAME_ID in FRAME_IDs:
        fJ  = pin.computeFrameJacobian(model, data, q, FRAME_ID)[:3, -12:]  # Take only the translation terms
        oR = data.oMf[FRAME_ID].rotation
        oJ = oR @ fJ  # Transformation from local frame to world frame
        J.append(oJ[:, -12:])
    return J
def computeErrors(oMgoal, oMtool, placements_ref, placements):
    err_CoM     = pin.log(oMtool.inverse()*oMgoal).vector
    err_Frames  = [Fi-Fi_ref for Fi, Fi_ref in zip(placements,placements_ref)]
    return [err_CoM] + err_Frames
def stepIK_Jacobian(q, oMgoal, frameTrans_ref, K, DT):
    ## Run the algorithms that outputs values in data
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    ## CURRENT CONFIGURATION
    oMtool      = data.oMf[IDX_TOOL]
    frameTrans  = [data.oMf[FRAME_ID].translation.copy() for FRAME_ID in FRAME_IDs]
    ## ERROR
    L_err       = computeErrors(oMgoal, oMtool, frameTrans_ref, frameTrans)
    ## JACOBIAN FROM CURRENT CONFIGURATION
    L_J         = computeJacobians(q, IDX_TOOL, FRAME_IDs)
    ## UPDATE CONFIGURATION
    err = np.concatenate(L_err)
    J = np.concatenate(L_J)
    vq = -K * pinv(J) @ err
    vq = vq.reshape((12,1))
    vq = np.concatenate((np.zeros((6,1)), vq))
    new_q = pin.integrate(model,q, vq * DT)
    return new_q, vq

def stepIK_InvGeom(oMbase, oMgoal, frameTrans_ref, delta):
    ## INVERSE GEOMETRY FOR FEET
    def cost(q):
        '''Compute score from a configuration'''
        frameTrans  = [robot.framePlacement(q,FRAME_ID).translation.copy() for FRAME_ID in FRAME_IDs]
        frameErrors = [norm(Fi-Fi_ref) for Fi, Fi_ref in zip(frameTrans,frameTrans_ref)]
        oMtool  = robot.framePlacement(q, IDX_TOOL)
        return sum(frameErrors) + norm(pin.log(oMtool.inverse() * oMbase).vector)
    q_12    = fmin_bfgs(cost, robot.q0)[7:]
    q       = np.concatenate([pin.SE3ToXYZQUAT(oMbase),q_12])
    oMtool  = robot.framePlacement(q, IDX_TOOL)
    ## ERROR
    err     = pin.log(oMtool.inverse()*oMgoal)
    ## UPDATE CONFIGURATION
    oMbase  = pin.exp(pin.log(oMbase)+delta*err)
    new_q   = np.concatenate([pin.SE3ToXYZQUAT(oMbase),q_12])
    return oMbase, new_q, err

def inverseKinematic(Rot_ref, CoM_ref, q_init, K=100, DT=dt, delta=0.1, q_0=robot.q0, epsilon=1e-3, IT_MAX = 100):
    ## REFERENCE PLACEMENTS
    oMgoal          = pin.SE3(Rot_ref,CoM_ref)
    frameTrans_ref  = [robot.framePlacement(q_0, FRAME_ID).translation.copy() for FRAME_ID in FRAME_IDs]
    ## INITIALIZATION LOOP
    q       = q_init.copy()
    oMbase  = robot.framePlacement(q, IDX_TOOL).copy()
    herr    = []
    while True :
        ## IK STEP
        q, err = stepIK_Jacobian(q, oMgoal, frameTrans_ref, K, DT)
        # oMbase, q, err = stepIK_InvGeom(oMbase, oMgoal, frameTrans_ref, delta)
        ## UPDATE LOOP
        herr.append(norm(err))
        if not len(herr)%1000: print(f"norm(err) = {norm(err)}")
        if (norm(err) < epsilon)or(len(herr) >= IT_MAX): break
    if len(herr) < IT_MAX:
        print(f"Convergence achieved in {len(herr)} turns with norm(err)={norm(err)}!")
    else:
        print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
    return q


def computeTrajectory(q_0=robot.q0, T=T):
    Q = []
    q = q_0.copy()
    CoM = CoM_0
    for t in T:
        if t<=Ts:
            CoM = X_CoM(t)
            Rot = Rot_CoM(t)
            Q.append(q.copy())
            q = inverseKinematic(Rot, CoM, q)
        else:
            Q.append((q_0).copy())
    np.save('Q.npy', Q, allow_pickle=True)
    return Q



def plot_controlledTrajectory(q_0=robot.q0, T=T, dt=dt):
    Q = np.load('Q.npy', allow_pickle=True)
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
    for qi in Q[id_Ts,:]: ax3.plot(T[id_Ts],qi,'ro')
    ax3.set_title("Q")
    ax3.set_ylabel("configurations")
    ax3.set_xlabel("time")
    plt.show()