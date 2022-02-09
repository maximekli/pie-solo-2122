from termios import FF1
import time
import numpy as np
from numpy.linalg import norm, solve, pinv
from scipy.integrate import quad
import pinocchio as pin
import example_robot_data
import matplotlib.pyplot as plt

robot   = example_robot_data.load('solo12')
NQ, NV  = robot.model.nq, robot.model.nv
model   = robot.model
data    = robot.data

jointNames = robot.model.names
JOINT_IDs = range(len(jointNames))
jointMasses = [i.mass for i in model.inertias]
jointLevers = [i.lever for i in model.inertias]
jointInertias = [i.inertia for i in model.inertias]

M       = np.array(jointMasses)
m       = sum(jointMasses)
g       = model.gravity.linear

q_0     = robot.q0
v_0     = robot.v0
CoM_0   = robot.com(q_0)
v_CoM_0 = robot.vcom(q_0,v_0)

dt      = 1e-3
Ts      = 0.2
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
ddy_CoM = lambda t : 0
# ddz_CoM = lambda t : f_z if t<Ts else 0 # accélération de poussée (donc pas gravité dans cette accélération là)
ddz_CoM = lambda t : (f_z if t<Ts else 0)+g[2]
a_CoM   = lambda t : np.array([ddx_CoM(t), ddy_CoM(t), ddz_CoM(t)])
# dx_CoM  = lambda t : quad(ddx_CoM, 0, t)[0]
# dy_CoM  = lambda t : 0
# dz_CoM  = lambda t : quad(ddz_CoM, 0, t)[0]
# v_CoM   = lambda t :  np.array([dx_CoM(t),dy_CoM(t),dz_CoM(t)])
# w_CoM   = lambda t : np.array([np.pi/Tt,0,0])
# x_CoM  = lambda t : quad(dx_CoM, 0, t)[0]
# y_CoM  = lambda t : 0
# z_CoM  = lambda t : quad(dz_CoM, 0, t)[0]
# X_CoM  = lambda t :  np.array([x_CoM(t),y_CoM(t),z_CoM(t)])

L_vCoM  = [v_CoM_0]
for t in T:
    if t==0: continue
    L_vCoM.append(L_vCoM[-1]+dt*a_CoM(t))
v_CoM  = lambda t :  L_vCoM[T.tolist().index(t)]

L_CoM  = [CoM_0]
for t in T:
    if t==0: continue
    L_CoM.append(L_CoM[-1]+dt*v_CoM(t))
X_CoM  = lambda t :  L_CoM[T.tolist().index(t)]


def computeJointsConfiguration(CoM_ref, q_init, K=1, q_0=robot.q0, epsilon=1e-4):
    q           = q_init
    dq          = np.ones(q_0.shape)
    ## REFERENCE FOOTS PLACEMENTS
    F1_ref      = robot.framePlacement(q_0, robot.model.getFrameId('FR_FOOT')).translation
    F2_ref      = robot.framePlacement(q_0, robot.model.getFrameId('HR_FOOT')).translation
    F3_ref      = robot.framePlacement(q_0, robot.model.getFrameId('FL_FOOT')).translation
    F4_ref      = robot.framePlacement(q_0, robot.model.getFrameId('HL_FOOT')).translation
    # Theta_ref   = TODO
    while norm(dq) > epsilon :
        ## CURRENT CONFIGURATION
        CoM         = robot.com(q)
        F1          = robot.framePlacement(q, robot.model.getFrameId('FR_FOOT')).translation
        F2          = robot.framePlacement(q, robot.model.getFrameId('HR_FOOT')).translation
        F3          = robot.framePlacement(q, robot.model.getFrameId('FL_FOOT')).translation
        F4          = robot.framePlacement(q, robot.model.getFrameId('HL_FOOT')).translation
        # Theta       = TODO

        ## ERROR
        err_CoM     = CoM - CoM_ref
        err_F1      = F1 - F1_ref
        err_F2      = F2 - F2_ref
        err_F3      = F3 - F3_ref
        err_F4      = F4 - F4_ref
        # err_Theta   = TODO
        err         = np.concatenate((err_CoM, err_F1, err_F2, err_F3, err_F4), axis=0)

        ## JACOBIAN FROM CURRENT CONFIGURATION
        J_CoM       = robot.Jcom(q)
        J_F1        = robot.computeFrameJacobian(q, robot.model.getFrameId('FR_FOOT'))[:3]
        J_F2        = robot.computeFrameJacobian(q, robot.model.getFrameId('HR_FOOT'))[:3]
        J_F3        = robot.computeFrameJacobian(q, robot.model.getFrameId('FL_FOOT'))[:3]
        J_F4        = robot.computeFrameJacobian(q, robot.model.getFrameId('HL_FOOT'))[:3]
        # J_Theta     = TODO
        J           = np.concatenate((J_CoM, J_F1, J_F2, J_F3, J_F4), axis=0)

        Jpinv       = pinv(J)
        dq          = np.concatenate([np.array([0]), Jpinv.dot(-K*err)])
        q           = q + dq
    return q



# ''' tau = M(q)ddq + C(q,dq) + g(q) '''
# ''' tau = M(q)ddq + b(q,dq) '''
# q   = rand(robot.model.nq)
# vq  = rand(robot.model.nv)
# aq0 = np.zeros(robot.model.nv)
# b = pin.rnea(robot.model,robot.data,q,vq,aq0)  # compute dynamic drift -- Coriolis, centrifugal, gravity
# M = pin.crba(robot.model,robot.data,q)         # compute mass matrix M


def computeTrajectory(CoM_0=CoM_0, q_0=robot.q0, T=T):
    CoM = CoM_0
    Q = [q_0]
    for t in T:
        CoM += dt*v_CoM(t)
        if t==0: continue
        if t<Ts:
            Q.append(computeJointsConfiguration(CoM, Q[-1]))
        else:
            Q.append(q_0)
    return Q

Q = computeTrajectory()

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
test_flight()



def compute_trajectoryCoM(Q=Q):
    N = len(Q)
    com = CoM_0
    CoM = [CoM_0]
    for i in range(N-1):
        dq = Q[i+1]-Q[i]
        J = robot.Jcom(Q[i])
        v = J.dot(dq[1:])
        com = com + dt*v
        CoM.append(com)
    return CoM
CoM = compute_trajectoryCoM(Q)

def test_compute_trajectoryCoM(CoM=CoM, T=T[0:len(CoM)], id_Ts=id_Ts):
    X0 = CoM[id_Ts]
    CoM = np.array(CoM)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.5)
    ax1.plot(T, CoM[:,0])
    ax1.plot(Ts, X0[0], 'ro')
    ax1.set_xlabel('time')
    ax1.set_ylabel('X')
    ax1.grid(True)
    ax2.plot(T, CoM[:,1])
    ax2.plot(Ts, X0[1], 'ro')
    ax2.set_xlabel('time')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax3.plot(T, CoM[:,2])
    ax3.plot(Ts, X0[2], 'ro')
    ax3.set_xlabel('time')
    ax3.set_ylabel('Z')
    ax3.grid(True)
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(CoM[:,0], CoM[:,2])
    ax1.plot(X0[0], X0[2], 'ro')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.grid(True)
    plt.show(block=False)
test_compute_trajectoryCoM()



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