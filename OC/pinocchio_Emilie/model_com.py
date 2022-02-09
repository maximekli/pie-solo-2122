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

dt      = 1e-2
Ts      = 0.1
L       = 10
h       = 10

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


def trajectory(T=T):
    CoM = [CoM_0]
    for t in T:
        if t==0: continue
        CoM.append(CoM[-1] + dt*v_CoM(t))
    return CoM
CoM = trajectory()

def test_trajectory(CoM=CoM, T=T, id_Ts=id_Ts):
    X0 = CoM[id_Ts]
    CoM = np.array([X-X0 for X in CoM])
    X0 = CoM[id_Ts]
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
test_trajectory()

def flight(T=T):
    Q   = [q_0]
    dQ  = [np.zeros(q_0.size)]
    q   = q_0
    for t in T:
        if t==0: continue
        J       = robot.Jcom(q)
        v       = v_CoM(t)
        w       = w_CoM(t)
        Jpinv   = pinv(J)
        dq      = np.concatenate([np.array([0]), Jpinv.dot(v)])
        q       = q + dt*dq
        dQ.append(dq)
        Q.append(q)
    return Q, dQ
Q, dQ = flight()
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
def test_flight(Q=Q, T=T, id_Ts=id_Ts):
    Q_Ts = Q[id_Ts]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)
    Q = np.array(Q)
    ax1.plot(T, Q[:,7+1])
    ax1.plot(Ts, Q[id_Ts,7+1], 'ro')
    ax1.set_xlabel('time')
    ax1.set_ylabel('joint 1: FL_HFE')
    ax1.grid(True)
    ax2.plot(T, Q[:,7+10])
    ax2.plot(Ts, Q[id_Ts,7+10], 'ro')
    ax2.set_xlabel('time')
    ax2.set_ylabel('joint 10: HL_KFE')
    ax2.grid(True)
    plt.show(block=False)
test_flight()

def compute_trajectory(Q=Q):
    com = CoM_0
    CoM = [CoM_0]
    for i in range(len(Q)-1):
        dq = Q[i+1]-Q[i]
        J = robot.Jcom(Q[i])
        v = J.dot(dq[1:])
        com = com + dt*v
        if (com-CoM_0)[2] < 0: break
        CoM.append(com)
    return CoM
CoM = compute_trajectory(Q)

def test_flight_CoM(CoM=CoM, T=T, id_Ts=id_Ts):
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
test_flight_CoM()


 


