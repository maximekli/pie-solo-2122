import numpy as np
import matplotlib.pyplot as plt
import example_robot_data
from utils import tangage

robot   = example_robot_data.load('solo12')
NQ, NV  = robot.model.nq, robot.model.nv
model   = robot.model
data    = robot.data

q_0     = robot.q0.copy()
v_0     = robot.v0.copy()
CoM_0   = robot.com(q_0).copy()
v_CoM_0 = robot.vcom(q_0,v_0).copy()

dt      = 1e-3
Ts      = 0.1
L       = 1
h       = 1
g       = model.gravity.linear

alpha   = np.arctan(4*h/L)
Vs      = np.sqrt(2*abs(g[2])*h)/np.sin(alpha)
Tt      = 2*np.sqrt(2*h/abs(g[2])) + Ts
f_x     = L*np.sqrt(abs(g[2])/(8*h))/Ts
f_z     = np.sqrt(2*abs(g[2])*h)/Ts+abs(g[2])
T       = np.arange(0, Tt, dt)
id_Ts   = T.tolist().index(Ts)

ddx_CoM = lambda t : f_x if t<Ts else 0
ddy_CoM = lambda t : 0
ddz_CoM = lambda t : (f_z if t<Ts else 0)+g[2]
A_CoM   = lambda t : np.array([ddx_CoM(t), ddy_CoM(t), ddz_CoM(t)])

dx_CoM  = lambda t : f_x*t if t<Ts else f_x*Ts
dy_CoM  = lambda t : 0
dz_CoM  = lambda t : (f_z*t if t<Ts else f_z*Ts)+g[2]*t
V_CoM   = lambda t : v_CoM_0 + np.array([dx_CoM(t),dy_CoM(t),dz_CoM(t)])
# w_CoM   = lambda t : np.array([np.pi/Tt,0,0])

x_CoM   = lambda t : f_x*t**2/2 if t<Ts else f_x*Ts*(t-Ts)+f_x*Ts**2/2
y_CoM   = lambda t : 0
z_CoM   = lambda t : (f_z*t**2/2 if t<Ts else f_z*Ts*(t-Ts)+f_z*Ts**2/2)+g[2]*t**2/2
X_CoM   = lambda t : CoM_0 + np.array([x_CoM(t),y_CoM(t),z_CoM(t)])
Rot_CoM = lambda t : tangage(-2*np.pi/Tt*t)


def plot_modelTrajectory(T=T):
    fig1, ax1 = plt.subplots()
    L_CoM = np.array([X_CoM(t) for t in T])
    ax1.plot(T,L_CoM)
    ax1.plot(T[id_Ts],L_CoM[id_Ts][0],'ro')
    ax1.plot(T[id_Ts],L_CoM[id_Ts][1],'ro')
    ax1.plot(T[id_Ts],L_CoM[id_Ts][2],'ro')
    plt.show(block=False)
    fig2, ax2 = plt.subplots()
    ax2.plot(L_CoM[:,0],L_CoM[:,2])
    L_CoM = np.array([X_CoM(t) for k, t in enumerate(T) if not k%100])
    V_CoM = np.array([Rot_CoM(t).dot(np.array([1,0,0])) for k, t in enumerate(T) if not k%100])
    ax2.quiver(L_CoM[:,0], L_CoM[:,2], V_CoM[:,0], V_CoM[:,2])
    plt.show()
