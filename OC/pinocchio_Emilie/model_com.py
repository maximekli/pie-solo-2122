import time
import numpy as np
from numpy.linalg import norm, solve, pinv
from scipy.integrate import quad
import pinocchio as pin
import example_robot_data
import matplotlib.pyplot as plt
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
L       = 2
h       = 2
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
V_CoM   = lambda t :  np.array([dx_CoM(t),dy_CoM(t),dz_CoM(t)])
# w_CoM   = lambda t : np.array([np.pi/Tt,0,0])

x_CoM  = lambda t : f_x*t**2/2 if t<Ts else f_x*Ts*(t-Ts)+f_x*Ts**2/2
y_CoM  = lambda t : 0
z_CoM  = lambda t : (f_z*t**2/2 if t<Ts else f_z*Ts*(t-Ts)+f_z*Ts**2/2)+g[2]*t**2/2
X_CoM  = lambda t :  np.array([x_CoM(t),y_CoM(t),z_CoM(t)])
Rot_CoM = lambda t : tangage(np.pi/Tt*t)