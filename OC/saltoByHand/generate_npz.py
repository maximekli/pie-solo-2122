import numpy as np
import example_robot_data
robot   = example_robot_data.load('solo12')

directory = 'trajectory_npz/'
file = 'backflip_by_hand.npz'



# Initial position
# The same as in inverse_kinematics.py
q0      = robot.q0.copy()
q0[7:13]= q0[13:]
dt = 1e-3

q = np.load('trajectory_npy/salto_Q.npy').transpose()
print(f'q shape : {q.shape}')
v = np.load('trajectory_npy/salto_vQ.npy').transpose()
print(f'v shape : {v.shape}')
tau = np.zeros((12, q.shape[1]))
# tau = np.load('trajectory_npy/salto_torques.npy').transpose()[0] # feed-forward (not successfully implemented here)
print(f'tau shape : {tau.shape}')

t = [i * dt for i in range(q.shape[1])]

np.savez(directory + file, q=q, v=v, tau=tau, t=t)

index = q.shape[1] - 500
print(index)

Kp_0 = 8
Kp_1 = 6

Kd_0 = 0.06
Kd_1 = 0.3

Kp = np.zeros((12, tau.shape[1]))
Kp[:, :index] = np.full((tau.shape[0], index), Kp_0)
Kp[:, index:] = np.full((tau.shape[0], tau.shape[1] - index), Kp_1)

Kd = np.zeros((12, tau.shape[1]))
Kd[:, :index] = np.full((tau.shape[0], index), Kd_0)
Kd[:, index:] = np.full((tau.shape[0], tau.shape[1] - index), Kd_1)

np.savez(directory + 'with_gains.' + file, q=q, v=v, tau=tau, t=t, Kp=Kp, Kd=Kd)

