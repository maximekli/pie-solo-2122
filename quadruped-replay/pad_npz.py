import sys

import numpy as np

if len(sys.argv) != 4:
    quit()

dt = 1e-3
q0_flipped = np.array([0., 0., 0.235, 0., 0., 0., 1.,
                        -0.1, np.pi-0.8, np.pi-1.6,
                        0.1, np.pi-0.8, np.pi-1.6,
                        -0.1, np.pi-0.8, 1.6,
                        0.1, np.pi-0.8, 1.6]).reshape((19, 1))

q0 = np.array([0., 0., 0.235, 0., 0., 0., 1.,
                0.1, 0.8, -1.6,
                -0.1, 0.8, -1.6,
                0.1, -0.8, 1.6,
                -0.1, -0.8, 1.6]).reshape((19, 1))

npzfile = np.load(sys.argv[1])
q = npzfile['q']
v = npzfile['v']
tau = npzfile['tau']
t = npzfile['t']

knots = int(sys.argv[2])
q_end = q0_flipped if sys.argv[3] == "flipped" else q0

q_padded = np.zeros((q.shape[0], knots))
v_padded = np.zeros((v.shape[0], knots))
tau_padded = np.zeros((tau.shape[0], knots))

q_padded[:, :q.shape[1]] = q
v_padded[:, :v.shape[1]] = v
tau_padded[:, :tau.shape[1]] = tau

q_padded[:, q.shape[1]:] = np.full((q.shape[0], knots - q.shape[1]), q_end)
v_padded[:, v.shape[1]:] = np.zeros((v.shape[0], 1))
tau_padded[:, tau.shape[1]:] = np.zeros((tau.shape[0], 1))

t_padded = [i*dt for i in range(knots)]

np.savez('padded.' + sys.argv[1], q=q_padded, v=v_padded, tau=tau_padded, t=t_padded)