import sys

import numpy as np

if len(sys.argv) != 2:
    quit()

dt = 1e-3

# normal gains
Kp_0 = 6.0
Kd_0 = 0.3

# lighter gains
Kp_1 = 1.5
Kd_1 = 0.1

index = 1060

npzfile = np.load(sys.argv[1])
q = npzfile['q']
v = npzfile['v']
tau = npzfile['tau']
t = npzfile['t']

Kp = np.zeros((12, tau.shape[1]))
Kp[:, :index] = np.full((tau.shape[0], index), Kp_0)
Kp[:, index:] = np.full((tau.shape[0], tau.shape[1] - index), Kp_1)

Kd = np.zeros((12, tau.shape[1]))
Kd[:, :index] = np.full((tau.shape[0], index), Kd_0)
Kd[:, index:] = np.full((tau.shape[0], tau.shape[1] - index), Kd_1)

np.savez('with_gains.' + sys.argv[1], q=q, v=v, tau=tau, t=t, Kp=Kp, Kd=Kd)