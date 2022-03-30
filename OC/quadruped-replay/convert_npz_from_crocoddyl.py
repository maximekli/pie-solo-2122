'''
Converts a npz file out of Crocoddyl into something understandable by the player
TODO: pad at the end to have a resting position
'''

import sys

import numpy as np

if len(sys.argv) == 1:
    quit()

dt_croco = 1e-2
dt = 1e-3
reps = int(dt_croco/dt)

npzfile = np.load(sys.argv[1])
xs = npzfile['xs']
us = npzfile['us']

q = xs[:-1, :19].repeat(10, axis=0).transpose()
v = xs[:-1, 19:].repeat(10, axis=0).transpose()
tau = us.repeat(10, axis=0).transpose()
t = [i*dt for i in range(reps * us.shape[0])]

np.savez('converted.' + sys.argv[1], q=q, v=v, tau=tau, t=t)