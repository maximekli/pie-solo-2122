import numpy as np

dt = 1e-3

q = np.load('trajectory_npy/salto_Q.npy').transpose()
v = np.load('trajectory_npy/salto_vQ.npy').transpose()
tau = np.zeros((12, q.shape[1]))

t = [i * dt for i in range(q.shape[1])]

np.savez('trajectory_npz/backflip_by_hand.npz', q=q, v=v, tau=tau, t=t)

