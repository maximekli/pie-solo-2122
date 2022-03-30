import sys

import numpy as np
import matplotlib.pyplot as plt

# param file name
if len(sys.argv) != 2:
    quit()

flipped = True
knots = 2000
knots1 = 850
knots2 = 1160

dt = 1e-3
q0_flipped = np.array([0., 0., 0.235, 0., 0., 0., 1.,
                       -0.1, np.pi-0.8, np.pi-1.6,
                       0.1, np.pi-0.8, np.pi-1.6,
                       -0.1, np.pi-0.8, 1.6,
                       0.1, np.pi-0.8, 1.6])

q0 = np.array([0., 0., 0.235, 0., 0., 0., 1.,
               0.1, 0.8, -1.6,
               -0.1, 0.8, -1.6,
               0.1, -0.8, 1.6,
               -0.1, -0.8, 1.6])

npzfile = np.load(sys.argv[1])
q = npzfile['q']
v = npzfile['v']
tau = npzfile['tau']
t = npzfile['t']

q_end = q0_flipped if flipped else q0

q_padded = np.zeros((q.shape[0], knots))
v_padded = np.zeros((v.shape[0], knots))
tau_padded = np.zeros((tau.shape[0], knots))

q_padded[:, :knots1] = q[:, :knots1]
v_padded[:, :knots1] = v[:, :knots1]
tau_padded[:, :knots1] = tau[:, :knots1]

q_padded[:, knots1] = q[:, knots1]
v_padded[:, knots1] = np.zeros(v.shape[0])

T = (knots2 - knots1) * dt
t0 = knots1 * dt
v_mean = np.zeros(v.shape[0])
v_mean[6:] = (q_end[7:] - q_padded[7:, knots1]) / T

for k in range(knots1 + 1, (knots1 + knots2)//2 + 1):
    t = k * dt
    v_padded[:, k] = 4 * v_mean / T * (t - t0)
    q_padded[7:, k] = q_padded[7:, k - 1] + dt * v_padded[6:, k]

t0 = (knots1 + knots2)/2.0 * dt
for k in range((knots1 + knots2)//2 + 1, knots2):
    t = k * dt
    v_padded[:, k] = - 4 * v_mean / T * (t - t0) + 2*v_mean
    q_padded[7:, k] = q_padded[7:, k - 1] + dt * v_padded[6:, k]

q_padded[:, knots2:] = np.full(
    (q.shape[0], knots - knots2), q_end.reshape(19, 1))

t_padded = [i*dt for i in range(knots)]

plt.figure()
plt.title('q')
for i in range(7, 19):
    plt.subplot(3, 4, i - 6)
    plt.plot(q_padded[i, :])

plt.figure()
plt.title('v')
for i in range(6, 18):
    plt.subplot(3, 4, i - 5)
    plt.plot(v_padded[i, :])

plt.show()

np.savez('trimmed_padded.' +
         sys.argv[1], q=q_padded, v=v_padded, tau=tau_padded, t=t_padded)
