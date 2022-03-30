import sys

import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    quit()

npzfile = np.load(sys.argv[1])
q = npzfile['q']
v = npzfile['v']
tau = npzfile['tau']
Kp = npzfile['Kp'] if 'Kp' in npzfile.files else np.zeros(tau.shape)
Kd = npzfile['Kd'] if 'Kd' in npzfile.files else np.zeros(tau.shape)
t = npzfile['t']

index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]


plt.figure()
plt.suptitle('q (rad)')
for i in range(12):
    plt.subplot(3, 4, index12[i])
    plt.plot(q[i + 7, :])

plt.figure()
plt.suptitle('v (rad/s)')
for i in range(12):
    plt.subplot(3, 4, index12[i])
    plt.plot(v[i + 6, :])

plt.figure()
plt.suptitle('tau ff (N.m)')
for i in range(12):
    plt.subplot(3, 4, index12[i])
    plt.plot(tau[i, :])

plt.figure()
plt.suptitle('Kp')
for i in range(12):
    plt.subplot(3, 4, index12[i])
    plt.plot(Kp[i, :])

plt.figure()
plt.suptitle('Kd')
for i in range(12):
    plt.subplot(3, 4, index12[i])
    plt.plot(Kd[i, :])

plt.figure()
plt.title('q')
plt.plot(q[8, :])
plt.vlines([850, 1060], min(q[8, :]), max(q[8, :]), 'r', 'dashed')
plt.xlabel('t (ms)')
plt.ylabel('q (rad)')

plt.figure()
plt.title('v')
plt.plot(v[7, :])
plt.vlines([850, 1060], min(v[7, :]), max(v[7, :]), 'r', 'dashed')
plt.xlabel('t (ms)')
plt.ylabel('v (rad/s)')
plt.show()
