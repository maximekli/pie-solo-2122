import sys

import numpy as np

if len(sys.argv) != 3:
    quit()

npzfile = np.load(sys.argv[1])
xs = npzfile['xs']
us = npzfile['us']

knots = int(sys.argv[2])
xs_trimmed = xs[:knots + 1, :]
us_trimmed = us[:knots, :]

np.savez('trimmed.' + sys.argv[1], xs=xs_trimmed, us=us_trimmed)