import numpy as np 

# Rotation matrix around Y axis
def rotMatY(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])