import numpy as np

###########
#  UTILS ##
###########

''' lacet, tangage, roulis = α, β, γ '''
def rotation_matrix(alpha, beta, gamma):
    R_x = np.array([[1 ,0             ,0             ],
                    [0 ,np.cos(gamma) ,-np.sin(gamma)],
                    [0 ,np.sin(gamma) ,np.cos(gamma) ]])
    R_y = np.array([[np.cos(beta)  ,0 ,np.sin(beta)],
                    [0             ,1 ,0           ],
                    [-np.sin(beta) ,0 ,np.cos(beta)]])
    R_z = np.array([[np.cos(alpha) ,-np.sin(alpha) ,0],
                    [np.sin(alpha) ,np.cos(alpha)  ,0],
                    [0             ,0              ,1]])
    return np.around(R_z.dot(R_y.dot(R_x)), decimals=10)

def tangage(beta):
    R_y = np.array([[np.cos(beta)  ,0 ,np.sin(beta)],
                    [0             ,1 ,0           ],
                    [-np.sin(beta) ,0 ,np.cos(beta)]])
    return R_y