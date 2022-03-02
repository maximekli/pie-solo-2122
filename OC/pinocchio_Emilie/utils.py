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
    
# 7+(0 mod 3) -> shoulders (not on x)
# 7+1 -> FL_shoulder (pi/2 -> bras vers l'arrière)
# 7+2 -> FL_knee (pi/2 -> avant bras vers l'arrière)
# 7+4 -> FR_shoulder (pi/2 -> bras vers l'arrière)
# 7+5 -> FR_knee (pi/2 -> avant bras vers l'arrière)
# 7+7 -> HL_shoulder (pi/2 -> bras vers l'arrière)
# 7+8 -> HL_knee (pi/2 -> avant bras vers l'arrière)
# 7+10 -> HR_shoulder (pi/2 -> bras vers l'arrière)
# 7+11 -> HR_knee (pi/2 -> avant bras vers l'arrière)

def computeCoM(q):
    l=0.16
    Zs = [0.16*np.cos(q[7+3*i+1])+0.16*np.cos(q[7+3*i+1]+q[7+3*i+2]) for i in range(4)]
    Xs = [0.16*np.sin(q[7+3*i+1])+0.16*np.sin(q[7+3*i+1]+q[7+3*i+2]) for i in range(4)]
    return np.array([sum(Xs)/4,0,sum(Zs)/4])
    
def computeRot(q):
    l=0.45/2
    w=0.30/2
    Zs = [0.16*np.cos(q[7+3*i+1])+0.16*np.cos(q[7+3*i+1]+q[7+3*i+2]) for i in range(4)]
    beta = np.arcsin(((Zs[0]+Zs[1])/2-(Zs[2]+Zs[3])/2)/l)
    gamma = np.arcsin(((Zs[0]+Zs[2])/2-(Zs[1]+Zs[3])/2)/w)
    return rotation_matrix(0, beta, gamma)
    