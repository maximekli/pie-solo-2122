import time
import numpy as np
from numpy.linalg import norm, solve
import pinocchio as pin
import example_robot_data
import matplotlib.pyplot as plt

# OUTDATED


robot   = example_robot_data.load('solo12')
NQ, NV  = robot.model.nq, robot.model.nv
model   = robot.model
data    = robot.data

jointNames = robot.model.names
JOINT_IDs = range(len(jointNames))
jointMasses = [i.mass for i in model.inertias]
jointLevers = [i.lever for i in model.inertias]
jointInertias = [i.inertia for i in model.inertias]

M       = np.array(jointMasses)
m       = sum(jointMasses)
g       = model.gravity.linear

q_0     = robot.q0
v_0     = robot.v0
CoM_0   = robot.com(q_0)
v_CoM_0 = robot.vcom(q_0,v_0)
dt      = 1e-3
Ts      = 0.2
Tt      = 2

Vx_0    = 10
Ry_lbd  = 5

def update_state(q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

def get_state(q):
    CoM             = robot.com(q)
    JCoM            = robot.Jcom(q)
    J               = robot.computeJointJacobians(q)
    jointPlacements = [robot.placement(q,JOINT_ID) for JOINT_ID in JOINT_IDs]
    jointCoMs       = np.array([p.translation for p in jointPlacements])
    jointRots       = np.array([p.rotation for p in jointPlacements])
    return locals()

def test_CoM():
    q = pin.randomConfiguration(model)
    dict = get_state(q)
    print(M.dot(dict['jointCoMs'])/m)
    print(dict['CoM'])
    print(dict['jointCoMs'][1])

''' JOINTS NAMES
    0:  'universe'   1: 'root_joint' 
    2:  'FL_HAA'     3: 'FL_HFE'    4:  'FL_KFE' 
    5:  'FR_HAA'     6: 'FR_HFE'    7:  'FR_KFE' 
    8:  'HL_HAA'     9: 'HL_HFE'   10:  'HL_KFE' 
   11:  'HR_HAA'    12: 'HR_HFE'   13:  'HR_KFE'
'''
''' FRAMES NAMES
joint 0 <- 0: 'universe'
joint 1 <- 1: 'root_joint' <- 2: 'base_link'

épaule joint 2 <- 3: 'FL_HAA' <- 4: 'FL_SHOULDER'     c'est la cheville
épaule joint 3 <- 5: 'FL_HFE' <- 6: 'FL_UPPER_LEG'       VVVVVVVVVV
coude  joint 4 <- 7: 'FL_KFE' <- 8: 'FL_LOWER_LEG' <- 9: 'FL_ANKLE' <- 10: 'FL_FOOT'

   11:  'FR_HAA'    12: 'FR_SHOULDER'  13:  'FR_HFE'    14: 'FR_UPPER_LEG'
   15:  'FR_KFE'    16: 'FR_LOWER_LEG' 17:  'FR_ANKLE'  18: 'FR_FOOT'
   
   19:  'HL_HAA'    20: 'HL_SHOULDER'  21:  'HL_HFE'    22: 'HL_UPPER_LEG'
   23:  'HL_KFE'    24: 'HL_LOWER_LEG' 25:  'HL_ANKLE'  26: 'HL_FOOT'
   
   27:  'HR_HAA'    28: 'HR_SHOULDER'  29:  'HR_HFE'    30: 'HR_UPPER_LEG'
   31:  'HR_KFE'    32: 'HR_LOWER_LEG' 33:  'HR_ANKLE'  34: 'HR_FOOT'
'''

def test_position_joints():
    q = pin.randomConfiguration(model)
    dict = get_state(q)
    # print(dict['jointRots'])
    print("###################")
    FL_HAA = dict['jointCoMs'][2]
    FL_HFE = dict['jointCoMs'][3]
    FL_KFE = dict['jointCoMs'][4]

dx_CoM  = lambda t : Vx_0*t/(0.9*Ts) if t<0.9*Ts else Vx_0
dy_CoM  = lambda t : 0
dz_CoM  = lambda t : -Ry_lbd*g[2]*t if t<0.9*Ts else (g[2]*(t-(1+Ry_lbd)*0.9*Ts)+(1+Ry_lbd)*g[2]*((t**2-(0.9*Ts)**2)/2-Ts*t+0.9*Ts**2)/(0.1*Ts) if t<Ts else (t-(1+Ry_lbd)/2*1.9*Ts)*g[2])
v_CoM   = lambda t :  np.array([dz_CoM(t),dx_CoM(t),dy_CoM(t)])
w_CoM   = lambda t : np.array([np.pi/Tt,0,0])

def test_dz():
    ddz_CoM = lambda t : (
        -Ry_lbd*g[2] if t<0.9*Ts
        else (
            g[2]+(1+Ry_lbd)*g[2]*(t-Ts)/(0.1*Ts) if t<Ts
            else g[2]
            )
    )
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)
    dt  = 0.01
    T   = np.arange(0, 2, dt)
    ax1.plot(T, [dz_CoM(t) for t in T])
    ax1.set_xlabel('time')
    ax1.set_ylabel('dz_CoM')
    ax1.grid(True)
    dZ  = [0]
    for i in range(len(T)-1):
        dZ.append(dZ[i]+ddz_CoM(T[i])*dt)
    ax2.plot(T, dZ)
    ax2.set_xlabel('time')
    ax2.set_ylabel('ddz_CoM')
    ax2.grid(True)
    plt.show(block=False)

test_dz()

def flight(T=Tt):
    from numpy.linalg import pinv
    Q   = [q_0]
    q   = q_0
    for t in np.arange(0, T, dt):
        J       = robot.Jcom(q)
        v       = v_CoM(t)
        w       = w_CoM(t)
        Jpinv   = pinv(J)
        dq      = np.concatenate([np.array([0]), Jpinv.dot(v)])
        q       = q + dt*dq
        Q.append(q)
    return Q

Q = flight()

def test_flight(Q):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)
    T = [dt*i for i in range(len(Q))]
    Q = np.array(Q)
    ax1.plot(T, Q[:,3])
    ax1.set_xlabel('time')
    ax1.set_ylabel('joint 3: FL_HFE')
    ax1.grid(True)
    ax2.plot(T, Q[:,10])
    ax2.set_xlabel('time')
    ax2.set_ylabel('joint 10: HL_KFE')
    ax2.grid(True)
    plt.show(block=False)

test_flight(Q)

def compute_trajectory(Q):
    com = CoM_0
    CoM = [CoM_0]
    for i in range(len(Q)-1):
        dq = Q[i+1]-Q[i]
        J = robot.Jcom(Q[i])
        v = J.dot(dq[1:])
        com = com + dt*v
        if (com-CoM_0)[2] < 0: break
        CoM.append(com)
    return CoM

CoM = compute_trajectory(Q)

def test_flight_CoM(CoM):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.5)
    T = [dt*i for i in range(len(CoM))]
    CoM = np.array(CoM)
    ax1.plot(T, CoM[:,1])
    ax1.set_xlabel('time')
    ax1.set_ylabel('X')
    ax1.grid(True)
    ax2.plot(T, CoM[:,2])
    ax2.set_xlabel('time')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax3.plot(T, CoM[:,0])
    ax3.set_xlabel('time')
    ax3.set_ylabel('Z')
    ax3.grid(True)
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(CoM[:,1], CoM[:,0])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.grid(True)
    plt.show(block=False)

test_flight_CoM(CoM)

def simulate(Q):
    robot.initDisplay(loadModel=True)
    for q in Q:
        update_state(q)
        robot.display(q)
        time.sleep(dt)

simulate(Q)

 







# robot.getJointJacobian(JOINT_ID)
# robot.jointJacobian(q,JOINT_ID)
# robot.computeFrameJacobian(q,FRAME_ID)
# robot.computeJointJacobian(q,JOINT_ID)
# robot.computeJointJacobians(q)

# robot.centroidal(q,v)
# robot.centroidalMap(q)
# robot.centroidalMomentum(q,v)
# robot.centroidalMomentumVariation(q,v,a)

# robot.model
# robot.data

# robot.com()
# robot.vcom(q,v)
# robot.acom(q,v,a)
# robot.Jcom(q)

# robot.index(JOINT_NAME)
# robot.nq
# robot.nv

''' tau = M(q)ddq + C(q,dq) + g(q) '''
# robot.gravity(q)
# robot.mass(q)

# robot.q0
# robot.v0
# robot.velocity(q,v,JOINT_ID)
# robot.classicalAcceleration(q,v,a,JOINT_ID)
# robot.acceleration(q,v,a,JOINT_ID)
# robot.nle(q,v)


# robot.forwardKinematics(q)
# robot.placement(q,JOINT_ID)
# robot.updateGeometryPlacements()


# robot.frameAcceleration(q,v,a,FRAME_ID)
# robot.frameClassicAcceleration(FRAME_ID)
# robot.frameClassicalAcceleration(q,v,a,FRAME_ID)
# robot.frameJacobian(q,FRAME_ID)
# robot.framePlacement(q,FRAME_ID)
# robot.frameVelocity(q,v,FRAME_ID)
# robot.framesForwardKinematics(q)
# robot.getFrameJacobian(FRAME_ID)