import numpy as np
from numpy.linalg import norm, pinv
from utils import rotMatY

import pinocchio as pin
import example_robot_data

from meshcat_viewer_wrapper import MeshcatVisualizer

robot = example_robot_data.load('solo12')
viz = MeshcatVisualizer(robot)


# Initial position
q0      = robot.q0.copy()
q0[7:13]= q0[13:]
CoM_0   = robot.com(q0).copy()

# Gravity
g       = robot.model.gravity.linear

# Model with its tunnings
Dt      = 1e-2 # IK q/vq integration Dt
epsilon = 0.01 # IK tolerance
dt      = 1e-3 # Simulation dt
tuning1 = 2.4
tuning2 = 1.08
L       = tuning1*1
h       = tuning1*1
rot     = tuning2*tuning1*2*np.pi
T_jump      = 0.044 # A first guess of the jump time (if you know it, put it here to save you some time)
T_lean  = T_jump*0.27
viz.display(q0)




print("Click on the link and press any key to see the result of IK")
input()

# Frames of interest indices for IK
IDX_BASE = robot.model.getFrameId('base_link') # CoM approx
IDX_FR_FOOT = robot.model.getFrameId('FR_FOOT')
IDX_FL_FOOT = robot.model.getFrameId('FL_FOOT')
IDX_HR_FOOT = robot.model.getFrameId('HR_FOOT')
IDX_HL_FOOT = robot.model.getFrameId('HL_FOOT')

'''
We do not know a priori when the jump occurs.
As it is an important part of the modelisation, we guess it a first time
and then it is updated until the update is negligeable.
'''
T_jump_err  = 1 
while T_jump_err>2*dt :
    # as T_jump is updated, the model should be updated too
    Tt      = np.sqrt(2*h/abs(g[2])) + T_jump
    f_x     = 2*L*np.sqrt(abs(g[2])/(8*h))/T_jump
    f_z     = np.sqrt(2*abs(g[2])*h)/T_jump + abs(g[2])
    F       = np.array([f_x, 0, f_z])
    X_0     = np.array([CoM_0[0], CoM_0[1], CoM_0[2]/2])
    X_jump  = X_0 + F*T_jump**2/2
    V_jump  = F*T_jump
    X_CoM   = lambda t : (X_0 + F*t**2/2 if t<T_jump else X_jump + V_jump*(t-T_jump)) + g*t**2/2
    theta   = lambda t : -rot*t/Tt if t<T_lean \
                        else rot*(t-2*T_lean)/Tt
    Rot_CoM = lambda t : rotMatY(theta(t))

    Q = [] # store the joint configurations
    vQ = [] # store the first joint errors of each IK point


    ## Reach initial position by gradient descent
    # Here we want to add every intermediate configuration of the IK as a configuration of the trajecctory.
    # This will compute the trajectory to follow in order to reach the initial position for the jump.
    print("PREPARE...")
    q = q0.copy()
    t_prep = 0
    max_err = 1 # if max_err <= epsilon then no need to compute the next configuration because it would be the same
    while t_prep<2:
        if max_err>epsilon :
            # base goal: initial position
            oMbase_goal = pin.SE3(Rot_CoM(0), X_CoM(0))
            # FR foot goal: on the ground, as in q0
            o_FR_goal = robot.framePlacement(q0, IDX_FR_FOOT).translation.copy()
            # FL foot goal: on the ground, as in q0
            o_FL_goal = robot.framePlacement(q0, IDX_FL_FOOT).translation.copy()
            # HR foot goal: on the ground, as in q0
            o_HR_goal = robot.framePlacement(q0, IDX_HR_FOOT).translation.copy()
            # HL foot goal: on the ground, as in q0
            o_HL_goal = robot.framePlacement(q0, IDX_HL_FOOT).translation.copy()

            pin.framesForwardKinematics(robot.model,robot.data,q)
            pin.computeJointJacobians(robot.model,robot.data,q)

            # base task
            oMbase = robot.data.oMf[IDX_BASE]
            o_Jbase = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_BASE)
            baseMgoal = oMbase.inverse()*oMbase_goal
            base_nu = pin.log(baseMgoal).vector
            max_err = norm(base_nu)

            vq = pinv(o_Jbase) @ base_nu

            # FR foot task
            oMFR = robot.data.oMf[IDX_FR_FOOT]
            o_JFRxyz = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_FR_FOOT, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            o_FRgoal = oMFR.translation - o_FR_goal
            max_err = max(max_err,norm(o_FRgoal))

            Pbase = np.eye(robot.nv) - pinv(o_Jbase) @ o_Jbase
            vq += pinv(o_JFRxyz @ Pbase) @ (-o_FRgoal - o_JFRxyz @ vq)

            # FL foot task
            oMFL = robot.data.oMf[IDX_FL_FOOT]
            o_JFLxyz = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_FL_FOOT, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            o_FLgoal = oMFL.translation - o_FL_goal
            max_err = max(max_err,norm(o_FLgoal))

            PbaseFR = Pbase - pinv(o_JFRxyz @ Pbase) @ o_JFRxyz @ Pbase
            vq += pinv(o_JFLxyz @ PbaseFR) @ (-o_FLgoal - o_JFLxyz @ vq)

            # HR foot task
            oMHR = robot.data.oMf[IDX_HR_FOOT]
            o_JHRxyz = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_HR_FOOT, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            o_HRgoal = oMHR.translation - o_HR_goal
            max_err = max(max_err,norm(o_HRgoal))

            PbaseFRFL = PbaseFR - pinv(o_JFLxyz @ PbaseFR) @ o_JFLxyz @ PbaseFR
            vq += pinv(o_JHRxyz @ PbaseFRFL) @ (-o_HRgoal - o_JHRxyz @ vq)

            # HL foot task
            oMHL = robot.data.oMf[IDX_HL_FOOT]
            o_JHLxyz = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_HL_FOOT, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            o_HLgoal = oMHL.translation - o_HL_goal
            max_err = max(max_err,norm(o_HLgoal))
            
            PbaseFRFLHR = PbaseFRFL - pinv(o_JHRxyz @ PbaseFRFL) @ o_JHRxyz @ PbaseFRFL
            vq += pinv(o_JHLxyz @ PbaseFRFLHR) @ (-o_HLgoal - o_JHLxyz @ vq)

            q = pin.integrate(robot.model, q, 0.5*vq * Dt)
            viz.display(q)

        Q.append(q)
        vQ.append(np.zeros(vq.shape))
        t_prep+=dt

    print("AND PUSH!")
    t_salto = 0
    max_herr = 0
    while True:
        # base goal: modelled trajectory position
        oMbase_goal = pin.SE3(Rot_CoM(t_salto), X_CoM(t_salto))
        # FR foot goal: on the ground, as in q0
        o_FR_goal = robot.framePlacement(q0, IDX_FR_FOOT).translation.copy()
        # FL foot goal: on the ground, as in q0
        o_FL_goal = robot.framePlacement(q0, IDX_FL_FOOT).translation.copy()
        # HR foot goal: on the ground, as in q0
        o_HR_goal = robot.framePlacement(q0, IDX_HR_FOOT).translation.copy()
        # HL foot goal: on the ground, as in q0
        o_HL_goal = robot.framePlacement(q0, IDX_HL_FOOT).translation.copy()

        herr_base = []
        herr_FR = []
        herr_FL = []
        herr_HR = []
        herr_HL = []
        
        ## Inverse Kinematics
        for i in range(400):
            pin.framesForwardKinematics(robot.model,robot.data,q)
            pin.computeJointJacobians(robot.model,robot.data,q)

            # base task
            oMbase = robot.data.oMf[IDX_BASE]
            o_Jbase = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_BASE)
            baseMgoal = oMbase.inverse()*oMbase_goal
            base_nu = pin.log(baseMgoal).vector
            herr_base.append(norm(base_nu))

            vq = pinv(o_Jbase) @ base_nu

            # FR foot task
            oMFR = robot.data.oMf[IDX_FR_FOOT]
            o_JFRxyz = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_FR_FOOT, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            o_FRgoal = oMFR.translation - o_FR_goal
            herr_FR.append(norm(o_FRgoal))

            Pbase = np.eye(robot.nv) - pinv(o_Jbase) @ o_Jbase
            vq += pinv(o_JFRxyz @ Pbase) @ (-o_FRgoal - o_JFRxyz @ vq)

            # FL foot task
            oMFL = robot.data.oMf[IDX_FL_FOOT]
            o_JFLxyz = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_FL_FOOT, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            o_FLgoal = oMFL.translation - o_FL_goal
            herr_FL.append(norm(o_FLgoal))

            PbaseFR = Pbase - pinv(o_JFRxyz @ Pbase) @ o_JFRxyz @ Pbase
            vq += pinv(o_JFLxyz @ PbaseFR) @ (-o_FLgoal - o_JFLxyz @ vq)

            # HR foot task
            oMHR = robot.data.oMf[IDX_HR_FOOT]
            o_JHRxyz = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_HR_FOOT, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            o_HRgoal = oMHR.translation - o_HR_goal
            herr_HR.append(norm(o_HRgoal))

            PbaseFRFL = PbaseFR - pinv(o_JFLxyz @ PbaseFR) @ o_JFLxyz @ PbaseFR
            vq += pinv(o_JHRxyz @ PbaseFRFL) @ (-o_HRgoal - o_JHRxyz @ vq)

            # HL foot task
            oMHL = robot.data.oMf[IDX_HL_FOOT]
            o_JHLxyz = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_HL_FOOT, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            o_HLgoal = oMHL.translation - o_HL_goal
            herr_HL.append(norm(o_HLgoal))

            PbaseFRFLHR = PbaseFRFL - pinv(o_JHRxyz @ PbaseFRFL) @ o_JHRxyz @ PbaseFRFL
            vq += pinv(o_JHLxyz @ PbaseFRFLHR) @ (-o_HLgoal - o_JHLxyz @ vq)

            q = pin.integrate(robot.model, q, vq * Dt)
            if not i : vQ.append(vq)

        max_herr = max(herr_base[-1],herr_FR[-1],herr_FL[-1],herr_HR[-1],herr_HL[-1])
        if max_herr<epsilon:
            Q.append(q)
            viz.display(q)
            t_salto+=dt
        else:
            # if max_herr >= epsilon after IK then it might be because the position is unreachable
            # in our case, it would be because a foot is not on the ground
            # it is the jump time!
            vQ = vQ[:-1]
            break
        
    T_jump_err = abs(T_jump-t_salto)
    if T_jump_err>2*dt : print(f"T_jump was {T_jump}s but the robot jumped at {t_salto}s.\n")
    else : print(f"T_jump is {T_jump}s.\n")
    T_jump = t_salto

# Second jump phase, added after experiments on simulation
# Solo12 push with its two last feet on the ground
print("AND RE-PUSH!")
q = Q[-1] # because the previous q was the one with max_herr>=epsilon
max_herr = 0
while True:
    # base goal: modelled trajectory position
    oMbase_goal = pin.SE3(Rot_CoM(t_salto), X_CoM(t_salto))
    # FR foot goal: on the ground, as in q0
    o_FR_goal = robot.framePlacement(q0, IDX_FR_FOOT).translation.copy()
    # FL foot goal: on the ground, as in q0
    o_FL_goal = robot.framePlacement(q0, IDX_FL_FOOT).translation.copy()

    herr_base = []
    herr_FR = []
    herr_FL = []
    
    ## Inverse Kinematics
    for i in range(400):
        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)

        # base task
        oMbase = robot.data.oMf[IDX_BASE]
        o_Jbase = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_BASE)
        baseMgoal = oMbase.inverse()*oMbase_goal
        base_nu = pin.log(baseMgoal).vector
        herr_base.append(norm(base_nu))

        vq = pinv(o_Jbase) @ base_nu

        # FR foot task
        oMFR = robot.data.oMf[IDX_FR_FOOT]
        o_JFRxyz = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_FR_FOOT, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        o_FRgoal = oMFR.translation - o_FR_goal
        herr_FR.append(norm(o_FRgoal))

        Pbase = np.eye(robot.nv) - pinv(o_Jbase) @ o_Jbase
        vq += pinv(o_JFRxyz @ Pbase) @ (-o_FRgoal - o_JFRxyz @ vq)

        # FL foot task
        oMFL = robot.data.oMf[IDX_FL_FOOT]
        o_JFLxyz = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_FL_FOOT, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        o_FLgoal = oMFL.translation - o_FL_goal
        herr_FL.append(norm(o_FLgoal))

        PbaseFR = Pbase - pinv(o_JFRxyz @ Pbase) @ o_JFRxyz @ Pbase
        vq += pinv(o_JFLxyz @ PbaseFR) @ (-o_FLgoal - o_JFLxyz @ vq)

        q = pin.integrate(robot.model, q, vq * Dt)
        if not i : vQ.append(vq)

    max_herr = max(herr_base[-1],herr_FR[-1],herr_FL[-1])
    if max_herr<epsilon:
        Q.append(q)
        viz.display(q)
        t_salto+=dt
    else: 
        # if max_herr >= epsilon after IK then it might be because the position is unreachable
        # in our case, it would be because a foot is not on the ground
        # salto can begin! next configurations are computed directly in the main file
        vQ = vQ[:-1]
        break

# Save salto configurations
np.save('trajectory_npy/ik_Q.npy', Q, allow_pickle=True)
np.save('trajectory_npy/ik_vQ.npy', vQ, allow_pickle=True)