import numpy as np
from numpy.linalg import norm, pinv

import pinocchio as pin
import example_robot_data

from meshcat_viewer_wrapper import MeshcatVisualizer

robot = example_robot_data.load('solo12')
viz = MeshcatVisualizer(robot)


q0      = robot.q0.copy()
q0[7]   = 0
q0[10]   = 0
q0[7:13]= q0[13:]
CoM_0   = robot.com(q0).copy()
g       = robot.model.gravity.linear

Dt      = 1e-2
dt      = 1e-3
tuning1 = 2.4
tuning2 = 1.08
L       = tuning1*1
h       = tuning1*1
rot     = tuning2*tuning1*2*np.pi
T_jump      = 0.044
T_lean  = T_jump*0.27
viz.display(q0)




print("Click on the link and press any key")
input()

IDX_BASE = robot.model.getFrameId('base_link')
IDX_FR_FOOT = robot.model.getFrameId('FR_FOOT')
IDX_FL_FOOT = robot.model.getFrameId('FL_FOOT')
IDX_HR_FOOT = robot.model.getFrameId('HR_FOOT')
IDX_HL_FOOT = robot.model.getFrameId('HL_FOOT')

def rotMatY(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])




T_jump_err  = 1
while T_jump_err>2*dt :
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

    print("PREPARE...")
    Q = []
    vQ = []
    t = 0
    q = q0.copy()
    max_err = 1
    ## INIT POS
    while t<2:
        if max_err > 0.01 :
            # base goal
            oMbase_goal = pin.SE3(Rot_CoM(0), X_CoM(0))
            # FR foot goal
            o_FR_goal = robot.framePlacement(q0, IDX_FR_FOOT).translation.copy()
            # FL foot goal
            o_FL_goal = robot.framePlacement(q0, IDX_FL_FOOT).translation.copy()
            # HR foot goal
            o_HR_goal = robot.framePlacement(q0, IDX_HR_FOOT).translation.copy()
            # HL foot goal
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

            q = pin.integrate(robot.model, q, vq * Dt)
            viz.display(q)

        Q.append(q)
        vQ.append(np.zeros(vq.shape))
        t+=dt

    print("AND PUSH!")
    t = 0
    max_herr = 0
    epsilon = 0.01
    while True:
        # base goal
        oMbase_goal = pin.SE3(Rot_CoM(t), X_CoM(t))
        # FR foot goal
        o_FR_goal = robot.framePlacement(q0, IDX_FR_FOOT).translation.copy()
        # FL foot goal
        o_FL_goal = robot.framePlacement(q0, IDX_FL_FOOT).translation.copy()
        # HR foot goal
        o_HR_goal = robot.framePlacement(q0, IDX_HR_FOOT).translation.copy()
        # HL foot goal
        o_HL_goal = robot.framePlacement(q0, IDX_HL_FOOT).translation.copy()

        herr_base = []
        herr_FR = []
        herr_FL = []
        herr_HR = []
        herr_HL = []
        
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
            t+=dt
        else: break
        
    T_jump_err = abs(T_jump-t)
    if T_jump_err>2*dt : print(f"T_jump was {T_jump}s but the robot jumped at {t}s.\n")
    else : print(f"T_jump is {T_jump}s.\n")
    T_jump = t

print("AND RE-PUSH!")
q = Q[-1]
max_herr = 0
epsilon = 0.01
while True:
    # base goal
    oMbase_goal = pin.SE3(Rot_CoM(t), X_CoM(t))
    # FR foot goal
    o_FR_goal = robot.framePlacement(q0, IDX_FR_FOOT).translation.copy()
    # FL foot goal
    o_FL_goal = robot.framePlacement(q0, IDX_FL_FOOT).translation.copy()

    herr_base = []
    herr_FR = []
    herr_FL = []
    
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
        t+=dt
    else: break


np.save('Q.npy', Q, allow_pickle=True)
np.save('vQ.npy', vQ, allow_pickle=True)