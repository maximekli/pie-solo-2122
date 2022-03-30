import time
import numpy as np
from numpy.linalg import norm, pinv
import matplotlib.pyplot as plt

import pinocchio as pin
import example_robot_data

from meshcat_viewer_wrapper import MeshcatVisualizer

robot = example_robot_data.load('solo12')
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)

print("Click on the link and press any key")
input()

IDX_BASE = robot.model.getFrameId('base_link')
IDX_FR_FOOT = robot.model.getFrameId('FR_FOOT')
IDX_FL_FOOT = robot.model.getFrameId('FL_FOOT')
IDX_HR_FOOT = robot.model.getFrameId('HR_FOOT')
IDX_HL_FOOT = robot.model.getFrameId('HL_FOOT')

q0 = robot.q0.copy()
dt = 1e-2

def rotMatY(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])

# base goal
oMbase_goal = pin.SE3(rotMatY(-np.pi/6),
                     np.array([0., 0., 0.15]))
# FR foot goal
o_FR_goal = robot.framePlacement(q0, IDX_FR_FOOT).translation.copy()
o_FR_goal[0] = 0.1

# FL foot goal
o_FL_goal = robot.framePlacement(q0, IDX_FL_FOOT).translation.copy()
o_FL_goal[0] = 0.15

# HR foot goal
o_HR_goal = robot.framePlacement(q0, IDX_HR_FOOT).translation.copy()
o_HR_goal[0] = -0.15

# HL foot goal
o_HL_goal = robot.framePlacement(q0, IDX_HL_FOOT).translation.copy()
o_HL_goal[0] = -0.1

q = q0.copy()
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

    q = pin.integrate(robot.model, q, vq * dt)

    viz.display(q)
    time.sleep(1e-3)

plt.figure()
plt.title('base')
plt.plot(herr_base)

plt.figure()
plt.title('FR-FL-HR-HL')
plt.plot(herr_FR)
plt.plot(herr_FL)
plt.plot(herr_HR)
plt.plot(herr_HL)

plt.show()