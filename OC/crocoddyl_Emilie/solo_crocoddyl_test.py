import os
import sys
import time

import numpy as np
import crocoddyl # /opt/openrobots/lib/python3.8/site-packages/crocoddyl/
import pinocchio
from crocoddyl.utils.quadruped import plotSolution
from gait_problem import SimpleQuadrupedalGaitProblem
import example_robot_data

WITHDISPLAY = True

solo = example_robot_data.load('solo12')
robot_model = solo.model

# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = 'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT'
# for i in range(robot_model.nframes):
#     print(robot_model.frames[i])
gait = SimpleQuadrupedalGaitProblem(robot_model, lfFoot, rfFoot, lhFoot, rhFoot)

# Defining the initial state of the robot
q0 = robot_model.referenceConfigurations['standing'].copy()
v0 = pinocchio.utils.zero(robot_model.nv)
x0 = np.concatenate([q0, v0])

# Defining the walking gait parameters
walking_gait = {'stepLength': 0.25,
                'stepHeight': 0.25,
                'timeStep': 1e-2,
                'stepKnots': 25,
                'supportKnots': 2}
jumping_gait = {'jumpHeight': 0.15,
                'jumpLength': [0.3, 0.3, 0.],
                'timeStep': 1e-2,
                'groundKnots': 20,
                'flyingKnots': 30    }
salto_gait = {  'jumpHeight': 1,
                'jumpLength': [1, 0, 0],
                'timeStep': 1e-2,
                'groundKnots': 20,
                'flyingKnots': 40    }
rot_gait = {  'jumpHeight': 1,
                'jumpRot': np.pi,
                'timeStep': 1e-2,
                'groundKnots': 20,
                'flyingKnots': 40    }
# Setting up the control-limited DDP solver
# solver = crocoddyl.SolverFDDP(
#     gait.createTurnProblem(
#         x0,
#         rot_gait['jumpHeight'],
#         rot_gait['jumpRot'],
#         rot_gait['timeStep'],
#         rot_gait['groundKnots'],
#         rot_gait['flyingKnots']))
solver = crocoddyl.SolverFDDP(
    gait.createSaltoProblem(
        x0,
        salto_gait['jumpHeight'],
        salto_gait['jumpLength'],
        salto_gait['timeStep'],
        salto_gait['groundKnots'],
        salto_gait['flyingKnots']))
# solver = crocoddyl.SolverFDDP(
#     gait.createMyJumpingProblem(
#         x0,
#         jumping_gait['jumpHeight'],
#         jumping_gait['jumpLength'],
#         jumping_gait['timeStep'],
#         jumping_gait['groundKnots'],
#         jumping_gait['flyingKnots']))
# solver = crocoddyl.SolverBoxDDP(
#     gait.createWalkingProblem(
#         x0,
#         walking_gait['stepLength'],
#         walking_gait['stepHeight'],
#         walking_gait['timeStep'],
#         walking_gait['stepKnots'],
#         walking_gait['supportKnots']))

# Add the callback functions
print('*** SOLVE ***')
solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solve the DDP problem
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.solve(xs, us, 100, False, 0.1)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(solo,frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)