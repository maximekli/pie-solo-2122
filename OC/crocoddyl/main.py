import sys
import datetime

import numpy as np

import crocoddyl
import example_robot_data
import pinocchio

from quadrupedal_jumping_problem import Solo12JumpingProblem, plotSolution

WITHDISPLAY = 'display' in sys.argv
WITHPLOT = 'plot' in sys.argv
SAVE = 'save' in sys.argv

# Loading the solo12 model
solo12 = example_robot_data.load('solo12')

# Defining the initial state of the robot
q0 = solo12.model.referenceConfigurations['standing'].copy()
v0 = pinocchio.utils.zero(solo12.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up de jumping problem
# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = 'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT'
jump = Solo12JumpingProblem(solo12.model, lfFoot, rfFoot, lhFoot, rhFoot)

timeStep = 1e-2
solver = crocoddyl.SolverFDDP(
    # jump.createSimpleJumpingProblem(x0, timeStep)
    # jump.createYawJumpingProblem(x0, timeStep)
    jump.createHalfBackflipProblem(x0, timeStep)
)

# Added the callback functions
print('*** SOLVE ***')
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(solo12, 4, 4, frameNames=[
        lfFoot, rfFoot, lhFoot, rhFoot])
    solver.setCallbacks(
        [crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(solo12, 4, 4, frameNames=[
        lfFoot, rfFoot, lhFoot, rhFoot])
    solver.setCallbacks([crocoddyl.CallbackVerbose(),
                        crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the DDP solver
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.solve(xs, us, 200, False)

# Save result in .npz file if requested
if SAVE:
    np.savez(datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.npz',
             xs=solver.xs, us=solver.us
             )

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(
        solo12, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    display.displayFromSolver(solver)

# Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[0]
    plotSolution(solver, figIndex=1, show=False)

    title = "convergence"
    log = solver.getCallbacks()[0]
    crocoddyl.plotConvergence(log.costs,
                              log.u_regs,
                              log.x_regs,
                              log.grads,
                              log.stops,
                              log.steps,
                              figTitle=title,
                              figIndex=3,
                              show=True)


def replay():
    display.displayFromSolver(solver)
