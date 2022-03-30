import crocoddyl
import pinocchio
import numpy as np


class Solo12JumpingProblem:
    def __init__(self, rmodel, lfFoot, rfFoot, lhFoot, rhFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.lfFootId = self.rmodel.getFrameId(lfFoot)
        self.rfFootId = self.rmodel.getFrameId(rfFoot)
        self.lhFootId = self.rmodel.getFrameId(lhFoot)
        self.rhFootId = self.rmodel.getFrameId(rhFoot)
        self.baseFrameId = self.rmodel.getFrameId('base_link')
        # Defining default state
        q0 = self.rmodel.referenceConfigurations["standing"]
        self.rmodel.defaultState = np.concatenate(
            [q0, np.zeros(self.rmodel.nv)])
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)

        self.half = 0.1946

    def createSimpleJumpingProblem(self, x0, timeStep):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        basePos0 = self.rdata.oMf[self.baseFrameId].translation
        print(basePos0)

        def rotMatY(rotAngle):
            return np.array([[np.cos(rotAngle), 0.0, np.sin(rotAngle)], [0.0, 1.0, 0.0], [-np.sin(rotAngle), 0.0, np.cos(rotAngle)]])

        loco3dModel = []

        delatSpring = 0.15
        springKnots = 50
        spring = []
        for i in range(springKnots):
            spring += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[self.lfFootId, self.rfFootId,
                                self.lhFootId, self.rhFootId],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=np.eye(3),
                frameTranslationTask=np.array(
                    [0.0, 0.0, basePos0[2] - delatSpring*(i + 1)/springKnots]),
            )]

        pushingKnots = 35
        pushing = []
        for i in range(pushingKnots):
            pushing += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[self.lfFootId, self.rfFootId,
                                self.lhFootId, self.rhFootId],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=None,
                frameTranslationTask=None,
            )]

        risingKnots = 20
        rising = []
        for i in range(risingKnots):
            rising += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=None,
                frameTranslationTask=None,
            )]

        h = 0.6
        atTheTop = [
            self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=rotMatY(0),
                frameTranslationTask=np.array(
                   [basePos0[0], basePos0[1], basePos0[2]+h]),
            )
        ]

        fallingKnots = 20
        falling = []
        for i in range(fallingKnots):
            falling += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=None,
                frameTranslationTask=None,
            )]

        footTask = [
            [self.lfFootId, pinocchio.SE3(
                np.eye(3), lfFootPos0)],
            [self.rfFootId, pinocchio.SE3(
                np.eye(3), rfFootPos0)],
            [self.lhFootId, pinocchio.SE3(
                np.eye(3), lhFootPos0)],
            [self.rhFootId, pinocchio.SE3(
                np.eye(3), rhFootPos0)]
        ]

        landed = [self.createSwingFootModel(
            timeStep=timeStep,
            supportFootIds=[self.lfFootId, self.rfFootId,
                            self.lhFootId, self.rhFootId],
            comTask=None,
            swingFootTask=footTask,
            frameRotationTask=rotMatY(0),
            frameTranslationTask=basePos0,
        )]

        loco3dModel += spring
        loco3dModel += pushing
        loco3dModel += rising
        loco3dModel += atTheTop
        loco3dModel += falling
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createYawJumpingProblem(self, x0, timeStep):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        basePos0 = self.rdata.oMf[self.baseFrameId].translation
        print(basePos0)

        def rotMatZ(rotAngle):
            return np.array([[np.cos(rotAngle), -np.sin(rotAngle), 0.0], [np.sin(rotAngle), np.cos(rotAngle), 0.0], [0.0, 0.0, 1.0]])

        loco3dModel = []

        delatSpring = 0.15
        springKnots = 50
        spring = []
        for i in range(springKnots):
            spring += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[self.lfFootId, self.rfFootId,
                                self.lhFootId, self.rhFootId],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=np.eye(3),
                frameTranslationTask=np.array(
                    [0.0, 0.0, basePos0[2] - delatSpring*(i + 1)/springKnots]),
            )]

        pushingKnots = 35
        pushing = []
        for i in range(pushingKnots):
            pushing += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[self.lfFootId, self.rfFootId,
                                self.lhFootId, self.rhFootId],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=None,
                frameTranslationTask=None,
            )]

        risingKnots = 20
        rising = []
        for i in range(risingKnots):
            rising += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=None,
                frameTranslationTask=None,
            )]

        h = 0.6
        atTheTop = [
            self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=None,
                frameTranslationTask=np.array(
                   [basePos0[0], basePos0[1], basePos0[2]+h]),
            )
        ]

        fallingKnots = 20
        falling = []
        for i in range(fallingKnots):
            falling += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=None,
                frameTranslationTask=None,
            )]

        footTask = [
            [self.lfFootId, pinocchio.SE3(
                rotMatZ(np.pi/2), rotMatZ(np.pi/2) @ lfFootPos0)],
            [self.rfFootId, pinocchio.SE3(
                rotMatZ(np.pi/2), rotMatZ(np.pi/2) @ rfFootPos0)],
            [self.lhFootId, pinocchio.SE3(
                rotMatZ(np.pi/2), rotMatZ(np.pi/2) @ lhFootPos0)],
            [self.rhFootId, pinocchio.SE3(
                rotMatZ(np.pi/2), rotMatZ(np.pi/2) @ rhFootPos0)]
        ]

        landed = [self.createSwingFootModel(
            timeStep=timeStep,
            supportFootIds=[self.lfFootId, self.rfFootId,
                            self.lhFootId, self.rhFootId],
            comTask=None,
            swingFootTask=footTask,
            frameRotationTask=rotMatZ(np.pi/2),
            frameTranslationTask=basePos0,
        )]

        loco3dModel += spring
        loco3dModel += pushing
        loco3dModel += rising
        loco3dModel += atTheTop
        loco3dModel += falling
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createHalfBackflipProblem(self, x0, timeStep):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        basePos0 = self.rdata.oMf[self.baseFrameId].translation
        print(basePos0)

        def rotMatY(rotAngle):
            return np.array([[np.cos(rotAngle), 0.0, np.sin(rotAngle)], [0.0, 1.0, 0.0], [-np.sin(rotAngle), 0.0, np.cos(rotAngle)]])

        loco3dModel = []

        delatSpring = 0.15
        springKnots = 50
        spring = []
        for i in range(springKnots):
            spring += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[self.lfFootId, self.rfFootId,
                                self.lhFootId, self.rhFootId],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=np.eye(3),
                frameTranslationTask=np.array(
                    [0.0, 0.0, basePos0[2] - delatSpring*(i + 1)/springKnots]),
            )]

        pushingKnots = 35
        pushing = []
        for i in range(pushingKnots):
            pushing += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[self.lfFootId, self.rfFootId,
                                self.lhFootId, self.rhFootId],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=None,
                frameTranslationTask=None,
            )]

        risingKnots = 20
        rising = []
        for i in range(risingKnots):
            rising += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=None,
                frameTranslationTask=None,
            )]

        delta = 0.2
        h = 0.6
        atTheTop = [
            self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=rotMatY(-np.pi/2),
                frameTranslationTask=np.array(
                   [-self.half - delta/2, 0.0, basePos0[2]+h]),
            )
        ]

        fallingKnots = 20
        falling = []
        for i in range(fallingKnots):
            falling += [self.createSwingFootModel(
                timeStep=timeStep,
                supportFootIds=[],
                comTask=None,
                swingFootTask=None,
                frameRotationTask=None,
                frameTranslationTask=None,
            )]

        footTask = [
            [self.lfFootId, pinocchio.SE3(
                np.eye(3), lfFootPos0 - np.array([4*self.half + delta, 0.0, 0.0]))],
            [self.rfFootId, pinocchio.SE3(
                np.eye(3), rfFootPos0 - np.array([4*self.half + delta, 0.0, 0.0]))],
            [self.lhFootId, pinocchio.SE3(
                np.eye(3), lhFootPos0 - np.array([delta, 0.0, 0.0]))],
            [self.rhFootId, pinocchio.SE3(
                np.eye(3), rhFootPos0 - np.array([delta, 0.0, 0.0]))]
        ]

        landed = [self.createSwingFootModel(
            timeStep=timeStep,
            supportFootIds=[self.lfFootId, self.rfFootId,
                            self.lhFootId, self.rhFootId],
            comTask=None,
            swingFootTask=footTask,
            frameRotationTask=rotMatY(-np.pi),
            frameTranslationTask=np.array(
                [-2*self.half - delta, 0.0, basePos0[2]]),
        )]

        loco3dModel += spring
        loco3dModel += pushing
        loco3dModel += rising
        loco3dModel += atTheTop
        loco3dModel += falling
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None, frameRotationTask=None, frameTranslationTask=None):
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(self.state, i, np.array([0., 0., 0.]), nu,
                                                           np.array([0., 50.]))
            contactModel.addContact(
                self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(
                self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, 1e3)
        if isinstance(frameRotationTask, np.ndarray):
            frameRotResidual = crocoddyl.ResidualModelFrameRotation(
                self.state, self.baseFrameId, frameRotationTask, nu)
            frameRotTrack = crocoddyl.CostModelResidual(
                self.state, frameRotResidual)
            costModel.addCost("frameRotTrack", frameRotTrack, 1e6)
        if isinstance(frameTranslationTask, np.ndarray):
            frameTransResidual = crocoddyl.ResidualModelFrameTranslation(
                self.state, self.baseFrameId, frameTranslationTask, nu)
            frameTransTrack = crocoddyl.CostModelResidual(
                self.state, frameTransResidual)
            costModel.addCost("frameTransTrack", frameTransTrack, 1e6)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, i, cone, nu)
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub))
            frictionCone = crocoddyl.CostModelResidual(
                self.state, coneActivation, coneResidual)
            costModel.addCost(
                self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,
                                                                                   nu)
                footTrack = crocoddyl.CostModelResidual(
                    self.state, frameTranslationResidual)
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6)

        stateWeights = np.array(
            [0.0] * 6 + [5.0, 1.0, 1.0] * 4 + [0.0] * 6 + [5.0, 1.0, 1.0] * 4)
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(
            stateWeights**2)
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model


def plotSolution(solver, figIndex=1, figTitle="", show=True):
    import matplotlib.pyplot as plt
    xs, us = [], []
    rmodel = solver.problem.runningModels[0].state.pinocchio
    xs, us = solver.xs, solver.us

    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu

    for i in range(nx):
        X[i] = [np.asscalar(x[i]) for x in xs]

    for i in range(nu):
        U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]

    # Plotting the joint positions, velocities and torques
    plt.figure(figIndex)
    plt.suptitle(figTitle)
    legJointNames = ['HAA', 'HFE', 'KFE']

    # LF foot
    plt.subplot(4, 3, 1)
    plt.title('joint position [rad]')
    [plt.plot(X[k], label=legJointNames[i])
     for i, k in enumerate(range(7, 10))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 2)
    plt.title('joint velocity [rad/s]')
    [plt.plot(X[k], label=legJointNames[i])
     for i, k in enumerate(range(nq + 6, nq + 9))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 3)
    plt.title('joint torque [Nm]')
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 3))]
    plt.ylabel('LF')
    plt.legend()

    # LH foot
    plt.subplot(4, 3, 4)
    [plt.plot(X[k], label=legJointNames[i])
     for i, k in enumerate(range(10, 13))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 5)
    [plt.plot(X[k], label=legJointNames[i])
     for i, k in enumerate(range(nq + 9, nq + 12))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(3, 6))]
    plt.ylabel('LH')
    plt.legend()

    # RF foot
    plt.subplot(4, 3, 7)
    [plt.plot(X[k], label=legJointNames[i])
     for i, k in enumerate(range(13, 16))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 8)
    [plt.plot(X[k], label=legJointNames[i])
     for i, k in enumerate(range(nq + 12, nq + 15))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 9)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 9))]
    plt.ylabel('RF')
    plt.legend()

    # RH foot
    plt.subplot(4, 3, 10)
    [plt.plot(X[k], label=legJointNames[i])
     for i, k in enumerate(range(16, 19))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 11)
    [plt.plot(X[k], label=legJointNames[i])
     for i, k in enumerate(range(nq + 15, nq + 18))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 12)
    [plt.plot(U[k], label=legJointNames[i])
     for i, k in enumerate(range(9, 12))]
    plt.ylabel('RH')
    plt.legend()
    plt.xlabel('knots')

    plt.figure(figIndex + 1)
    plt.suptitle(figTitle)
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    for x in xs:
        q = x[:nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
    plt.plot(Cx, Cy)
    plt.title('CoM position')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    if show:
        plt.show()
