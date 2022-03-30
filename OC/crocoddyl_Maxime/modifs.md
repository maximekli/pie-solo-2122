```
delatSpring = 0.2
springKnots = 10
spring = []
for i in range(springKnots):
    spring += [self.createSwingFootModel(
        timeStep=timeStep,
        supportFootIds=[self.lfFootId, self.rfFootId,
                        self.lhFootId, self.rhFootId],
        comTask=None,
        swingFootTask=None,
        frameRotationTask=None,
        frameTranslationTask=np.array(
            [0.0, 0.0, comRef[2] - delatSpring*(i + 1)/springKnots]),
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
        # supportFootIds=[self.lhFootId, self.rhFootId],
        supportFootIds=[],
        comTask=None,
        swingFootTask=None,
        frameRotationTask=None,
        frameTranslationTask=None,
    )]

delta = 0.2
atTheTop = [
    self.createSwingFootModel(
        timeStep=timeStep,
        # supportFootIds=[self.lhFootId, self.rhFootId],
        supportFootIds=[],
        comTask=None,
        swingFootTask=None,
        frameRotationTask=rotMat(-np.pi/2),
        # frameRotationTask=None,
        frameTranslationTask=np.array([-self.half, 0.0, 0.4]),
        # frameTranslationTask=np.array(
        #    [-self.half - delta/2, 0.0, comRef[2]+0.4]),
    )
]

fallingKnots = 20
falling = []
for i in range(fallingKnots):
    falling += [self.createSwingFootModel(
        timeStep=timeStep,
        # supportFootIds=[self.lhFootId, self.rhFootId],
        supportFootIds=[],
        comTask=None,
        swingFootTask=None,
        frameRotationTask=None,
        frameTranslationTask=None,
    )]

landingKnots = 5
landing = []
for i in range(landingKnots):
    landing += [self.createSwingFootModel(
        timeStep=timeStep,
        supportFootIds=[self.lfFootId, self.rfFootId,
                        self.lhFootId, self.rhFootId],
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
    frameRotationTask=rotMat(-np.pi),
    frameTranslationTask=np.array(
        [-2*self.half - delta, 0.0, comRef[2]]),
)]

loco3dModel += spring
loco3dModel += pushing
loco3dModel += rising
loco3dModel += atTheTop
loco3dModel += falling
loco3dModel += landing
loco3dModel += landed
```

```
def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None, frameRotationTask=None, frameTranslationTask=None):
    """ Action model for a swing foot phase.

    :param timeStep: step duration of the action model
    :param supportFootIds: Ids of the constrained feet
    :param comTask: CoM task
    :param swingFootTask: swinging foot task
    :return action model for a swing foot phase
    """
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

    # stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10.] * 6 + [1.] *
    #                        (self.rmodel.nv - 6))
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

    lb = np.concatenate(
        [self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
    ub = np.concatenate(
        [self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
    stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
    stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(lb, ub))
    stateBounds = crocoddyl.CostModelResidual(
        self.state, stateBoundsActivation, stateBoundsResidual)
    costModel.addCost("stateBounds", stateBounds, 1e3)

    # Creating the action model for the KKT dynamics with simpletic Euler
    # integration scheme
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                    costModel, 0., True)
    model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
    return model
```