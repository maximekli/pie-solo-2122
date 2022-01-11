
import crocoddyl
import pinocchio
import numpy as np
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem
import example_robot_data

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
    return R_z.dot(R_y.dot(R_x))

class SimpleQuadrupedalGaitProblem(SimpleQuadrupedalGaitProblem):
    def createMySwingFootModel(self, timeStep, supportFootIds, comTask=None, rotationTask=None, swingFootTask=None):
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
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, 1e3)
        if isinstance(rotationTask, np.ndarray):
            rotationResidual = crocoddyl.ResidualModelFrameRotation(self.state,
                                                                    self.rmodel.getFrameId("base_link"),
                                                                    rotationTask,
                                                                    nu)
            rotationTrack = crocoddyl.CostModelResidual(self.state, rotationResidual)
            costModel.addCost("rotationTrack", rotationTrack, 1e6)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone, nu)
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation( self.state,
                                                                                    i[0],
                                                                                    i[1].translation,
                                                                                    nu)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6)

        # stateWeights = np.array([10.] * self.rmodel.nv + [10.] * self.rmodel.nv)
        stateWeights = np.array([1.] * 3 + [1.] * 3 + [1.] * (self.rmodel.nv - 6) + [100.] * 6 + [100.] * (self.rmodel.nv - 6))
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e-3)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
        stateBounds = crocoddyl.CostModelResidual(self.state, stateBoundsActivation, stateBoundsResidual)
        costModel.addCost("stateBounds", stateBounds, 1e3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model

    def createMyImpulseModel(self, supportFootIds, swingFootTask=None, JMinvJt_damping=1e-12, r_coeff=0.0):
        """ Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel3D(self.state, i)
            impulseModel.addImpulse(self.rmodel.frames[i].name + "_impulse", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                # framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                #     self.state,
                #     i[0],
                #     i[1],
                #     0)
                # placTrack = crocoddyl.CostModelResidual(self.state, framePlacementResidual)
                # costModel.addCost(self.rmodel.frames[i[0]].name + "_placTrack", placTrack, 1e8)

                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state,
                    i[0],
                    i[1].translation,
                    0)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_transTrack", footTrack, 1e7)

                # frameRotationResidual = crocoddyl.ResidualModelFrameRotation(
                #     self.state,
                #     i[0],
                #     i[1].rotation,
                #     0)
                # rotationTrack = crocoddyl.CostModelResidual(self.state, frameRotationResidual)
                # costModel.addCost(self.rmodel.frames[i[0]].name + "_rotTrack", rotationTrack, 1e8)

        stateWeights = np.array([1.] * self.rmodel.nv + [1.] * self.rmodel.nv)
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, 0)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        costModel.addCost("stateReg", stateReg, 1e-3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel)
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model
    
    def createMyJumpingProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        df = jumpLength[2] - rfFootPos0[2]
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)

        loco3dModel = []
        takeOff = [
            self.createMySwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(groundKnots)
        ]

        comTask = np.array([jumpLength[0], jumpLength[1], jumpLength[2] + jumpHeight])
        flyingUpPhase = [
            self.createMySwingFootModel(
                timeStep,
                [], # No support foot
                comTask=comTask * (k + 1) / flyingKnots + comRef,
                rotationTask=None
            ) for k in range(flyingKnots)
        ]
        flyingDownPhase = [
            self.createMySwingFootModel(
                timeStep,
                [], # No support foot
                comTask=None,
                rotationTask=None
            ) for k in range(flyingKnots)
        ]

        f0 = jumpLength
        framesId = [self.lfFootId,
                    self.rfFootId,
                    self.lhFootId,
                    self.rhFootId]
        Pos0 = np.array([self.rdata.oMf[frameId].translation for frameId in framesId])
        Pos1 = np.array([Pos0[i] + f0 for i, frameId in enumerate(framesId)])
        Rot0 = np.array([self.rdata.oMf[frameId].rotation for frameId in framesId])
        footTask = [[framesId[i], pinocchio.SE3(Rot0[i], Pos1[i])] for i in range(len(framesId))]
        landingPhase = [
            self.createMyImpulseModel(
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                swingFootTask=footTask
            )
        ]
        f0[2] = df
        landed = [
            self.createMySwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                comTask=comRef + f0,
                rotationTask=None
            ) for k in range(groundKnots)
        ]

        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createTurnProblem(self, x0, jumpHeight, jumpRot, timeStep, groundKnots, flyingKnots):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        df = [0,0,rfFootPos0[2]]
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        comRef = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)
        rotationRef = self.rdata.oMf[self.rmodel.getFrameId("base_link")].rotation

        ''' TAKE OFF PHASE '''
        takeOff = [
            self.createMySwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(groundKnots)
        ]

        ''' FLYING UP PHASE '''
        comTask = np.array([0, 0, jumpHeight])
        rotationTask = rotation_matrix(jumpRot, 0, 0).dot(rotationRef)
        flyingUpPhase = [
            self.createMySwingFootModel(
                timeStep,
                [], # No support foot
                comTask=comTask * ((-4/(flyingKnots-1)**2)*k**2+ (4/(flyingKnots-1))*k) + comRef,
                rotationTask=rotation_matrix(jumpRot*(k+1)/(2*flyingKnots), 0, 0).dot(rotationRef)
            ) for k in range(flyingKnots)
        ]

        ''' FLYING DOWN PHASE '''
        flyingDownPhase = [
            self.createMySwingFootModel(
                timeStep,
                [],
                comTask=None,
                rotationTask=rotation_matrix(jumpRot*(k+flyingKnots+1)/(2*flyingKnots), 0, 0).dot(rotationRef)
            ) for k in range(flyingKnots)
        ]
        
        ''' LANDING PHASE '''
        framesId = [self.lfFootId,
                    self.rfFootId,
                    self.lhFootId,
                    self.rhFootId]
        # framesName = [frame.name for frame in self.rmodel.frames.tolist()]
        # framesId = [self.rmodel.getFrameId(frameName) for frameName in framesName]

        Pos0 = np.array([self.rdata.oMf[frameId].translation for frameId in framesId])
        Pos1 = np.array([rotationTask.dot(Pos0[i]-comRef)+comRef for i, frameId in enumerate(framesId)])
        Rot0 = np.array([self.rdata.oMf[frameId].rotation for frameId in framesId])
        Rot1 = np.array([rotationTask.dot(Rot0[i]) for i, frameId in enumerate(framesId)])
        PosTask = [[framesId[i], pinocchio.SE3(Rot1[i], Pos1[i])] for i in range(len(framesId))]

        landingPhase = [
            self.createMyImpulseModel(
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                swingFootTask=PosTask
            )
        ]

        ''' LANDED PHASE '''
        landed = [
            self.createMySwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                comTask=None, # comRef,
                rotationTask=rotationTask
            ) for k in range(groundKnots)
        ]

        loco3dModel = []
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createSaltoProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        df = [0,0,rfFootPos0[2]]
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        comRef = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)
        rotationRef = self.rdata.oMf[self.rmodel.getFrameId("base_link")].rotation

        rot = np.pi
        ''' TAKE OFF PHASE '''
        takeOff = [
            self.createMySwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(groundKnots)
        ]
        ''' TAKE OFF PHASE 2 '''
        takeOff2 = [
            self.createMySwingFootModel(
                timeStep,
                [self.lhFootId, self.rhFootId],
            ) for k in range(groundKnots)
        ]

        ''' FLYING UP PHASE '''
        comTask = np.array([jumpLength[0], jumpLength[1], jumpLength[2] + jumpHeight])
        flyingUpPhase = [
            self.createMySwingFootModel(
                timeStep,
                [], # No support foot
                comTask=comTask * (k+1)/flyingKnots + comRef,
                rotationTask=rotation_matrix(0, rot*(k+1)/(2*flyingKnots), 0).dot(rotationRef)
            ) for k in range(flyingKnots)
        ]

        ''' FLYING DOWN PHASE '''
        comTask = np.array(jumpLength)
        flyingDownPhase = [
            self.createMySwingFootModel(
                timeStep,
                [],
                comTask=comTask * (k+1)/flyingKnots + comRef,
                rotationTask=rotation_matrix(0, rot*(k+flyingKnots+1)/(2*flyingKnots), 0).dot(rotationRef)
            ) for k in range(flyingKnots)
        ]
        
        ''' LANDING PHASE '''
        framesId = [self.lfFootId,
                    self.rfFootId,
                    self.lhFootId,
                    self.rhFootId]
        Pos0 = np.array([self.rdata.oMf[frameId].translation for frameId in framesId])
        Pos1 = np.array([rotation_matrix(0, rot, 0).dot(Pos0[i]) + jumpLength for i, frameId in enumerate(framesId)])
        Rot0 = np.array([self.rdata.oMf[frameId].rotation for frameId in framesId])
        PosTask = [[framesId[i], pinocchio.SE3(Rot0[i], Pos1[i])] for i in range(len(framesId))]
        # Pos0 = np.array([self.rdata.oMf[frameId].translation for frameId in framesId])
        # Pos1 = np.array([Pos0[i] + jumpLength for i, frameId in enumerate(framesId)])
        # Rot0 = np.array([self.rdata.oMf[frameId].rotation for frameId in framesId])
        # PosTask = [[framesId[i], pinocchio.SE3(Rot0[i], Pos1[i])] for i in range(len(framesId))]

        landingPhase = [
            self.createMyImpulseModel(
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                swingFootTask=PosTask
            )
        ]

        ''' LANDED PHASE '''
        landed = [
            self.createMySwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                comTask=comTask+comRef,
                rotationTask=None
            ) for k in range(groundKnots)
        ]

        loco3dModel = []
        loco3dModel += takeOff
        loco3dModel += takeOff2
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem