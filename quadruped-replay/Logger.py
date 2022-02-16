'''This class will log 1d array in Nd matrix from device and qualisys object'''
import numpy as np
from datetime import datetime as datetime
from time import time
import pinocchio as pin

class Logger():
    def __init__(self, device=None, qualisys=None, logSize=60e3):
        logSize = np.int(logSize)
        self.logSize = logSize
        nb_motors = 12
        self.k = 0

        # Allocate the data:
        # IMU and actuators:
        self.q_mes = np.zeros([logSize, nb_motors])
        self.v_mes = np.zeros([logSize, nb_motors])
        self.torquesFromCurrentMeasurment = np.zeros([logSize, nb_motors])
        self.baseOrientation = np.zeros([logSize, 3])
        self.baseOrientationQuat = np.zeros([logSize, 4])
        self.baseAngularVelocity = np.zeros([logSize, 3])
        self.baseLinearAcceleration = np.zeros([logSize, 3])
        self.baseAccelerometer = np.zeros([logSize, 3])
        self.current = np.zeros(logSize)
        self.voltage = np.zeros(logSize)
        self.energy = np.zeros(logSize)
        self.q_des = np.zeros([logSize, nb_motors])
        self.v_des = np.zeros([logSize, nb_motors])

        # Motion capture:
        self.mocapPosition = np.zeros([logSize, 3])
        self.mocapVelocity = np.zeros([logSize, 3])
        self.mocapAngularVelocity = np.zeros([logSize, 3])
        self.mocapOrientationMat9 = np.zeros([logSize, 3, 3])
        self.mocapOrientationQuat = np.zeros([logSize, 4])

        # Timestamps
        self.tstamps = np.zeros(logSize)

    def sample(self, device, qualisys=None):

        # Logging from the device (data coming from the robot)
        self.q_mes[self.k] = device.joints.positions
        self.v_mes[self.k] = device.joints.velocities
        self.baseOrientation[self.k] = device.imu.attitude_euler
        self.baseOrientationQuat[self.k] = device.imu.attitude_quaternion
        self.baseAngularVelocity[self.k] = device.imu.gyroscope
        self.baseLinearAcceleration[self.k] = device.imu.linear_acceleration
        self.baseAccelerometer[self.k] = device.imu.accelerometer
        self.torquesFromCurrentMeasurment[self.k] = device.joints.measured_torques
        self.current[self.k] = device.powerboard.current
        self.voltage[self.k] = device.powerboard.voltage
        self.energy[self.k] = device.powerboard.energy

        # Logging from qualisys (motion capture)
        if qualisys is not None:
            self.mocapPosition[self.k] = qualisys.getPosition()
            self.mocapVelocity[self.k] = qualisys.getVelocity()
            self.mocapAngularVelocity[self.k] = qualisys.getAngularVelocity()
            self.mocapOrientationMat9[self.k] = qualisys.getOrientationMat9()
            self.mocapOrientationQuat[self.k] = qualisys.getOrientationQuat()
        else:  # Logging from PyBullet simulator through fake device
            self.mocapPosition[self.k] = device.baseState[0]
            self.mocapVelocity[self.k] = device.baseVel[0]
            self.mocapAngularVelocity[self.k] = device.baseVel[1]
            self.mocapOrientationMat9[self.k] = device.rot_oMb
            self.mocapOrientationQuat[self.k] = device.baseState[1]

        # Logging timestamp
        self.tstamps[self.k] = time()

        self.k += 1

    def saveAll(self, fileName="dataSensors"):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')

        np.savez_compressed(fileName + date_str + ".npz",
                            q_mes=self.q_mes,
                            v_mes=self.v_mes,
                            baseOrientation=self.baseOrientation,
                            baseOrientationQuat=self.baseOrientationQuat,
                            baseAngularVelocity=self.baseAngularVelocity,
                            baseLinearAcceleration=self.baseLinearAcceleration,
                            baseAccelerometer=self.baseAccelerometer,
                            torquesFromCurrentMeasurment=self.torquesFromCurrentMeasurment,
                            current=self.current,
                            voltage=self.voltage,
                            energy=self.energy,
                            mocapPosition=self.mocapPosition,
                            mocapVelocity=self.mocapVelocity,
                            mocapAngularVelocity=self.mocapAngularVelocity,
                            mocapOrientationMat9=self.mocapOrientationMat9,
                            mocapOrientationQuat=self.mocapOrientationQuat,
                            tstamps=self.tstamps)

    def processMocap(self):

        self.mocap_pos = np.zeros([self.logSize, 3])
        self.mocap_h_v = np.zeros([self.logSize, 3])
        self.mocap_b_w = np.zeros([self.logSize, 3])
        self.mocap_RPY = np.zeros([self.logSize, 3])
   
        for i in range(self.logSize):
            self.mocap_RPY[i] = pin.rpy.matrixToRpy(pin.Quaternion(self.mocapOrientationQuat[i]).toRotationMatrix())

        # Robot world to Mocap initial translation and rotation
        mTo = np.array([self.mocapPosition[0, 0], self.mocapPosition[0, 1], 0.0])  
        mRo = pin.rpy.rpyToMatrix(0.0, 0.0, self.mocap_RPY[0, 2])

        for i in range(self.logSize):
            oRb = self.mocapOrientationMat9[i]

            oRh = pin.rpy.rpyToMatrix(0.0, 0.0, self.mocap_RPY[i, 2] - self.mocap_RPY[0, 2])

            self.mocap_h_v[i] = (oRh.transpose() @ mRo.transpose() @ self.mocapVelocity[i].reshape((3, 1))).ravel()
            self.mocap_b_w[i] = (oRb.transpose() @ self.mocapAngularVelocity[i].reshape((3, 1))).ravel()
            self.mocap_pos[i] = (mRo.transpose() @ (self.mocapPosition[i, :] - mTo).reshape((3, 1))).ravel()

    def custom_suptitle(self, name):
        from matplotlib import pyplot as plt

        fig = plt.gcf()
        fig.suptitle(name)
        fig.canvas.manager.set_window_title(name)

    def plotAll(self, params, replay):
        from matplotlib import pyplot as plt

        t_range = np.array([k*params.dt for k in range(self.logSize)])

        self.processMocap()

        index6 = [1, 3, 5, 2, 4, 6]
        index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

        ####
        # Desired & Measured actuator positions
        ####
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            h1, = plt.plot(t_range, replay.q[:, i], color='r', linewidth=3)
            h2, = plt.plot(t_range, self.q_mes[:, i], color='b', linewidth=3)
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [rad]")
            plt.legend([h1, h2], ["Ref "+lgd1[i % 3]+" "+lgd2[int(i/3)],
                                  lgd1[i % 3]+" "+lgd2[int(i/3)]], prop={'size': 8})
        self.custom_suptitle("Actuator positions")

        ####
        # Desired & Measured actuator velocity
        ####
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            h1, = plt.plot(t_range, replay.v[:, i], color='r', linewidth=3)
            h2, = plt.plot(t_range, self.v_mes[:, i], color='b', linewidth=3)
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [rad]")
            plt.legend([h1, h2], ["Ref "+lgd1[i % 3]+" "+lgd2[int(i/3)],
                                  lgd1[i % 3]+" "+lgd2[int(i/3)]], prop={'size': 8})
        self.custom_suptitle("Actuator velocities")

        ####
        # FF torques & FB torques & Sent torques & Meas torques
        ####
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            tau_fb = replay.P[:, i] * (replay.q[:, i] - self.q_mes[:, 6]) + \
                replay.D[:, i] * (replay.v[:, i] - self.v_mes[:, i])
            h1, = plt.plot(t_range, replay.FF[:, i] * replay.tau[:, i], "r", linewidth=3)
            h2, = plt.plot(t_range, tau_fb, "b", linewidth=3)
            h3, = plt.plot(t_range, replay.FF[:, i] * replay.tau[:, i] + tau_fb, "g", linewidth=3)
            h4, = plt.plot(t_range[:-1], self.torquesFromCurrentMeasurment[1:, i],
                           "violet", linewidth=3, linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [Nm]")
            tmp = lgd1[i % 3]+" "+lgd2[int(i/3)]
            plt.legend([h1, h2, h3, h4], ["FF "+tmp, "FB "+tmp, "PD+ "+tmp, "Meas "+tmp], prop={'size': 8})
            plt.ylim([-8.0, 8.0])
        self.custom_suptitle("Torques")

        ####
        # Power supply profile
        ####
        
        plt.figure()
        for i in range(3):
            if i == 0:
                ax0 = plt.subplot(3, 1, i+1)
            else:
                plt.subplot(3, 1, i+1, sharex=ax0)

            if i == 0:
                plt.plot(t_range, self.current[:], linewidth=2)
                plt.ylabel("Bus current [A]")
            elif i == 1:
                plt.plot(t_range, self.voltage[:], linewidth=2)
                plt.ylabel("Bus voltage [V]")
            else:
                plt.plot(t_range, self.energy[:], linewidth=2)
                plt.ylabel("Bus energy [J]")
                plt.xlabel("Time [s]")
        self.custom_suptitle("Energy profiles")

        ####
        # Measured & Reference position and orientation (ideal world frame)
        ####
        lgd = ["Pos X", "Pos Y", "Pos Z", "Roll", "Pitch", "Yaw"]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)

            if i < 3:
                plt.plot(t_range, self.mocap_pos[:, i], "k", linewidth=3)
            else:
                plt.plot(t_range, self.mocap_RPY[:, i-3], "k", linewidth=3)
            plt.legend(["Motion capture"], prop={'size': 8})
            plt.ylabel(lgd[i])
        self.custom_suptitle("Position and orientation")

        ####
        # Measured & Reference linear and angular velocities (horizontal frame)
        ####
        lgd = ["Linear vel X", "Linear vel Y", "Linear vel Z",
               "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)

            if i < 3:
                plt.plot(t_range, self.mocap_h_v[:, i], "k", linewidth=3)
            else:
                plt.plot(t_range, self.mocap_b_w[:, i-3], "k", linewidth=3)
            
            plt.legend(["Motion capture"], prop={'size': 8})
            plt.ylabel(lgd[i])
        self.custom_suptitle("Linear and angular velocities")

        ###############################
        # Display all graphs and wait #
        ###############################
        plt.show(block=True)