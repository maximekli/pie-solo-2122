import os
import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

path = os.path.dirname(os.path.abspath(__file__))

class SoloEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file= path + "/../models/solo_description/robots/solo12.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=False,
        healthy_z_range=(-.1, float("inf")),
        healthy_angle_range=(-3.15,3.15), #TBD
        healthy_angle_vel_range=(-float("inf"),float("inf")), #TBD
        reset_noise_scale=5e-2,
        exclude_current_positions_from_observation=True,
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._healthy_angle_vel_range = healthy_angle_vel_range

        self.lift_flag = False

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.init_qpos = 0
        self.init_qvel = 0

        mujoco_env.MujocoEnv.__init__(self, xml_file, 4)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action): #TBD
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        """
            qpos =  x,y,z,
                    quaternion for body orientation,
                    joints angles,
            qvel =  vx,vy,vz,
                    roll, pitch, yaw,
                    joints angular velocity
        """
        pos_body = self.sim.data.qpos[:3]
        # orientation = self.sim.data.qpos[3:7]
        pos_joints = self.sim.data.qpos[7:]
        # vel_body = self.sim.data.qvel[:3]
        # rot_body = self.sim.data.qvel[3:6]
        vel_joints = self.sim.data.qvel[6:]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        min_angle_vel, max_angle_vel = self._healthy_angle_vel_range

        healthy_z = min_z < pos_body[-1] < max_z
        healthy_joints_pos = np.all(np.logical_and(min_angle < pos_joints, pos_joints < max_angle))
        healthy_joints_vel = np.all(np.logical_and(min_angle_vel < vel_joints, vel_joints < max_angle_vel))

        if False:
            print("healthy_z: ", healthy_z)
            print("healthy_joints_pos: ", healthy_joints_pos)
            print("healthy_joints_vel: ", healthy_joints_vel)

        is_healthy = all((healthy_joints_pos, healthy_z,healthy_joints_vel))

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        """
        observation =   [
                        pos with or without current position,
                        vel(clipped at -10, 10)
                        ]
        """
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def _compute_reward(self):
        # four points of contact with ground
        feet_on_ground_reward = 0
        #if self.sim.data.ncon == 4 :
        #    feet_on_ground_reward = 10

        # proportionnal reward with z-position and liftoff
        jump_reward = 0
        z = self.sim.data.qpos[2]
        if self.sim.data.ncon == 0:
            if self.lift_flag :
                jump_reward += z*5
            else :
                self.lift_flag = True
                jump_reward += 100
        
        # zvel_reward = self.sim.data.qvel[2]>0 * 100

        rewards = feet_on_ground_reward + jump_reward #+ zvel_reward
        return rewards


    def step(self, action):
        #self.frame_skip = 50
        self.do_simulation(action, self.frame_skip)
        self.render()

        ctrl_cost = self.control_cost(action)
        rewards = self._compute_reward()
        costs = 0 #TBD

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = 0

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        self.lift_flag = False

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
