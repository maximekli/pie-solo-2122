import os
import math
import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

path = os.path.dirname(os.path.abspath(__file__))

class SoloEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file= path + "/../models/solo_description/robots/solo12.xml",
        ctrl_cost_weight=1e-3,
        healthy_reward=0.1,
        terminate_when_unhealthy=True,
        healthy_state_range=(-1000.0, 1000.0),
        healthy_z_range=(-0.3, float("inf")),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=False,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range

        #self._healthy_angle_range = healthy_angle_range
        #self._healthy_angle_vel_range = healthy_angle_vel_range

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

    @property
    def is_healthy(self):
        """
            qpos =  x linear position,z linear position,y angular position,
                    joints angles,
            qvel =  vx linear velocity,vz linear velocity,vy angular velocity,
                    joints angular velocity
        """
        #pos_body = self.sim.data.qpos[:3]
        #pos_joints = self.sim.data.qpos[3:]
        #vel_body = self.sim.data.qvel[:3]
        #vel_joints = self.sim.data.qvel[3:]

        state = self.state_vector()
        #print(state)

        min_z, max_z = self._healthy_z_range
        min_state, max_state = self._healthy_state_range

        #min_angle, max_angle = self._healthy_angle_range
        #min_angle_vel, max_angle_vel = self._healthy_angle_vel_range
        z = self.sim.data.qpos[1]
        healthy_z = min_z < self.sim.data.qpos[1] < max_z
        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))

        #healthy_joints_pos = np.all(np.logical_and(min_angle < pos_joints, pos_joints < max_angle))
        #healthy_joints_vel = np.all(np.logical_and(min_angle_vel < vel_joints, vel_joints < max_angle_vel))

        #print(z)
        #if healthy_z :
            #print("healthy_10z: ", healthy_z)
            #print("healthy_joints_pos: ", healthy_joints_pos)
            #print("healthy_joints_vel: ", healthy_joints_vel)

        penetration_margin = - 0.5
        penetration_state = [self.sim.data.contact[i].dist > penetration_margin for i in range(self.sim.data.ncon)]

        is_healthy = all((healthy_state, healthy_z, all(penetration_state)))



        return is_healthy

    @property
    def done(self):
        #print("Is unhealthy?", self.is_healthy)
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

    def _euler_from_quaternion(self,q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w,x,y,z = q
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z
    
    def _print_floor_contacts(self):
        print('number of contacts', self.sim.data.ncon)
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1_name = self.sim.model.geom_id2name(contact.geom1)
            geom2_name = self.sim.model.geom_id2name(contact.geom2)
            if (geom1_name == 'floor' or geom2_name == 'floor'):
                print('contact', i)
                print('dist', contact.dist)
                print('geom1', contact.geom1, geom1_name)
                print('geom2', contact.geom2, geom2_name)
                geom1_body = self.sim.model.geom_bodyid[self.sim.data.contact[i].geom1]
                print(' Contact force on geom1 body', self.sim.data.cfrc_ext[geom1_body])
        

    def _compute_reward(self):
        jump_reward = 0

        ## REWARD BASED ON THE PITCH ANGLE OF THE BASE 
        backroll_angle = - self.sim.data.qpos[2]  # rooty - angle (rad)
        backroll = - self.sim.data.qvel[2]  # rooty - angular velocity (rad/s)
        height = self.sim.data.qpos[1]  # rootz - position (m)
        backslide = - self.sim.data.qvel[0]  # rootx - velocity (m/s)
        backslide_position = - self.sim.data.qpos[0]  # rootx - position (m)

        # Deep Mind Reward
        # jump_reward = backroll * (1.0 + .3 * height + .05 * backslide)

        # Backflip Reward
        if height >0:
            jump_reward = 30 * (backroll_angle * height + backroll * self.sim.data.qvel[1])
        if backroll_angle == 2*np.pi:
            jump_reward += 100

        rewards = jump_reward + self.healthy_reward
        return rewards


    def step(self, action):

        self.do_simulation(action, self.frame_skip)
        self.render()

        reward = self._compute_reward()
        observation = self._get_obs()
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

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
