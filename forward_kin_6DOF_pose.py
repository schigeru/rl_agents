import numpy as np
from math import cos, sin, pi
import time
import random
import gym
from gym import spaces
import transformations as tf


metadata = {'render.modes': ['human']}


class PandaRobotGymEnv(gym.Env):

    def __init__(self, max_steps=50):
        super(PandaRobotGymEnv, self).__init__()
        self._env_step_counter = 0
        self.done = False
        self._max_steps = max_steps
        self.observation_space = spaces.Box(np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -2, -2, -2]),
                                            np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 2.8973, 3.7525, 2, 2, 2]))
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1, -1, -1]), np.array([+1, +1, +1, +1, +1, +1]))
        self.goal = np.array([0, 0, 0])
        self.quat_goal = np.array([1, 0, 0.0075, 0])
        self.joint1 = 0
        self.joint2 = -0.4
        self.joint3 = 0
        self.joint4 = -1.17
        self.joint5 = 0
        self.joint6 = 0.785
        self.joint7 = pi / 4
        self.joint1_threshold_min = -2.8973
        self.joint2_threshold_min = -1.7628
        self.joint3_threshold_min = -2.8973
        self.joint4_threshold_min = -3.0718
        self.joint5_threshold_min = -2.8973
        self.joint6_threshold_min = -0.0175
        self.joint1_threshold_max = 2.8973
        self.joint2_threshold_max = 1.7628
        self.joint3_threshold_max = 2.8973
        self.joint4_threshold_max = -0.0698
        self.joint5_threshold_max = 2.8973
        self.joint6_threshold_max = 3.7525
        self.joint_state = np.array([0, -0.4, 0, -1.17, 0, 0.785, pi / 4])
        self.alpha_vec_sin = np.array([0, sin(-pi / 2), sin(pi / 2), sin(pi / 2), sin(-pi / 2), sin(pi / 2), sin(pi / 2), 0, 0, 0, 0])
        self.alpha_vec_cos = np.array([1, cos(-pi / 2), cos(pi / 2), cos(pi / 2), cos(-pi / 2), cos(pi / 2), cos(pi / 2), 1, 1, 1, 1])
        self.a_vec = np.array([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0, 0, 0, 0])
        self.d_vec = np.array([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107, 0, 0, 0.0584])

    def set_joint_state(self):
        self.joint_state = np.array([self.joint1, self.joint2, self.joint3, self.joint4,
                                    self.joint5, self.joint6, self.joint7])

    def get_t_n(self, sin_theta, cos_theta, n):
        a = self.a_vec[n]
        d = self.d_vec[n]
        sin_alpha = self.alpha_vec_sin[n]
        cos_alpha = self.alpha_vec_cos[n]

        t_n = np.array([
            [cos_theta, -sin_theta, 0, a],
            [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -d * sin_alpha],
            [sin_theta * sin_alpha, cos_theta * sin_alpha, cos_alpha, d * cos_alpha],
            [0, 0, 0, 1]])

        return t_n

    def get_trans_matrix(self):
        theta_vec = self.joint_state
        theta_vec = np.append(theta_vec, [0, -pi / 4, 0, 0])
        sin_theta = np.sin(theta_vec)
        cos_theta = np.cos(theta_vec)
        trans_matrix = np.identity(4)

        for i in range(0, 11):
            trans_matrix = np.dot(trans_matrix, self.get_t_n(sin_theta[i], cos_theta[i], i))

        return trans_matrix

    def calc_quaternion_norm(self, quat):

        if quat[0] < 0:
            quat = quat * -1

        quat_reward = np.linalg.norm(self.quat_goal - quat)

        return quat_reward

    def reset(self):
        goalx = random.randrange(8, 13)/20
        goaly = random.randrange(-4, 5)/20
        goalz = random.randrange(6, 11)/20
        goal = np.array([goalx, goaly, goalz])
        self.goal = goal
        self._env_step_counter = 0
        self.done = False
        self.joint1 = 0
        self.joint2 = -0.4
        self.joint3 = 0
        self.joint4 = -1.17
        self.joint5 = 0
        self.joint6 = 0.785
        self.joint7 = pi / 4

        self.set_joint_state()
        trans_matrix = self.get_trans_matrix()
        hand_pos = trans_matrix[0:3, 3]
        vector_dist = self.goal - hand_pos
        obs = self.joint_state
        obs = np.append(obs, vector_dist)
        return np.array(obs)

    def step(self, action):
        self.done = False
        self.joint1 = self.joint1 + action[0] / 20
        self.joint2 = self.joint2 + action[1] / 20
        self.joint3 = self.joint3 + action[2] / 20
        self.joint4 = self.joint4 + action[3] / 20
        self.joint5 = self.joint5 + action[4] / 20
        self.joint6 = self.joint6 + action[5] / 20
        self.set_joint_state()

        trans_matrix = self.get_trans_matrix()
        hand_pos = trans_matrix[0:3, 3]
        goal_dist = np.linalg.norm(self.goal - hand_pos)
        vector_dist = self.goal - hand_pos
        obs = self.joint_state
        obs = np.append(obs, vector_dist)

        q = tf.quaternion_from_matrix(trans_matrix)
        quaternion_dist = self.calc_quaternion_norm(q)



        if self.joint1 < self.joint1_threshold_min or self.joint1 > self.joint1_threshold_max \
                or self.joint2 < self.joint2_threshold_min or self.joint2 > self.joint2_threshold_max \
                or self.joint3 < self.joint3_threshold_min or self.joint3 > self.joint3_threshold_max \
                or self.joint4 < self.joint4_threshold_min or self.joint4 > self.joint4_threshold_max \
                or self.joint5 < self.joint5_threshold_min or self.joint5 > self.joint5_threshold_max \
                or self.joint6 < self.joint6_threshold_min or self.joint6 > self.joint6_threshold_max:

            self.done = True
            reward = -50
            return obs, reward, self.done, {}

        else:

            if self._env_step_counter >= self._max_steps:
                reward = 0
                self.done = True
                return obs, reward, self.done, {}

            elif goal_dist < 0.01:
                reward = 0
                #print("Action: ", action)
                #print("Handpos: ", hand_pos)
                #print("Goal: ", self.goal)
                #print("Observation ", obs)
                #print("reward target reached: ", reward)
                self.done = True
                return obs, reward, self.done, {}

            elif hand_pos[2] < 0.01:
                #print("Gripper touched Ground")
                reward = -100
                done = True
                return obs, reward, self.done, {}

            else:
                reward = -goal_dist - quaternion_dist
                #print("Action: ", action)
                #print("Handpos: ", hand_pos)
                #print("Goal: ", self.goal)
                #print("Quaternion", q)
                #print("Observation ", obs)
                #print("quatreward", quaternion_dist)
                #print("reward: ", reward)
                self._env_step_counter += 1
                return obs, reward, self.done, {}


    def render(self, mode='human'):
        print(self.done, self._env_step_counter)