import numpy as np
from math import cos, sin, pi
import time
import random
import gym
from gym import spaces


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
        self.goal = [0, 0, 0]
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
        #self.joint0_matrix = {"a": 0, "d": 0, "alpha": 0}
        self.joint1_matrix = {"a": 0, "d": 0.333, "alpha": 0}
        self.joint2_matrix = {"a": 0, "d": 0, "alpha": -pi/2}
        self.joint3_matrix = {"a": 0, "d": 0.316, "alpha": pi/2}
        self.joint4_matrix = {"a": 0.0825, "d": 0, "alpha": pi/2}
        self.joint5_matrix = {"a": -0.0825, "d": 0.384, "alpha": -pi/2}
        self.joint6_matrix = {"a": 0, "d": 0, "alpha": pi/2}
        self.joint7_matrix = {"a": 0.088, "d": 0, "alpha": pi/2}

    def set_joint_state(self):
        self.joint_state = np.array([self.joint1, self.joint2, self.joint3, self.joint4,
                                    self.joint5, self.joint6, self.joint7])


    def calc_handpos(self):
        trans1 = np.array([[cos(self.joint_state[0]), -sin(self.joint_state[0]), 0, 0],
                           [sin(self.joint_state[0]), cos(self.joint_state[0]), 0, 0],
                           [0, 0, 1, 0.333],
                           [0, 0, 0, 1]])
        trans2 = np.array([[cos(self.joint_state[1]), -sin(self.joint_state[1]), 0, 0],
                           [0, 0, 1, 0],
                           [-sin(self.joint_state[1]), -cos(self.joint_state[1]), 0, 0],
                           [0, 0, 0, 1]])
        trans3 = np.array([[cos(self.joint_state[2]), -sin(self.joint_state[2]), 0, 0],
                           [0, 0, -1, -0.316],
                           [sin(self.joint_state[2]), cos(self.joint_state[2]), 0, 0],
                           [0, 0, 0, 1]])
        trans4 = np.array([[cos(self.joint_state[3]), -sin(self.joint_state[3]), 0, 0.0825],
                           [0, 0, -1, 0],
                           [sin(self.joint_state[3]), cos(self.joint_state[3]), 0, 0],
                           [0, 0, 0, 1]])
        trans5 = np.array([[cos(self.joint_state[4]), -sin(self.joint_state[4]), 0, -0.0825],
                           [0, 0, 1, 0.384],
                           [-sin(self.joint_state[4]), -cos(self.joint_state[4]), 0, 0],
                           [0, 0, 0, 1]])
        trans6 = np.array([[cos(self.joint_state[5]), -sin(self.joint_state[5]), 0, 0],
                           [0, 0, -1, 0],
                           [sin(self.joint_state[5]), cos(self.joint_state[5]), 0, 0],
                           [0, 0, 0, 1]])
        trans7 = np.array([[cos(self.joint_state[6]), -sin(self.joint_state[6]), 0, 0.088],
                           [0, 0, -1, 0],
                           [sin(self.joint_state[6]), cos(self.joint_state[6]), 0, 0],
                           [0, 0, 0, 1]])
        trans8 = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0.1655],
                           [0, 0, 0, 1]])

        handpos = np.matmul(trans7, trans8)
        handpos = np.matmul(trans6, handpos)
        handpos = np.matmul(trans5, handpos)
        handpos = np.matmul(trans4, handpos)
        handpos = np.matmul(trans3, handpos)
        handpos = np.matmul(trans2, handpos)
        handpos = np.matmul(trans1, handpos)

        return [handpos[0][3], handpos[1][3], handpos[2][3]]

    def reset(self):
        #goal_pool = [[0.417, 0,  0.479], [0.447, 0, 0.559], [0.377, 0, 0.663], [0.595, 0,  0.32],
        #             [0.303, 0, 0.353], [0.147, 0,  0.599], [-0.445,  0,  0.778],
        #             [-0.254, 0, 0.824]]
        #pointer = np.random.randint(0, 8, 1)
        #pointer = pointer[0]
        #self.goal = np.array(goal_pool[pointer])
        goalx = random.randint(5, 15)/20
        goaly = random.randint(5, 15)/20
        goalz = random.randint(5, 20)/20
        goal = [goalx, goaly, goalz]
        self.goal = np.array(goal)
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
        hand_pos = self.calc_handpos()
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

        hand_pos = self.calc_handpos()
        goal_dist = np.linalg.norm(self.goal - hand_pos)
        vector_dist = self.goal - hand_pos
        obs = self.joint_state
        obs = np.append(obs, vector_dist)

        if self.joint1 < self.joint1_threshold_min or self.joint1 > self.joint1_threshold_max \
                or self.joint2 < self.joint2_threshold_min or self.joint2 > self.joint2_threshold_max \
                or self.joint3 < self.joint3_threshold_min or self.joint3 > self.joint3_threshold_max \
                or self.joint4 < self.joint4_threshold_min or self.joint4 > self.joint4_threshold_max \
                or self.joint5 < self.joint5_threshold_min or self.joint5 > self.joint5_threshold_max \
                or self.joint6 < self.joint6_threshold_min or self.joint6 > self.joint6_threshold_max:

            self.done = True
            reward = -100
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
                reward = -goal_dist
                #print("Action: ", action)
                #print("Handpos: ", hand_pos)
                #print("Goal: ", self.goal)
                #print("Observation ", obs)
                #print("reward: ", reward)
                self._env_step_counter += 1
                return obs, reward, self.done, {}


    def render(self, mode='human'):
        print(self.done, self._env_step_counter)