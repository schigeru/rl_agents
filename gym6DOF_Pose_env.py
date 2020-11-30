#!/usr/bin/env python3

import math
import numpy as np
import os
import time
import sys
import copy
import rospy
import moveit_msgs.msg
import geometry_msgs.msg
import random
import csv
import gym
from gym import spaces
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.msg import LinkState
from std_msgs.msg import Float64
from std_msgs.msg import String
from panda_rl.srv import ResetJoints, ResetJointsResponse
from panda_rl.srv import StepAction, StepActionResponse

metadata = {'render.modes': ['human']}


class PandaRobotGymEnv(gym.Env):

    def __init__(self, max_steps=50):
        super(PandaRobotGymEnv, self).__init__()
        self.stepnode = rospy.ServiceProxy('step_env', StepAction, persistent=True)
        self.res = rospy.ServiceProxy('reset_env', ResetJoints, persistent=True)
        self._env_step_counter = 0
        self.done = False
        self._max_steps = max_steps
        self.observation_space = spaces.Box(np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -2, -2, -2]),
                                            np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 2.8973, 3.7525, 2, 2, 2]))
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1, -1, -1]), np.array([+1, +1, +1, +1, +1, +1]))
        self.goal = [0, 0, 0]

    def reset(self):
        rospy.wait_for_service("reset_env")
        try:
            #goal_pool = [[0.417, 0,  0.479], [0.447, 0, 0.559], [0.377, 0, 0.663], [0.595, 0,  0.32],
            #             [0.303, 0, 0.353], [0.147, 0,  0.599], [-0.445,  0,  0.778],
            #             [-0.254, 0, 0.824]]
            #pointer = np.random.randint(0, 8, 1)
            #pointer = pointer[0]
            #self.goal = goal_pool[pointer]

            goalx = random.randint(5, 15) / 20
            goaly = random.randint(5, 15) / 20
            goalz = random.randint(5, 20) / 20

            self.goal = [goalx, goaly, goalz]

            response = self.res(self.goal)
            self._env_step_counter = 0
            self.done = False
            return np.array(response.obs)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def step(self, action):
        rospy.wait_for_service("step_env")
        try:
            response = self.stepnode(action, self.goal)
            obs = response.obs
            reward = response.reward
            self.done = response.done
            self._env_step_counter += 1

            if self._env_step_counter >= self._max_steps:
                reward = 0
                self.done = True

            return np.array(obs), np.array(reward), np.array(self.done), {}
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def render(self, mode='human'):
        print(self.done, self._env_step_counter)
