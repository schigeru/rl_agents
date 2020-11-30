#!/usr/bin/env python

import math
import os
import numpy as np
import time
import sys
import copy
import rospy
import moveit_msgs.msg
import geometry_msgs.msg
import random
import csv
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.msg import LinkState
from std_msgs.msg import Float64
from std_msgs.msg import String
from sensor_msgs.msg import Joy
import moveit_commander
from panda_rl.srv import ResetJoints, ResetJointsResponse


group_name = "panda_arm"
move_group = moveit_commander.MoveGroupCommander(group_name)


def get_hand_position():
    msg = rospy.wait_for_message('/gazebo/link_states', LinkStates)
    hand_position_x = (msg.pose[9].position.x + msg.pose[10].position.x) / 2
    hand_position_y = (msg.pose[9].position.y + msg.pose[10].position.y) / 2
    hand_position_z = (msg.pose[9].position.z + msg.pose[10].position.z) / 2
    hand_position = [hand_position_x, hand_position_y, hand_position_z]
    hand_position = np.round(hand_position, 5)
    return hand_position


def vector2points(v, u):
    v = np.array(v)
    u = np.array(u)
    vector = u - v
    vector = np.round(vector, 5)
    return vector


def reset(msg):
    joint_reset = [0, -0.4, 0, -1.17, 0, 0.785, math.pi/4]
    move_group.go(joint_reset, wait=True)
    move_group.stop()
    hand_pos = get_hand_position()
    goal = msg.goal
    print(goal)
    vector = vector2points(hand_pos, goal)
    obs = joint_reset
    obs = np.round(obs, 5)
    obs = np.append(obs, vector)
    print("Observation: ", obs)
    print("Handpos: ", hand_pos)
    print("Goal: ", goal)
    return ResetJointsResponse(obs=obs)


rospy.init_node('reset_service_11315', anonymous=False)
s = rospy.Service('reset_env_11315', ResetJoints, reset)
print("reset_node_11315 aktiv")
rospy.spin()
