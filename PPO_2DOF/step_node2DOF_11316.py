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
from panda_rl.srv import StepAction, StepActionResponse

group_name = "panda_arm"
move_group = moveit_commander.MoveGroupCommander(group_name)


def vector2points(v, u):
    v = np.array(v)
    u = np.array(u)
    vector = u - v
    return vector


def get_hand_position():
    msg = rospy.wait_for_message('/gazebo/link_states', LinkStates)
    hand_positionx = (msg.pose[9].position.x + msg.pose[10].position.x) / 2
    hand_positiony = (msg.pose[9].position.y + msg.pose[10].position.y) / 2
    hand_positionz = (msg.pose[9].position.z + msg.pose[10].position.z) / 2
    hand_position = [hand_positionx, hand_positiony, hand_positionz]
    return hand_position


def goal_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    distance = np.linalg.norm(x-y)
    return distance


def take_action(msg):
    done = False
    goal = msg.goal
    joint_state = move_group.get_current_joint_values()
    joint_state[0] = joint_state[0] + (msg.action[0]/20)
    joint_state[1] = joint_state[1] + (msg.action[1]/20)
    joint_state[2] = joint_state[2] + (msg.action[2]/20)
    joint_state[3] = joint_state[3] + (msg.action[3]/20)
    joint_state[4] = joint_state[4] + (msg.action[4]/20)
    joint_state[5] = joint_state[5] + (msg.action[5]/20)
    joint_state[6] = joint_state[6]

    if joint_state[0] < joint1_threshold_min or joint_state[0] > joint1_threshold_max \
        or joint_state[1] < joint2_threshold_min or joint_state[1] > joint2_threshold_max \
        or joint_state[2] < joint3_threshold_min or joint_state[2] > joint3_threshold_max \
        or joint_state[3] < joint4_threshold_min or joint_state[3] > joint4_threshold_max \
        or joint_state[4] < joint5_threshold_min or joint_state[4] > joint5_threshold_max \
        or joint_state[5] < joint6_threshold_min or joint_state[5] > joint6_threshold_max:

        hand_position = get_hand_position()
        v = vector2points(hand_position, goal)
        obs = joint_state
        obs.extend(v)
        done = True
        reward = -100
        return StepActionResponse(obs=obs, reward=reward, done=done)

    else:
        try:
            move_group.go(joint_state, wait=True)
            move_group.stop()

        except Exception as e:
            print("No Movement possible: %s" % e)
            done = True
            reward = -100
            obs = move_group.get_current_joint_values()
            hand_position = get_hand_position()
            d = goal_distance(hand_position, goal)
            v = vector2points(hand_position, goal)
            obs.extend(v)
            return StepActionResponse(obs=obs, reward=reward, done=done)

        obs = move_group.get_current_joint_values()
        hand_position = get_hand_position()
        d = goal_distance(hand_position, goal)
        v = vector2points(hand_position, goal)
        obs.extend(v)

        if d < 0.02:
            reward = 0
            print("target reached")
            done = True
            return StepActionResponse(obs=obs, reward=reward, done=done)

        elif hand_position[2] < 0.01:
            print("Gripper touched Ground")
            reward = -100
            done = True
            return StepActionResponse(obs=obs, reward=reward, done=done)

        else:
            reward = -d
            print(reward)
            return StepActionResponse(obs=obs, reward=reward, done=done)


joint1_threshold_min = -2.8973 + 0.1
joint2_threshold_min = -1.7628 + 0.1
joint3_threshold_min = -2.8973 + 0.1
joint4_threshold_min = -3.0718 + 0.1
joint5_threshold_min = -2.8973 + 0.1
joint6_threshold_min = -0.0175 + 0.1

joint1_threshold_max = 2.8973 - 0.1
joint2_threshold_max = 1.7628 - 0.1
joint3_threshold_max = 2.8973 - 0.1
joint4_threshold_max = -0.0698 - 0.1
joint5_threshold_max = 2.8973 - 0.1
joint6_threshold_max = 3.7525 - 0.1


rospy.init_node('step_service_11316', anonymous=False)
print("step_node_11316 aktiv")
s = rospy.Service('step_env_11316', StepAction, take_action)
rospy.spin()


