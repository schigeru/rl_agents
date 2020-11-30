#!/usr/bin/env python
import os
import math
import numpy as np
import random
import time
import rospy
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.msg import LinkState
import moveit_commander
from math import sin, cos, pi, atan, sqrt, atan2, asin

os.environ['ROS_MASTER_URI'] = "http://localhost:11315/"
os.environ['GAZEBO_MASTER_URI'] = "http://localhost:11345/"
rospy.init_node('tester', anonymous=False)
group_name = "panda_arm"
move_group = moveit_commander.MoveGroupCommander(group_name)

def calc_handorientation(rotation_matrix):
    print(rotation_matrix)

    yaw = atan2(rotation_matrix[1][0], rotation_matrix[0][0])
    pitch = atan2(-rotation_matrix[2][0], sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2))
    roll = atan2(rotation_matrix[2][1], rotation_matrix[2][2])

    tr = rotation_matrix[0][0] + rotation_matrix[1][1] + rotation_matrix[2][2] + 1.0

    # If the trace is nonzero, it's a nondegenerate rotation
    if tr > 1e-5:
        s = math.sqrt(tr)
        w = s * 0.5
        s = 0.5 / s
        x = (rotation_matrix[2][1] - rotation_matrix[1][2]) * s
        y = (rotation_matrix[0][2] - rotation_matrix[2][0]) * s
        z = (rotation_matrix[1][0] - rotation_matrix[0][1]) * s
        return [w, x, y, z]
    else:
        # degenerate it's a rotation of 180 degrees
        nxt = [1, 2, 0]
        # check for largest diagonal entry
        i = 0
        if rotation_matrix[1][1] > rotation_matrix[0][0]:
            i = 1
        if rotation_matrix[2][2] > max(rotation_matrix[0][0], rotation_matrix[1][1]):
            i = 2
        j = nxt[i]
        k = nxt[j]
        M = rotation_matrix

        q = [0.0] * 4
        s = math.sqrt((M[i][i] - (M[j][j] + M[k][k])) + 1.0)
        q[i] = s * 0.5

        if abs(s) < 1e-7:
            raise ValueError("Could not solve for quaternion... Invalid rotation matrix?")
        else:
            s = 0.5 / s
            q[3] = (M[k][j] - M[j][k]) * s
            q[j] = (M[i][j] + M[j][i]) * s
            q[k] = (M[i][k] + M[i][k]) * s
        w, x, y, z = q[3], q[0], q[1], q[2]
        return [x, y, z, w]


    #if rotation_matrix[2][0] != 1 and rotation_matrix[2][0] != -1:
    #    theta = -asin(rotation_matrix[2][0])
    #    theta2 = pi - theta
    #    psi = atan2(rotation_matrix[2][1]/cos(theta), rotation_matrix[2][2] / cos(theta))
    #    psi2 = atan2(rotation_matrix[2][1]/cos(theta2), rotation_matrix[2][2] / cos(theta2))
    #    phi = atan2(rotation_matrix[1][0]/cos(theta), rotation_matrix[0][0] / cos(theta))
    #    phi2 = atan2(rotation_matrix[1][0]/cos(theta2), rotation_matrix[0][0] / cos(theta2))
    #return [theta, psi, phi]

def calc_handpos(joint_state):
    trans1 = np.array([[cos(joint_state[0]), -sin(joint_state[0]), 0, 0],
                       [sin(joint_state[0]), cos(joint_state[0]), 0, 0],
                       [0, 0, 1, 0.333],
                       [0, 0, 0, 1]])
    trans2 = np.array([[cos(joint_state[1]), -sin(joint_state[1]), 0, 0],
                       [0, 0, 1, 0],
                       [-sin(joint_state[1]), -cos(joint_state[1]), 0, 0],
                       [0, 0, 0, 1]])
    trans3 = np.array([[cos(joint_state[2]), -sin(joint_state[2]), 0, 0],
                       [0, 0, -1, -0.316],
                       [sin(joint_state[2]), cos(joint_state[2]), 0, 0],
                       [0, 0, 0, 1]])
    trans4 = np.array([[cos(joint_state[3]), -sin(joint_state[3]), 0, 0.0825],
                       [0, 0, -1, 0],
                       [sin(joint_state[3]), cos(joint_state[3]), 0, 0],
                       [0, 0, 0, 1]])
    trans5 = np.array([[cos(joint_state[4]), -sin(joint_state[4]), 0, -0.0825],
                       [0, 0, 1, 0.384],
                       [-sin(joint_state[4]), -cos(joint_state[4]), 0, 0],
                       [0, 0, 0, 1]])
    trans6 = np.array([[cos(joint_state[5]), -sin(joint_state[5]), 0, 0],
                       [0, 0, -1, 0],
                       [sin(joint_state[5]), cos(joint_state[5]), 0, 0],
                       [0, 0, 0, 1]])
    trans7 = np.array([[cos(joint_state[6]), -sin(joint_state[6]), 0, 0.088],
                       [0, 0, -1, 0],
                       [sin(joint_state[6]), cos(joint_state[6]), 0, 0],
                       [0, 0, 0, 1]])
    trans8 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0.1655],
                       [0, 0, 0, 1]])

    trans_matrix = np.matmul(trans7, trans8)
    trans_matrix = np.matmul(trans6, trans_matrix)
    trans_matrix = np.matmul(trans5, trans_matrix)
    trans_matrix = np.matmul(trans4, trans_matrix)
    trans_matrix = np.matmul(trans3, trans_matrix)
    trans_matrix = np.matmul(trans2, trans_matrix)
    trans_matrix = np.matmul(trans1, trans_matrix)

    return trans_matrix

def get_hand_position():
    msg = rospy.wait_for_message('/gazebo/link_states', LinkStates)
    hand_positionx = (msg.pose[9].position.x + msg.pose[10].position.x) / 2
    hand_positiony = (msg.pose[9].position.y + msg.pose[10].position.y) / 2
    hand_positionz = (msg.pose[9].position.z + msg.pose[10].position.z) / 2
    hand_position = [hand_positionx, hand_positiony, hand_positionz]
    #hand_position = np.round(hand_position, 3)
    return hand_position


def reset():
    joint_reset = [0, -0.4, 0, -1.17, 0, 0.785, math.pi/4]
    move_group.go(joint_reset, wait=True)
    move_group.stop()


def main():

    reset()
    for i in range(1, 30):
        reset()
        joint_values = move_group.get_current_joint_values()
        #joint_values[0] = joint_values[0] + random.randint(1, 20) / 10
        #joint_values[1] = joint_values[1] + random.randint(1, 20) / 10
        #joint_values[2] = joint_values[2] + random.randint(1, 20) / 10
        #joint_values[3] = joint_values[3] + random.randint(1, 20) / 10
        #joint_values[4] = joint_values[4] + random.randint(1, 20) / 10
        #joint_values[5] = joint_values[5] + random.randint(1, 20) / 10
        #joint_values[6] = joint_values[6] + random.randint(1, 20) / 10


        move_group.go(joint_values, wait=True)
        move_group.stop()
        joint_values = move_group.get_current_joint_values()
        hand_pos = get_hand_position()
        hand_pos_fwk = calc_handpos(joint_values)
        rotation_matrix = hand_pos_fwk[0:-1, 0:-1]
        orientation = calc_handorientation(rotation_matrix)
        print(orientation)
        hand_pos_fwk = [hand_pos_fwk[0][3], hand_pos_fwk[1][3], hand_pos_fwk[2][3]]
        error = np.array(hand_pos) - np.array(hand_pos_fwk)
        print("Handpos: ", hand_pos)
        print("Handpos FWK: ", hand_pos_fwk)
        print("error: ", error)
        if error[0] > 0.01 or error[1] > 0.01 or error[2] > 0.01:
            print("big error")
        raw_input("Press Enter")




if __name__ == '__main__':
    main()
