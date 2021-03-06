#!/usr/bin/env python3

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from gym2DOF_env import PandaRobotGymEnv
import rospy
import numpy as np
import os

policy_name = "reaching_policy"


def main():

    os.environ['ROS_MASTER_URI'] = "http://localhost:11315/"
    os.environ['GAZEBO_MASTER_URI'] = "http://localhost:11345/"
    robot = PandaRobotGymEnv(rosport=11315)
    robot = DummyVecEnv([lambda: robot])
    policy_kwargs = dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])])
    model = PPO2.load("/home/valentin/gazebo_logs/pandareach_2DOF_nogazebo/reaching_policy/checkpoint-0", env=robot)

    obs = robot.reset()

    for i in range(0, 500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = robot.step(action)


if __name__ == '__main__':
    main()



