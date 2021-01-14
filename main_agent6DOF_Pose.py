#!/usr/bin/env python3

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from gym6DOF_Pose_env import PandaRobotGymEnv
import rospy
import numpy as np
import os

policy_name = "reaching_policy"


def main():
    robot = PandaRobotGymEnv()
    robot = DummyVecEnv([lambda: robot])
    model = PPO2.load("/home/valentin/gazebo_logs/pandareach_6DOF_Pose_nogazebo/reaching_policy/5mil_timesteps", env=robot)

    obs = robot.reset()

    for i in range(0, 500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = robot.step(action)

if __name__ == '__main__':
    main()



