#!/usr/bin/env python3

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from forward_kin_6DOF import PandaRobotGymEnv
import numpy as np

policy_name = "reaching_policy"


def main():

    robot = PandaRobotGymEnv()
    robot = DummyVecEnv([lambda: robot])
    policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    model = PPO2(MlpPolicy, robot, n_steps=512, nminibatches=32, learning_rate=0.0003, policy_kwargs=policy_kwargs,
                 verbose=1, tensorboard_log="/home/valentin/gazebo_logs/pandareach_6DOF_nogazebo/")
    model.learn(total_timesteps=1000000)
    model.save("/home/valentin/gazebo_logs/pandareach_6DOF_nogazebo/reaching_policy/checkpoint-0")

if __name__ == '__main__':
    main()



