#!/usr/bin/env python3

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from forward_kin import PandaRobotGymEnv
import numpy as np

policy_name = "reaching_policy"


def main():

    robot = PandaRobotGymEnv()
    robot = DummyVecEnv([lambda: robot])
    policy_kwargs = dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])])
    model = PPO2(MlpPolicy, robot, n_steps=512, nminibatches=32, learning_rate=0.0003, policy_kwargs=policy_kwargs,
                 verbose=1, tensorboard_log="/home/valentin/gazebo_logs/pandareach_2DOF_nogazebo/")
    model.learn(total_timesteps=1000000)
    model.save("/home/valentin/gazebo_logs/pandareach_2DOF_nogazebo/reaching_policy/checkpoint-0")





'''    for i in range(83, 100):
        if i == 0:
            model.learn(total_timesteps=1024)
            model.save("/home/valentin/gazebo_logs/pandareach_2DOF/" + policy_name + "/checkpoint-0")
            print("Saving model to /home/valentin/gazebo_logs/pandareach_2DOF/" + policy_name + "/checkpoint-0")
        else:
            model = PPO2.load("/home/valentin/gazebo_logs/pandareach_2DOF/" + policy_name + "/checkpoint-" + str(i-1), env=robot)
            model.learn(total_timesteps=1024)
            model.save("/home/valentin/gazebo_logs/pandareach_2DOF/" + policy_name + "/checkpoint-" + str(i))
            print("Saving model to /home/valentin/gazebo_logs/pandareach_2DOF/" + policy_name + "/checkpoint-" + str(i))'''



if __name__ == '__main__':
    main()



