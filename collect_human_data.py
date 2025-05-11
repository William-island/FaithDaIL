import argparse, yaml
import gym
import os
# import d4rl
import numpy as np
import torch
from tqdm import trange
from algos import ALGO
import wandb
import time
import pandas as pd
import pickle
from torch.utils.tensorboard import SummaryWriter
from envs import make_train_env, make_eval_env
from utils.utils import data_buffer_general

class DataCollector:
    def __init__(self, args, env):
        self.env = env
        self.data_size = args.data_size
        self.last_obs = None

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.human_data_buffer = data_buffer_general(self.data_size, self.observation_space, self.action_space)

        self.index = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # self.seed = 100

        # init log and writer
        self.log_dir = f"datasets/{self.index}/"
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        # eposode number
        self.episode_num = 0
        self.total_reward = 0
        self.success_num = 0

    def collect_data(self):
        # get action
        obs, reward, done, info = self.env.step([0., 0.])

        # store data
        if info['takeover'] or info["takeover_start"]:
            self.human_data_buffer.add(self.last_obs, np.array(info['raw_action']), obs, reward, done)
        else:
            self.human_data_buffer.add(self.last_obs, np.array([0., 0.]), obs, reward, done)


        # update last_obs
        self.last_obs = obs

        # update episode infomation
        self.total_reward += info["step_reward"]
        self.success_num += int(info["arrive_dest"])

        if done:
            print('save some thing')
            self.episode_num += 1
            self.writer.add_scalar('Takeover/total_reward', self.total_reward, self.episode_num)
            self.writer.add_scalar('Takeover/success_rate', self.success_num, self.episode_num)
            # save safety cost
            self.writer.add_scalar('Takeover/safety_cost', round(self.env.total_cost, 2), self.episode_num)

            self.reset_env()


    def reset_env(self):
        # seed=np.random.randint(100, 150)
        self.last_obs = self.env.reset(np.random.randint(100, 200))

        # self.last_obs = self.env.reset(self.seed)
        # self.seed += 1
        # if self.seed >= 150:
        #     self.seed = 100

    def save_data(self):
        human_data = self.human_data_buffer.get_all()
        save_dir = self.log_dir
        with open(save_dir + f"human_data_{self.index}.pkl", 'wb') as f:
            pickle.dump(human_data, f)
        print(f"***save human data to {save_dir}***")


    def load_data(self, path):
        with open(path, 'rb') as f:
            human_data = pickle.load(f)
            self.human_data_buffer.load(human_data)
        print(f"***load human data from {path}***")
        print(f"***data size: {self.human_data_buffer.size}***")



    def collect_human_data(self):
        if args.to_load_data:
            self.load_data(args.load_data_path)

        # init something
        self.reset_env()

        while(self.human_data_buffer.size() < self.data_size):
            self.collect_data()

            if self.human_data_buffer.size() % 1000 == 0 and self.human_data_buffer.size()>0:
                print(f"Collected {self.human_data_buffer.size()} data points, save data")
                self.save_data()



def main(args):
    np.random.seed(81)

    ## init training env
    train_env = make_train_env(args)

    data_collector = DataCollector(args, train_env) 

    # algorithm
    data_collector.collect_human_data()



if __name__ == '__main__':
    ## make args
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default="metadrive", choices=["metadrive", "carla"])
    parser.add_argument('--data_size', type=int, default=30_000)

    parser.add_argument('--to_load_data', type=bool, default=False)
    parser.add_argument('--load_data_path', type=str, default="./datasets/human_data_10000.pkl")
    
    
    # with open(f"./configs/configs_{algo}.yaml", "r") as file:
    #     config = yaml.safe_load(file)
    # args = parser.parse_args(namespace=argparse.Namespace(**config))
    args = parser.parse_args()

    main(args)