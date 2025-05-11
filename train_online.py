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
from envs import make_train_env




def main(args):
    ## init training env
    train_env = make_train_env(args)

    # algorithm
    algo = ALGO(args, train_env)

    ## policy.learn
    algo.online_learn()



if __name__ == '__main__':
    ## choose algo
    algo = 'online_drm_odice_isw'   # 'online_drm_odice_isw_wob'
    env_name = "metadrive" # "metadrive" or "carla"

    ## make args
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default=env_name, choices=["metadrive", "carla"])
    parser.add_argument('--algo', type=str, choices=['online_drm_odice_isw']\
                        , default=algo)
    
    with open(f"./configs/{env_name}_configs/configs_{algo}.yaml", "r") as file:
        config = yaml.safe_load(file)
    args = parser.parse_args(namespace=argparse.Namespace(**config))

    main(args)
