import os
import torch
import pandas as pd
from algos.networks import DeterministicPolicy
# from metadrive import MetaDriveEnv
from envs import make_eval_env
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
import time
from argparse import ArgumentParser
import argparse
from utils.utils import evaluate_once
from tqdm import tqdm


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_env(args):
    env = make_eval_env(args)
    return env

def init_policy(args, ckpt_path, env):

    # Initialize model
    policy = DeterministicPolicy(env.observation_space, args.features_dim, env.action_space, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    # Load pth
    checkpoint = torch.load(ckpt_path)
    # checkpoint = torch.load(ckpt_path + f"/eta_{args.eta}_Lambda_{args.Lambda}_checkpoint_{args.train_steps}.pth")
    policy.load_state_dict(checkpoint['policy'])
    policy.to(DEFAULT_DEVICE)
    print(f"***load the model from {ckpt_path}***")
    # Set eval mode
    policy.eval()
    return policy



def eval_metadrive(policy, env, i, writer=None):
        ## new version with whole evaluation
        normalized_returns = []
        normalized_cost = []
        normalized_success = []

        for seed in range(1000, 1100):# range(1000, 1050):
            returns, costs, success = evaluate_once(env, policy, seed)
            normalized_returns.append(returns)
            normalized_cost.append(costs)
            normalized_success.append(success)
            print(f"\t Rewards: {returns:.3f}, costs: {costs:.3f}, success: {success:.3f}")

            # for _ in range(5):
            #     returns, costs, success = evaluate_once(env, policy, seed)
            #     normalized_returns.append(returns)
            #     normalized_cost.append(costs)
            #     normalized_success.append(success)
            #     print(f"\t Rewards: {returns:.3f}, costs: {costs:.3f}, success: {success:.3f}")
        
    
        normalized_returns = np.array(normalized_returns).mean()
        normalized_cost = np.array(normalized_cost).mean()
        normalized_success = np.array(normalized_success).mean()

        if writer is not None:
            writer.add_scalar('Eval/normalized return mean', normalized_returns, i)
            writer.add_scalar('Eval/normalized cost mean', normalized_cost, i)
            writer.add_scalar('Eval/normalized success mean', normalized_success, i)

        print("---------------------------------------")
        print(f"Env: PVP, Evaluation steps {i} rewards: {normalized_returns:.3f}, costs: {normalized_cost:.3f}, success: {normalized_success:.3f}")
        print("---------------------------------------")

        return normalized_returns, normalized_cost, normalized_success


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default="metadrive", choices=["metadrive", "carla"])
    
    with open("./configs/metadrive_configs/configs_online_drm_odice_isw.yaml", "r") as file:
        config = yaml.safe_load(file)
    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args = parser.parse_args(namespace=argparse.Namespace(**config))

    # fix the random seed
    np.random.seed(0)

    # models directory
    log_path = "/"
    # models_path = log_path + "last_policy/"
    models_path = log_path + "models/"
    # get all the models in the directory
    models = sorted(os.listdir(models_path), key=lambda x: int(x[11:-4]))

    # init env
    env = init_env(args)


    writer = SummaryWriter(log_dir=log_path)

    # evaluate all the models
    for model in tqdm(models[:]):
        order = int(model[11:-4])//256
        if order % 5 == 0:
            print(f"Evaluating model {model}")
            policy = init_policy(args, models_path + model, env)
            eval_metadrive(policy, env, int(model[11:-4]), writer)

    