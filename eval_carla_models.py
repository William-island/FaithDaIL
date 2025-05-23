import os
import torch
import pandas as pd
# from algos.models import GaussianPolicy
# from metadrive import MetaDriveEnv
import numpy as np
import yaml
import time
from argparse import ArgumentParser
import argparse
# from utils.utils import evaluate_carla_once
from envs import make_eval_env
from algos.networks import DeterministicPolicy
from torch.utils.tensorboard import SummaryWriter
import pickle
from utils.utils import to_torchify


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_carla_env(args):
    env = make_eval_env(args)
    return env



def init_policy(args, env, ckpt_path):

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


class EVAL:
    def __init__(self):
        self.last_obs = None

    def evaluate_carla_once(self, env, policy, deterministic=True):
        # evalute specific seed
        # obs = env.reset()
        total_reward = 0.
        done, i = False, 0
        while not done:
            with torch.no_grad():
                action = policy.act(to_torchify(self.last_obs), deterministic=deterministic).cpu().numpy()
            obs, reward, done, info = env.step(action)
            self.last_obs = obs
            # obs, reward, done, _, info = env.step(action)
            total_reward += reward[0]
            i += 1
            # if i > 3000:
            #     print("Episode is too long")
            #     break
        return total_reward, info[0]['route_completion'], info[0]['success']

    def evaluate(self, policy, env, idx, writer=None, log_path=None):
            ## new version with whole evaluation

            # test evaluation
            # evaluate_carla_once(env, policy)
            # real evaluation
            # returns = np.array([evaluate_carla_once(env, policy) for _ in range(15)])
            returns = []
            self.last_obs = env.reset()
            for i in range(15):
                print(f"Evaluating {i+1}th episode!!!!!")
                returns.append(self.evaluate_carla_once(env, policy))
            returns = np.array(returns)

            if writer is not None:
                writer.add_scalar('Eval/rewards', returns[:,0].mean(), i)
                writer.add_scalar('Eval/route_completion', returns[:,1].mean(), i)
                writer.add_scalar('Eval/route_success', returns[:,2].mean(), i)

            
            if log_path is not None:
                os.makedirs(log_path + "eval/", exist_ok=True)
                with open(log_path + "eval/" + f"eval_{idx}.pkl", "wb") as f:
                    pickle.dump(returns, f)

            print("---------------------------------------")
            print(f"Step:{idx} episodes: rewards: {returns[:,0].mean():.3f}, route_completion: {returns[:,1].mean():.3f}, route_success: {returns[:,2].mean():.3f}")
            print("---------------------------------------")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default="carla")
    parser.add_argument('--algo', type=str, default='hitl_odice')
    parser.add_argument('--seed', type=int, default=1)

    # parser.add_argument("--port", default=9000, type=int, help="Carla server port.")
    parser.add_argument("--port", required=True, type=int, help="Carla server port.")
    parser.add_argument(
        "--obs_mode",
        default="birdview",
        choices=["birdview", "first", "birdview42", "firststack"],
        help="The observation mode."
    )

    parser.add_argument("-s","--start_order",  type=int, required=True, help="The start index of the model.")
    parser.add_argument("-e", "--end_order",  type=int, required=True, help="The end index of the model.")
    
    with open(f"./configs/carla_configs/configs_FaithDaIL.yaml", "r") as file:
        config = yaml.safe_load(file)
    args = parser.parse_args(namespace=argparse.Namespace(**config))

    # fix the random seed
    np.random.seed(0)

    # models directory
    log_path = "/"
    # models_path = log_path + "models/"
    models_path = log_path + "models/"
    # get all the models in the directory
    models = sorted(os.listdir(models_path), key=lambda x: int(x[11:-4]))[:]

    # init env
    env = init_carla_env(args)

    # writer = SummaryWriter(log_path)
    writer = None
    # EVAL
    eval_obj = EVAL()
    # evaluate all the models
    STEP_INTERVAL = 256 # 512
    EVAL_INTERVAL = 10 # 4
    for i, model in enumerate(models[:]):
        idx = int(model[11:-4])//STEP_INTERVAL
        if idx % EVAL_INTERVAL == 0 and idx >= EVAL_INTERVAL*args.start_order and idx < EVAL_INTERVAL*args.end_order:
            policy = init_policy(args, env, models_path + model)
            eval_obj.evaluate(policy, env, idx, writer, log_path)
    