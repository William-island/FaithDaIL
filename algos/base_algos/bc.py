import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from algos.networks import DeterministicPolicy
import copy
import os
from utils.utils import DEFAULT_DEVICE



class BC(nn.Module):
    def __init__(self, args, observation_space, action_space):
        super().__init__()
        # make policy and value function

        # choose the policy
        policy = DeterministicPolicy(observation_space, args.features_dim, action_space, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.policy_lr)


    def update(self, observations, actions):
        # update policy
        self.policy_optimizer.zero_grad()
        actions_pred = self.policy(observations)
        loss = F.mse_loss(actions_pred, actions)
        loss.backward()
        self.policy_optimizer.step()
        return loss.item()


    def save(self, model_dir, step):
        checkpoint = {
            'step': step,
            'policy': self.policy.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
        }
        save_dir = model_dir + "models/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(checkpoint, save_dir + f"/checkpoint_{step}.pth")
        print(f"***save models to {model_dir}***")

    def load(self, model_dir, step):
        checkpoint = torch.load(model_dir + f"/checkpoint_{step}.pth")
        # self.step = checkpoint['step']
        self.policy.load_state_dict(checkpoint['policy'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        print(f"***load the model from {model_dir}***")