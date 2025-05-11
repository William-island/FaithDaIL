import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

from utils.utils import mlp
from .features_extractor import make_features_extractor
from .preprocessing import preprocess_obs, get_action_dim


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
epsilon = 1e-6


class DeterministicPolicy_old(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            action = self(obs)
            action = torch.clip(action, min=-1.0, max=1.0)
            return action
        






class DeterministicPolicy(nn.Module):
    def __init__(self, obs_space, features_dim, act_space, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.features_extractor, self.features_dim = make_features_extractor(obs_space, features_dim)
        self.observation_space = obs_space
        self.action_space = act_space

        self.normalize_images = True
        self.act_dim = get_action_dim(act_space)

        self.net = mlp([self.features_dim, *([hidden_dim] * n_hidden), self.act_dim],
                       output_activation=nn.Tanh)

    def forward(self, obs):
        features = self.extract_features(obs)
        return self.net(features)

    def act(self, obs, deterministic=True, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            action = self(obs)
            action = torch.clip(action, min=-1.0, max=1.0)
            return action
        
    def extract_features(self, obs):
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs)
    

class GaussianPolicy(nn.Module):
    def __init__(self, obs_space, features_dim, act_space, hidden_dim=256, n_hidden=2, use_tanh="False"):
        super().__init__()
        self.use_tanh = use_tanh
        self.act_dim = get_action_dim(act_space)

        self.features_extractor, self.features_dim = make_features_extractor(obs_space, features_dim)
        self.observation_space = obs_space

        self.normalize_images = True

        self.net = mlp([self.features_dim, *([hidden_dim] * n_hidden), self.act_dim])
        self.log_std = nn.Parameter(torch.zeros(self.act_dim, dtype=torch.float32))

    def forward(self, obs):
        features = self.extract_features(obs)
        mean = self.net(features)
        if self.use_tanh:
            mean = torch.tanh(mean)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            action = dist.mean if deterministic else dist.rsample()
            action = torch.clip(action, min=-1.0, max=1.0)
            return action
        
    def extract_features(self, obs):
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs)