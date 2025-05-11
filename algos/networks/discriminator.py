import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

from utils.utils import mlp, build_mlp

from .features_extractor import make_features_extractor
from .preprocessing import preprocess_obs, get_action_dim



class Discriminator_old(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_units=(256, 256),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=obs_dim + act_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))

    def calculate_weight(self, states, actions):
        # calculate the weight of d_\pi^e/d_\pi
        with torch.no_grad():
            return torch.exp(self.forward(states, actions))
        


class Discriminator(nn.Module):
    def __init__(self, obs_space, features_dim, act_space, hidden_units=(256, 256),
                 hidden_activation=nn.Tanh()):
        super().__init__()
        self.features_extractor, self.features_dim = make_features_extractor(obs_space, features_dim)
        self.observation_space = obs_space
        self.action_space = act_space

        self.normalize_images = True
        self.act_dim = get_action_dim(act_space)

        self.net = build_mlp(
            input_dim=self.features_dim+self.act_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        features = self.extract_features(states)
        return self.net(torch.cat([features, actions], dim=-1))

    def calculate_weight(self, states, actions):
        # calculate the weight of d_\pi^e/d_\pi
        with torch.no_grad():
            return torch.exp(self.forward(states, actions))
        
    def extract_features(self, obs):
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs)