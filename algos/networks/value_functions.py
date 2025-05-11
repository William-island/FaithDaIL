import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import mlp
from .features_extractor import make_features_extractor
from .preprocessing import preprocess_obs
# All networks with name {Net}Hook are used for monitoring representation of state when forwarding
# Use self.vf.fc2.register_forward_hook(self.get_activation()) to record state representation and then calculate cosine similarity
# Please check https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html for more details

class ValueFunction(nn.Module):
    def __init__(self, state_dim, layer_norm=False, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, layer_norm=layer_norm, squeeze_output=True)

    def forward(self, state):
        return self.v(state)

class ValueFunctionHook(nn.Module):
    def __init__(self, state_dim, layer_norm=False, hidden_dim=256, squeeze_output=True, use_orthogonal=False):
        super().__init__()
        self.use_layer_norm = layer_norm
        self.squeeze_output = squeeze_output
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        if use_orthogonal:
            nn.init.orthogonal_(self.fc1.weight)
            nn.init.orthogonal_(self.fc2.weight)
            nn.init.orthogonal_(self.fc3.weight)
        self.activation = nn.ReLU()
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, state):
        x = self.activation(self.layer_norm1(self.fc1(state))) if self.use_layer_norm else self.activation(self.fc1(state))
        x = self.activation(self.layer_norm2(self.fc2(x))) if self.use_layer_norm else self.activation(self.fc2(x))
        value = self.fc3(x).squeeze(-1) if self.squeeze_output else self.fc3(x)
        return value

class TwinV_old(nn.Module):
    def __init__(self, state_dim, layer_norm=False, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v1 = mlp(dims, layer_norm=layer_norm, squeeze_output=True)
        self.v2 = mlp(dims, layer_norm=layer_norm, squeeze_output=True)

    def both(self, state):
        return torch.stack([self.v1(state), self.v2(state)], dim=0)

    def forward(self, state):
        return torch.min(self.both(state), dim=0)[0]
    





class TwinV(nn.Module):
    def __init__(self, obs_space, features_dim, layer_norm=False, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.features_extractor1, self.features_dim = make_features_extractor(obs_space, features_dim)
        self.features_extractor2, self.features_dim = make_features_extractor(obs_space, features_dim)

        self.observation_space = obs_space
        self.normalize_images = True

        dims = [self.features_dim, *([hidden_dim] * n_hidden), 1]
        self.v1 = mlp(dims, layer_norm=layer_norm, squeeze_output=True)
        self.v2 = mlp(dims, layer_norm=layer_norm, squeeze_output=True)

    def both(self, obs):
        features1 = self.extract_features(obs, self.features_extractor1)
        features2 = self.extract_features(obs, self.features_extractor2)
        return torch.stack([self.v1(features1), self.v2(features2)], dim=0)

    def forward(self, state):
        return torch.min(self.both(state), dim=0)[0]
    

    def extract_features(self, obs, features_extractor=None):
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)

