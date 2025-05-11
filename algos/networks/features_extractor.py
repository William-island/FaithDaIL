'''
features_extractor.py from pvp
'''
import gym
import gymnasium
import torch as th
from torch import nn

from utils.utils import same_padding
from .preprocessing import get_flattened_obs_dim




predefined_filters = {
    (240, 320): [
        [16, [12, 16], [7, 9]],
        [32, [6, 6], 4],
        [256, [9, 9], 1],
    ],
    (180, 320): [
        [16, [9, 16], [5, 9]],  # output: 36, 36
        [32, [3, 3], 2],  # output: 18, 18
        [64, [3, 3], 2],  # output: 9, 9
        [128, [3, 3], 3],  # output: 3, 3
        [256, [3, 3], 3],  # output: 1, 1
    ],
    (84, 84): [
        [16, [4, 4], 3],  # output: 28, 28
        [32, [3, 3], 2],  # output: 14, 14
        [64, [3, 3], 2],  # output: 7, 7
        [128, [3, 3], 2],  # output: 4, 4
        [256, [4, 4], 4],  # output: 1, 1
    ],

    # (42, 42): [
    #     [32, [4, 4], 3],  # output: 14, 14
    #     [64, [3, 3], 2],  # output: 7, 7
    #     [128, [3, 3], 2],  # output: 4, 4
    #     [256, [4, 4], 4],  # output: 1, 1
    # ],

    # PZH: A very tiny network!
    (42, 42): [
        [16, [4, 4], 3],  # output: 14, 14
        [32, [3, 3], 2],  # output: 7, 7
        [64, [3, 3], 2],  # output: 4, 4
        [128, [4, 4], 4],  # output: 1, 1
    ],
}





class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """
    def __init__(self, observation_space: gymnasium.Space, features_dim: int = 0):
        super(BaseFeaturesExtractor, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()



class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """
    def __init__(self, observation_space: gymnasium.Space):
        super(FlattenExtractor, self).__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)



class ImageFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(ImageFeaturesExtractor, self).__init__(observation_space, features_dim)
        if isinstance(observation_space, gymnasium.spaces.Box):
            obs_shape = observation_space.shape
            self.use_dict_obs_space = False
        else:
            obs_shape = observation_space["image"].shape
            self.use_dict_obs_space = True
        input_image_size = obs_shape[1:]
        self.filters = predefined_filters[input_image_size]
        layers = []
        input_size = obs_shape[0]
        for output_size, kernel, stride in self.filters:
            padding, input_image_size = same_padding(input_image_size, kernel, stride)
            layers.append(nn.ZeroPad2d(padding))
            layers.append(nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=stride))
            layers.append(nn.ReLU())
            input_size = output_size
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.use_dict_obs_space:
            obs_feat = self.cnn(observations["image"])
            other_feat = observations["speed"]
            ret = th.cat([obs_feat, other_feat], dim=1)
        else:
            ret = self.cnn(observations)
        assert ret.shape[-1] == self._features_dim
        return ret





def make_features_extractor(observation_space: gymnasium.Space, features_dim: int = 0):
    """
    Create a feature extractor based on the observation space. and renew the features_dim.

    :param observation_space:
    :param features_dim:
    :return:
    """
    if isinstance(observation_space, gymnasium.spaces.Box):
        return FlattenExtractor(observation_space), get_flattened_obs_dim(observation_space)
    elif isinstance(observation_space, gym.spaces.Dict):
        return ImageFeaturesExtractor(observation_space, features_dim), features_dim
    else:
        raise NotImplementedError("Unsupported observation space type")