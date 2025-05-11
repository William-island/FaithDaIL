import csv
from datetime import datetime
import json
from pathlib import Path
import random
import string
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import copy
import time
from collections import deque
from collections.abc import Iterable

import gymnasium
import gym


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, layer_norm=False, squeeze_output=False, use_orthogonal=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        if use_orthogonal:
            fc = nn.Linear(dims[i], dims[i+1])
            nn.init.orthogonal_(fc.weight)
            layers.append(fc)
        else:
            layers.append(nn.Linear(dims[i], dims[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i+1]))
        layers.append(activation())
    if use_orthogonal:
        fc = nn.Linear(dims[-2], dims[-1])
        nn.init.orthogonal_(fc.weight)
        layers.append(fc)
    else:
        layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        # assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x

def torchify_dict(d):
    return {k: torchify(v) for k, v in d.items()}

def cat_dicts(dicts,dims=0):
    return {k: torch.cat([d[k] for d in dicts], dim=dims) for k in dicts[0].keys()}

def to_torchify(x):
    # unified interface for converting to torch tensor
    if isinstance(x, dict):
        return torchify_dict(x)
    else:
        return torchify(x)
    
def to_cat(xs, dim=0):
    # unified interface for concatenating tensors
    if isinstance(xs[0], dict):
        return {k: torch.cat([x[k] for x in xs], dim=dim) for k in xs[0].keys()}
    else:
        return torch.cat(xs, dim=dim)
    

def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


def extract_done_makers(dones):
    (ends, ) = np.where(dones)
    starts = np.concatenate(([0], ends[:-1] + 1))
    length = ends - starts + 1
    return starts, ends, length


def _sample_indces(dataset, batch_size):
    try: 
        dones = dataset["timeouts"].cpu().numpy()
    except:
        dones = dataset["terminals"].cpu().numpy()
    starts, ends, lengths = extract_done_makers(dones)
    # credit to Dibya Ghosh's GCSL codebase
    trajectory_indces = np.random.choice(len(starts), batch_size)
    proportional_indices_1 = np.random.rand(batch_size)
    proportional_indices_2 = np.random.rand(batch_size)
    # proportional_indices_2 = 1
    time_dinces_1 = np.floor(
        proportional_indices_1 * (lengths[trajectory_indces] - 1)
    ).astype(int)
    time_dinces_2 = np.floor(
        proportional_indices_2 * (lengths[trajectory_indces])
    ).astype(int)
    start_indices = starts[trajectory_indces] + np.minimum(
        time_dinces_1,
        time_dinces_2
    )
    goal_indices = starts[trajectory_indces] + np.maximum(
        time_dinces_1,
        time_dinces_2
    )

    return start_indices, goal_indices


# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    n, device = len(dataset['states']), dataset['states'].device
    batch = {}
    # indices_0 = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    for k, v in dataset.items():
        if k == "trajectory_terminals":
            continue
        else:
            batch[k] = v[indices]
    return batch
        

def rvs_sample_batch(dataset, batch_size):
    start_indices, goal_indices = _sample_indces(dataset, batch_size)
    dict = {}
    for k, v in dataset.items():
        if (k == "observations") or (k == "actions"):
            dict[k] = v[start_indices]
    dict["next_observations"] = dataset["observations"][goal_indices]
    dict["rewards"] = 0
    dict["terminals"] = 0
    return dict



def get_pvp_env(type='eval'):
    from metadrive import MetaDriveEnv

    config = {
        # Environment setting:
        "out_of_route_done": True,  # Raise done if out of route.
        "num_scenarios": 50,  # There are totally 50 possible maps.
        "start_seed": 100,  # We will use the map 100 ~ 150 as the default training environment.
        "traffic_density": 0.06,

        # Set up the control device. Default to use keyboard with the pop-up interface.
        "manual_control": False,
        "controller": "keyboard",  # Selected from [keyboard, xbox, steering_wheel]
        "use_render": False,  # Use the render
        "random_agent_model": False,  # Make the agent of model False

        # Visualization
        "vehicle_config": {
            "show_dest_mark": True,  # Show the destination in a cube.
            "show_line_to_dest": True,  # Show the line to the destination.
            "show_line_to_navi_mark": True,  # Show the line to next navigation checkpoint.
        }
    }

    if type == 'train':
        config["use_render"] = True

    # if type == 'eval':
    #     config["start_seed"] = 1000 # we will test the model on the map 1000 ~ 1050

    # if type == 'specific':
    #     pass

    env = MetaDriveEnv(config)

    return env


def evaluate_once(env, policy, seed=None, deterministic=True):
    # evalute specific seed
    if seed is None:
        raise ValueError("Seed is None")
    obs = env.reset(seed=int(seed))

    total_reward = 0.
    total_cost = 0.
    success = False

    done, i = False, 0
    while not done:
        with torch.no_grad():
            action = policy.act(torchify(obs).unsqueeze(0), deterministic=deterministic).cpu().numpy()[0]
        # obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)

        total_reward += info["step_reward"]

        total_cost += info['cost']
        # print(f"{i}: {info['cost']}")

        success = info["arrive_dest"]

        i += 1
        if i > 3000:
            print("Episode is too long")
            break

    ret = env.get_episode_result()
        
    # return total_reward, total_cost, success
    return ret['episode_reward'], ret['episode_cost'], ret['success']


def evaluate_carla_once(env, policy, deterministic=True):
    # evalute specific seed
    obs = env.reset()
    total_reward = 0.
    done, i = False, 0
    while not done:
        with torch.no_grad():
            action = policy.act(to_torchify(obs), deterministic=deterministic).cpu().numpy()
        obs, reward, done, info = env.step(action)
        # obs, reward, done, _, info = env.step(action)
        total_reward += reward[0]
        i += 1
        # if i > 3000:
        #     print("Episode is too long")
        #     break
    return total_reward, info[0]['route_completion'], info[0]['success']


def evaluate(env, policy, mean, std, deterministic=True):
    # get random seed between 100~150 for testing
    obs,_ = env.reset(seed=np.random.randint(100, 150))
    total_reward = 0.
    done, i = False, 0
    while not done:
        obs = (obs - mean)/std
        with torch.no_grad():
            action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
        # obs, reward, done, info = env.step(action)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        i += 1
        if i > 3000:
            print("Episode is too long")
            break
    return total_reward



def evaluate_por(env, policy, goal_policy, mean, std, deterministic=True):
    obs = env.reset()
    total_reward = 0.
    done, i = False, 0
    while not done:
        obs = (obs - mean)/std
        with torch.no_grad():
            g = goal_policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
            action = policy.act(torchify(np.concatenate([obs, g])), deterministic=deterministic).cpu().numpy()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        i += 1
    return total_reward


def evaluate_rvs(env, policy, mean, std, deterministic=True):
    obs = env.reset()
    goal = np.array(env.target_goal)
    goal = (goal - mean[:2])/std[:2]
    total_reward = 0.
    done, i = False, 0
    while not done:
        obs = (obs - mean)/std
        with torch.no_grad():
            if i % 100 == 0:
                print('current location:', obs[:2])
            action = policy.act(torchify(np.concatenate([obs, goal])), deterministic=deterministic).cpu().numpy()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        i += 1
    return total_reward


def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def save(dir ,filename, env_name, network_model):
    if not os.path.exists(dir):
        os.mkdir(dir)
    file = dir + env_name + "-" + filename 
    torch.save(network_model.state_dict(), file)
    print(f"***save the {network_model} model to {file}***")
    

def load(dir, filename, env_name, network_model):
    file = dir + env_name + "-" + filename
    if not os.path.exists(file):
        raise FileExistsError("Doesn't exist the model")
    network_model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
    print(f"***load the model from {file}***")


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()

def dataset_T_trajs(dataset, T, terminate_on_end=False):
    """
    Returns T trajs from dataset.
    """
    N = dataset['rewards'].shape[0]
    return_traj = []
    obs_traj = [[]]
    next_obs_traj = [[]]
    action_traj = [[]]
    reward_traj = [[]]
    done_traj = [[]]

    for i in range(N-1):
        obs_traj[-1].append(dataset['observations'][i].astype(np.float32))
        next_obs_traj[-1].append(dataset['observations'][i+1].astype(np.float32))
        action_traj[-1].append(dataset['actions'][i].astype(np.float32))
        reward_traj[-1].append(dataset['rewards'][i].astype(np.float32))
        done_traj[-1].append(bool(dataset['terminals'][i]))

        final_timestep = dataset['timeouts'][i] | dataset['terminals'][i]
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            return_traj.append(np.sum(reward_traj[-1]))
            obs_traj.append([])
            next_obs_traj.append([])
            action_traj.append([])
            reward_traj.append([])
            done_traj.append([])

    # select T trajectories
    inds_all = list(range(len(obs_traj)))
    inds = inds_all[:T]
    inds = list(inds)

    print('# select {} trajs in the dataset'.format(T))

    obs_traj = [obs_traj[i] for i in inds]
    next_obs_traj = [next_obs_traj[i] for i in inds]
    action_traj = [action_traj[i] for i in inds]
    reward_traj = [reward_traj[i] for i in inds]
    done_traj = [done_traj[i] for i in inds]

    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    return {
        'observations': concat_trajectories(obs_traj),
        'actions': concat_trajectories(action_traj),
        'next_observations': concat_trajectories(next_obs_traj),
        'rewards': concat_trajectories(reward_traj),
        'terminals': concat_trajectories(done_traj),
    }

def dataset_split_expert(dataset, split_x, exp_num, terminate_on_end=False):
    """
    Returns D_e and expert data in D_o of setting 1 in the paper.
    """
    N = dataset['rewards'].shape[0]
    return_traj = []
    obs_traj = [[]]
    next_obs_traj = [[]]
    action_traj = [[]]
    reward_traj = [[]]
    done_traj = [[]]

    for i in range(N-1):
        obs_traj[-1].append(dataset['observations'][i].astype(np.float32))
        next_obs_traj[-1].append(dataset['observations'][i+1].astype(np.float32))
        action_traj[-1].append(dataset['actions'][i].astype(np.float32))
        reward_traj[-1].append(dataset['rewards'][i].astype(np.float32))
        done_traj[-1].append(bool(dataset['terminals'][i]))

        final_timestep = dataset['timeouts'][i] | dataset['terminals'][i]
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            return_traj.append(np.sum(reward_traj[-1]))
            obs_traj.append([])
            next_obs_traj.append([])
            action_traj.append([])
            reward_traj.append([])
            done_traj.append([])

    # select 10 trajectories
    inds_all = list(range(len(obs_traj)))
    inds_succ = inds_all[:exp_num]
    inds_o = inds_succ[-split_x:]
    inds_o = list(inds_o)
    inds_succ = list(inds_succ)
    inds_e = set(inds_succ) - set(inds_o)
    inds_e = list(inds_e)

    print('# select {} trajs in expert dataset as D_e'.format(len(inds_e)))
    print('# select {} trajs in expert dataset as expert data in D_o'.format(len(inds_o)))

    obs_traj_e = [obs_traj[i] for i in inds_e]
    next_obs_traj_e = [next_obs_traj[i] for i in inds_e]
    action_traj_e = [action_traj[i] for i in inds_e]
    reward_traj_e = [reward_traj[i] for i in inds_e]
    done_traj_e = [done_traj[i] for i in inds_e]

    obs_traj_o = [obs_traj[i] for i in inds_o]
    next_obs_traj_o = [next_obs_traj[i] for i in inds_o]
    action_traj_o = [action_traj[i] for i in inds_o]
    reward_traj_o = [reward_traj[i] for i in inds_o]
    done_traj_o = [done_traj[i] for i in inds_o]

    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    dataset_e = {
        'observations': concat_trajectories(obs_traj_e),
        'actions': concat_trajectories(action_traj_e),
        'next_observations': concat_trajectories(next_obs_traj_e),
        'rewards': concat_trajectories(reward_traj_e),
        'terminals': concat_trajectories(done_traj_e),
    }

    dataset_o = {
        'observations': concat_trajectories(obs_traj_o),
        'actions': concat_trajectories(action_traj_o),
        'next_observations': concat_trajectories(next_obs_traj_o),
        'rewards': concat_trajectories(reward_traj_o),
        'terminals': concat_trajectories(done_traj_o),
    }

    return dataset_e, dataset_o

def dataset_mix_trajs(expert_dataset, random_dataset, split_num, exp_num):
    dataset_o = dataset_T_trajs(random_dataset, 1000)
    dataset_o['flags'] = np.zeros_like(dataset_o['terminals']).astype(np.float32)
    dataset_e, dataset_o_extra = dataset_split_expert(expert_dataset, split_num, exp_num)
    dataset_e['flags'] = np.ones_like(dataset_e['terminals']).astype(np.float32)
    dataset_o_extra['flags'] = np.ones_like(dataset_o_extra['terminals']).astype(np.float32)
    for key in dataset_o.keys():
        dataset_o[key] = np.concatenate([dataset_o[key], dataset_o_extra[key]], 0)
    return dataset_e, dataset_o




## self writen data buffer class
class data_buffer():
    def __init__(self, max_size, obs_dim, act_dim):
        self.max_size = max_size
        self.pos = 0
        self.full = False

        # init numpy array of observations, actions, next_observations, rewards, terminals
        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, ), dtype=np.float32)
        self.terminals = np.zeros((max_size, ), dtype=np.float32)

    def add(self, obs, act, next_obs, reward, terminal):
        self.observations[self.pos] = obs
        self.actions[self.pos] = act
        self.next_observations[self.pos] = next_obs
        self.rewards[self.pos] = reward
        self.terminals[self.pos] = terminal

        self.pos = (self.pos + 1) % self.max_size

        if self.pos == 0:
            self.full = True

    def sample(self, batch_size, indices=None):
        if indices is None:
            indices = np.random.choice(self.size(), batch_size, replace=False)
        return dict(
            observations=self.observations[indices],
            actions=self.actions[indices],
            next_observations=self.next_observations[indices],
            rewards=self.rewards[indices],
            terminals=self.terminals[indices]
        )
    
    def size(self):
        if self.full:
            return self.max_size
        else:
            return self.pos
        
    def is_full(self):
        return self.pos == 0 and self.full
    
    def is_real_full(self):
        return self.full
        
    def get_all(self)-> np.array:
        return dict(
            observations=self.observations[:self.size()],
            actions=self.actions[:self.size()],
            next_observations=self.next_observations[:self.size()],
            rewards=self.rewards[:self.size()],
            terminals=self.terminals[:self.size()]
        )
    
    def load(self, data_dict):
        self.observations = data_dict['observations']
        self.actions = data_dict['actions']
        self.next_observations = data_dict['next_observations']
        self.rewards = data_dict['rewards']
        self.terminals = data_dict['terminals']
        assert len(self.observations) <= self.max_size
        self.pos = len(self.rewards) % self.max_size
        if len(self.rewards) == self.max_size:
            self.full = True

    def extend(self, data_dict):
        length = len(data_dict['observations'])
        assert self.size() + length < self.max_size

        self.observations[self.pos:self.pos+length] = data_dict['observations']
        self.actions[self.pos:self.pos+length] = data_dict['actions']
        self.next_observations[self.pos:self.pos+length] = data_dict['next_observations']
        self.rewards[self.pos:self.pos+length] = data_dict['rewards']
        self.terminals[self.pos:self.pos+length] = data_dict['terminals']

        self.pos = (self.pos + length) % self.max_size


## self writen data buffer class
class data_buffer_general():
    def __init__(self, max_size, obs_space, act_space):
        self.max_size = max_size
        self.pos = 0
        self.full = False
        act_dim = act_space.shape[0]
        self.observation_space = obs_space
        if isinstance(self.observation_space, gymnasium.spaces.Box):
            self.obs_shape = obs_space.shape
        elif isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {key: sub_space.shape for key, sub_space in obs_space.spaces.items()}
        
        # init numpy array of observations, actions, next_observations, rewards, terminals
        if isinstance(self.obs_shape, dict):
            self.observations = {
                key: np.zeros((self.max_size,) + _obs_shape, dtype=self.observation_space[key].dtype)
                for key, _obs_shape in self.obs_shape.items()
            }
            self.next_observations = {
                key: np.zeros((self.max_size,) + _obs_shape, dtype=self.observation_space[key].dtype)
                for key, _obs_shape in self.obs_shape.items()
            }
        else:
            self.observations = np.zeros((max_size,) + self.obs_shape, dtype=self.observation_space.dtype)
            self.next_observations = np.zeros((max_size,) + self.obs_shape, dtype=self.observation_space.dtype)

        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, ), dtype=np.float32)
        self.terminals = np.zeros((max_size, ), dtype=np.float32)

    def add(self, obs, act, next_obs, reward, terminal):
        if isinstance(self.obs_shape, dict):
            for key in self.obs_shape.keys():
                self.observations[key][self.pos] = obs[key]
                self.next_observations[key][self.pos] = next_obs[key]
        else:
            self.observations[self.pos] = obs
            self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = act
        self.rewards[self.pos] = reward
        self.terminals[self.pos] = terminal

        self.pos = (self.pos + 1) % self.max_size

        if self.pos == 0:
            self.full = True

    def sample(self, batch_size, indices=None):
        if indices is None:
            indices = np.random.choice(self.size(), batch_size, replace=False)
        if isinstance(self.obs_shape, dict):
            res = dict(
                        observations={key: self.observations[key][indices] for key in self.obs_shape.keys()},
                        actions=self.actions[indices],
                        next_observations={key: self.next_observations[key][indices] for key in self.obs_shape.keys()},
                        rewards=self.rewards[indices],
                        terminals=self.terminals[indices]
                    )
        else:
            res = dict(
                        observations=self.observations[indices],
                        actions=self.actions[indices],
                        next_observations=self.next_observations[indices],
                        rewards=self.rewards[indices],
                        terminals=self.terminals[indices]
                    )
        return res
    
    def size(self):
        if self.full:
            return self.max_size
        else:
            return self.pos
        
    def set_size(self, size):
        if size > self.max_size:
            raise ValueError("size is larger than max_size")
        self.pos = size
        if size == self.max_size:
            self.full = True
        else:
            self.full = False
        
    def get_all(self)-> np.array:
        if isinstance(self.obs_shape, dict):
            return dict(
                observations={key: self.observations[key][:self.size()] for key in self.obs_shape.keys()},
                actions=self.actions[:self.size()],
                next_observations={key: self.next_observations[key][:self.size()] for key in self.obs_shape.keys()},
                rewards=self.rewards[:self.size()],
                terminals=self.terminals[:self.size()]
            )
        else:
            return dict(
                observations=self.observations[:self.size()],
                actions=self.actions[:self.size()],
                next_observations=self.next_observations[:self.size()],
                rewards=self.rewards[:self.size()],
                terminals=self.terminals[:self.size()]
            )
    
    def load(self, data_dict):
        length = len(data_dict['actions'])
        assert length <= self.max_size
        if isinstance(self.obs_shape, dict):
            for key in self.obs_shape.keys():
                self.observations[key][:length] = data_dict['observations'][key]
                self.next_observations[key][:length] = data_dict['next_observations'][key]
        else:
            self.observations[:length] = data_dict['observations']
            self.next_observations[:length] = data_dict['next_observations']
        self.actions[:length] = data_dict['actions']
        self.rewards[:length] = data_dict['rewards']
        self.terminals[:length] = data_dict['terminals']
        
        self.pos = length % self.max_size
        if length == self.max_size:
            self.full = True

    def extend(self, data_dict):
        length = len(data_dict['observations'])
        assert self.size() + length < self.max_size

        for key in self.obs_shape.keys():
            self.observations[key][self.pos:self.pos+length] = data_dict['observations'][key]
            self.next_observations[key][self.pos:self.pos+length] = data_dict['next_observations'][key]
        self.actions[self.pos:self.pos+length] = data_dict['actions']
        self.rewards[self.pos:self.pos+length] = data_dict['rewards']
        self.terminals[self.pos:self.pos+length] = data_dict['terminals']

        self.pos = (self.pos + length) % self.max_size


        
# special version for FTPL to store history
class history_buffer():
    def __init__(self, max_size, batch_size, obs_dim, act_dim):
        self.max_size = max_size
        self.batch_size = batch_size
        self.pos = 0
        self.full = False

        # init numpy array of observations, actions, next_observations, rewards, terminals
        self.states = np.zeros((max_size, batch_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, batch_size, act_dim), dtype=np.float32)
        self.states_exp = np.zeros((max_size, batch_size, obs_dim), dtype=np.float32)
        self.actions_exp = np.zeros((max_size, batch_size, act_dim), dtype=np.float32)
        

    def add(self, states, actions, states_exp, actions_exp):
        self.states[self.pos] = states
        self.actions[self.pos] = actions
        self.states_exp[self.pos] = states_exp
        self.actions_exp[self.pos] = actions_exp

        self.pos = (self.pos + 1) % self.max_size

        if self.pos == 0:
            self.full = True

    def sample(self, index):
        return dict(
            states = torchify(self.states[index]),
            actions = torchify(self.actions[index]),
            states_exp = torchify(self.states_exp[index]),
            actions_exp = torchify(self.actions_exp[index])
        )
        
    
    def size(self):
        if self.full:
            return self.max_size
        else:
            return self.pos
        

# extended history buffer
class history_buffer_extended():
    def __init__(self, max_size, batch_size, obs_dim, act_dim):
        self.max_size = max_size
        self.batch_size = batch_size
        self.pos = 0
        self.full = False

        # init numpy array of observations, actions, next_observations, rewards, terminals
        self.states = np.zeros((max_size, batch_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, batch_size, act_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, batch_size, obs_dim), dtype=np.float32)
        self.terminals = np.zeros((max_size, batch_size, ), dtype=np.float32)
        self.states_exp = np.zeros((max_size, batch_size, obs_dim), dtype=np.float32)
        self.actions_exp = np.zeros((max_size, batch_size, act_dim), dtype=np.float32)
        

    def add(self, states, actions, next_state, terminal, states_exp, actions_exp):
        self.states[self.pos] = states
        self.actions[self.pos] = actions
        self.next_states[self.pos] = next_state
        self.terminals[self.pos] = terminal
        self.states_exp[self.pos] = states_exp
        self.actions_exp[self.pos] = actions_exp

        self.pos = (self.pos + 1) % self.max_size

        if self.pos == 0:
            self.full = True

    def sample(self, index):
        return dict(
            states = torchify(self.states[index]),
            actions = torchify(self.actions[index]),
            next_states = torchify(self.next_states[index]),
            terminals = torchify(self.terminals[index]),
            states_exp = torchify(self.states_exp[index]),
            actions_exp = torchify(self.actions_exp[index])
        )
        
    
    def size(self):
        if self.full:
            return self.max_size
        else:
            return self.pos



## self writen general data buffer class with history
class history_batch_buffer_general():
    def __init__(self, max_size, batch_size, obs_space, act_space):
        self.max_size = max_size
        self.pos = 0
        self.full = False
        self.observation_space = obs_space
        act_dim = act_space.shape[0]
        if isinstance(self.observation_space, gymnasium.spaces.Box):
            self.obs_shape = obs_space.shape
        elif isinstance(self.observation_space, gymnasium.spaces.Dict):
            self.obs_shape = {key: sub_space.shape for key, sub_space in obs_space.spaces.items()}
        

        # init numpy array of observations, actions, next_observations, rewards, terminals
        if isinstance(self.obs_shape, dict):
            self.observations = {
                key: np.zeros((self.max_size, batch_size,) + _obs_shape, dtype=self.observation_space[key].dtype)
                for key, _obs_shape in self.obs_shape.items()
            }
            self.next_observations = {
                key: np.zeros((self.max_size, batch_size,) + _obs_shape, dtype=self.observation_space[key].dtype)
                for key, _obs_shape in self.obs_shape.items()
            }
            self.observations_exp = { 
                key: np.zeros((self.max_size, batch_size,) + _obs_shape, dtype=self.observation_space[key].dtype)
                for key, _obs_shape in self.obs_shape.items()
            }
        else:
            self.observations = np.zeros((max_size, batch_size,) + self.obs_shape, dtype=self.observation_space.dtype)
            self.next_observations = np.zeros((max_size, batch_size,) + self.obs_shape, dtype=self.observation_space.dtype)
            self.observations_exp = np.zeros((max_size, batch_size,) + self.obs_shape, dtype=self.observation_space.dtype)

        self.actions = np.zeros((max_size, batch_size, act_dim), dtype=np.float32)
        self.terminals = np.zeros((max_size, batch_size,), dtype=np.float32)
        self.actions_exp = np.zeros((max_size, batch_size, act_dim), dtype=np.float32)



    def add(self, obs, act, next_obs, terminal, obs_exp, act_exp):
        # self.observations[self.pos] = obs
        if isinstance(self.obs_shape, dict):
            for key in self.obs_shape.keys():
                self.observations[key][self.pos] = obs[key]
                self.next_observations[key][self.pos] = next_obs[key]
                self.observations_exp[key][self.pos] = obs_exp[key]
        else:
            self.observations[self.pos] = obs
            self.next_observations[self.pos] = next_obs
            self.observations_exp[self.pos] = obs_exp

        self.actions[self.pos] = act
        self.terminals[self.pos] = terminal
        self.actions_exp[self.pos] = act_exp

        self.pos = (self.pos + 1) % self.max_size

        if self.pos == 0:
            self.full = True

    def sample(self, index):
        if isinstance(self.obs_shape, dict):
            res = dict(
                        observations = {key: self.observations[key][index] for key in self.obs_shape.keys()},
                        actions = self.actions[index],
                        next_observations = {key: self.next_observations[key][index] for key in self.obs_shape.keys()},
                        terminals = self.terminals[index],
                        observations_exp = {key: self.observations_exp[key][index] for key in self.obs_shape.keys()},
                        actions_exp = self.actions_exp[index]
                    )
        else:
            res = dict(
                        states = to_torchify(self.observations[index]),
                        actions =  to_torchify(self.actions[index]),
                        next_states =  to_torchify(self.next_observations[index]),
                        terminals =  to_torchify(self.terminals[index]),
                        states_exp =  to_torchify(self.observations_exp[index]),
                        actions_exp =  to_torchify(self.actions_exp[index])
                    )
        return res
    
    def size(self):
        if self.full:
            return self.max_size
        else:
            return self.pos
        

















def eval(step, writer, env, policy, args):
        eval_returns = np.array([evaluate_once(env, policy) for _ in range(args.n_eval_episodes)])
        # normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        normalized_returns = eval_returns
        return_info = {}
        return_info["normalized return mean"] = normalized_returns.mean()
        return_info["normalized return std"] = normalized_returns.std()
        return_info["percent difference 10"] = (normalized_returns[: 10].min() - normalized_returns[: 10].mean()) / normalized_returns[: 10].mean()
        # wandb.log(return_info, step=step)
        writer.add_scalar('Eval/normalized return mean', normalized_returns.mean(), step)

        print("---------------------------------------")
        print(f"Env: {args.env_name}, Evaluation over {args.n_eval_episodes} episodes: score: {normalized_returns.mean():.3f}")
        print("---------------------------------------")

        return normalized_returns.mean()


## special version for odice
def compute_ensemble_scores(models, o):
    outputs = []

    # # numpy -> tensor
    # o_tensor = torch.from_numpy(o).to(DEFAULT_DEVICE)

    # # Confirm the shape is correct
    # if len(o_tensor.shape) == 1:
    #     o_tensor = o_tensor.unsqueeze(0)

    for model in models:
        outputs.append(model.act(o, deterministic=True))

    # compute difference score
    outputs = torch.stack(outputs, dim=0)
    diff_score = torch.sum(torch.std(outputs, dim=0)).item()

    # scale up the score
    diff_score = diff_score * 10

    return diff_score

## special version for odice
def compute_ensemble_scores_batch(models, o):
    outputs = []

    # # numpy -> tensor
    # o_tensor = torch.from_numpy(o).to(DEFAULT_DEVICE)

    # # Confirm the shape is correct
    # if len(o_tensor.shape) == 1:
    #     o_tensor = o_tensor.unsqueeze(0)

    for model in models:
        outputs.append(model.act(o, deterministic=True).unsqueeze(1))

    # compute difference score
    outputs = torch.cat(outputs, dim=1)
    diff_scores = torch.sum(torch.std(outputs, dim=1),dim=1)

    # scale up the score
    diff_scores = (diff_scores * 10)+1

    return diff_scores












class ForceFPS:
    UNLIMITED = "UnlimitedFPS"
    FORCED = "ForceFPS"

    def __init__(self, fps, start=False):
        self.init_fps = fps
        if start:
            print("We will force the FPS to be near {}".format(fps))
            self.state = self.FORCED
            self.fps = fps + 1  # If we set to 10, FPS will jump in 9~10.
            self.interval = 1 / self.fps
        else:
            self.state = self.UNLIMITED
            self.fps = None
            self.interval = None
        self.dt_stack = deque(maxlen=10)
        self.last_time = time.time()

    def clear(self):
        self.dt_stack.clear()
        self.last_time = time.time()

    def sleep_if_needed(self):
        if self.fps is None:
            return
        self.dt_stack.append(time.time() - self.last_time)
        average_dt = sum(self.dt_stack) / len(self.dt_stack)
        if (self.interval - average_dt) > 0:
            time.sleep(self.interval - average_dt)
        self.last_time = time.time()


def merge_dicts(d1, d2):
    """
    Args:
        d1 (dict): Dict 1.
        d2 (dict): Dict 2.

    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged


def deep_update(
    original, new_dict, new_keys_allowed=False, allow_new_subkey_list=None, override_all_if_type_changes=None
):
    """Updates original dict with values from new_dict recursively.

    If new key is introduced in new_dict, then if new_keys_allowed is not
    True, an error will be thrown. Further, for sub-dicts, if the key is
    in the allow_new_subkey_list, then new subkeys can be introduced.

    Args:
        original (dict): Dictionary with default values.
        new_dict (dict): Dictionary with values to be updated
        new_keys_allowed (bool): Whether new keys are allowed.
        allow_new_subkey_list (Optional[List[str]]): List of keys that
            correspond to dict values where new subkeys can be introduced.
            This is only at the top level.
        override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (dict), iff the "type" key in that value dict changes.
    """
    allow_new_subkey_list = allow_new_subkey_list or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise Exception("Unknown config parameter `{}` ".format(k))

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Allowed key -> ok to add new subkeys.
            elif k in allow_new_subkey_list:
                deep_update(original[k], value, True)
            # Non-allowed key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original




def same_padding(in_size, filter_size, stride_size):
    """
    xxx: Copied from RLLib.

    Note: Padding is added to match TF conv2d `same` padding. See
    www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution

    Args:
        in_size (tuple): Rows (Height), Column (Width) for input
        stride_size (Union[int,Tuple[int, int]]): Rows (Height), column (Width)
            for stride. If int, height == width.
        filter_size (tuple): Rows (Height), column (Width) for filter

    Returns:
        padding (tuple): For input into torch.nn.ZeroPad2d.
        output (tuple): Output shape after padding and convolution.
    """
    in_height, in_width = in_size
    if isinstance(filter_size, int):
        filter_height, filter_width = filter_size, filter_size
    else:
        filter_height, filter_width = filter_size

    stride_size = stride_size if isinstance(stride_size, Iterable) else [stride_size, stride_size]
    stride_height, stride_width = stride_size

    out_height = np.ceil(float(in_height) / float(stride_height))
    out_width = np.ceil(float(in_width) / float(stride_width))

    pad_along_height = int(((out_height - 1) * stride_height + filter_height - in_height))
    pad_along_width = int(((out_width - 1) * stride_width + filter_width - in_width))
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    output = (out_height, out_width)
    return padding, output