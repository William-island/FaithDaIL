import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
# import wandb
import numpy as np
import time
from utils.utils import DEFAULT_DEVICE, update_exponential_moving_average, history_batch_buffer_general, data_buffer_general, to_torchify, torchify_dict, evaluate_once, get_pvp_env
from utils.utils import history_buffer_extended
import os
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm
from .pynol import iFTPL_Dp_policy_wob

from envs.vec_env import (
    DummyVecEnv,
    VecTransposeImage,
)

np.random.seed(81)



class ONLINE_DRM_ODICE_ISW(nn.Module):
    def __init__(self, args, env):
        super().__init__()

        # init env
        self.env_name = args.env_name
        if self.env_name == 'carla':
            env = self._wrap_env(env)
        self.env = env
        self.total_timesteps = args.train_steps
        self.train_freq = args.train_freq

        # init some parameters
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.epoch_disc = args.epoch_disc
        self.epoch_policy = args.epoch_policy


        # init model
        T = (self.total_timesteps//self.train_freq)
        self.model = iFTPL_Dp_policy_wob(T = T, device = DEFAULT_DEVICE, args=args, observation_space= self.observation_space, action_space=self.action_space)



        # init buffer
        self.replay_buffer = None
        self.human_data_buffer = None
        self.sub_buffer = None
        self.init_buffer(buffer_size=args.buffer_size, human_buffer_size=args.human_buffer_size, sub_buffer_size=args.sub_buffer_size, obs_space=self.observation_space, act_space=self.action_space)

        # init writer
        now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_dir = f"{args.log_dir}/{args.env_name}/{args.algo}/{now}/"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        ## other parameters
        self.train_freq = args.train_freq
        self.last_obs = None
        # self.gradient_steps = args.gradient_steps

        # balance sample or not
        self.balance_sample = args.balance_sample
        print(f"Balance Sample: {self.balance_sample}")

        self.batch_size = args.batch_size
        self.eval_period = args.eval_period
        self.n_eval_episodes = args.n_eval_episodes
        self.learning_steps_disc = 0
        self.learning_steps_policy = 0

        self.step = 0
        self.policy = None

        self.policy_algo = args.policy_algo

        self.loss_bases_list = []
        self.loss_bases_dir = log_dir + "loss_bases/"
        os.makedirs(self.loss_bases_dir, exist_ok=True)

        self.takeover_cost = 0


    def _wrap_env(self, env):
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose:
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        env = DummyVecEnv([lambda: env])
        if self.env_name == 'carla':
            env = VecTransposeImage(env)

        return env



    def init_buffer(self, buffer_size, human_buffer_size, sub_buffer_size, obs_space, act_space):
        self.replay_buffer = data_buffer_general(buffer_size, obs_space, act_space)
        self.human_data_buffer = data_buffer_general(human_buffer_size, obs_space, act_space)

        ## add a sub buffer to update the discriminator
        self.sub_buffer = data_buffer_general(sub_buffer_size, obs_space, act_space)

    
    def reset_env(self):
        # seed=np.random.randint(100, 150)
        if self.env_name == 'metadrive':
            self.last_obs = self.env.reset(np.random.randint(100, 200))
        elif self.env_name == 'carla':
            self.last_obs = self.env.reset()
    
    def _store_transition(self, obs, action, next_obs, reward, done, info):
        # only one environment
        if self.env_name == 'carla':
            obs, action, next_obs, reward, done, info = self._handle_vec_data(obs, action, next_obs, reward, done, info)

        # store data to replay buffer or human buffer
        if info['takeover'] or info["takeover_start"]:
            self.human_data_buffer.add(obs, np.array(info['raw_action']), next_obs, 0, done)
            # print("Human")
            self.takeover_cost += 1
            self.writer.add_scalar('Takeover/takeover_cost', self.takeover_cost, self.step)
        else:
            self.sub_buffer.add(obs, action, next_obs, 0, done)
            # print("\t\tPolicy")

        self.replay_buffer.add(obs, np.array(info['raw_action']), next_obs, 0, done)

    def _handle_vec_data(self, obs, action, next_obs, reward, done, info):
        obs_res = {}
        next_obs_res = {}
        for key in obs.keys():
            obs_res[key] = obs[key][0]
            next_obs_res[key] = next_obs[key][0]

        return obs_res, action[0], next_obs_res, reward[0], done[0], info[0]

    def collect_data(self):
        # get action
        with torch.no_grad():
            if self.env_name == 'metadrive':
                action = self.policy.act(to_torchify(self.last_obs).unsqueeze(0), deterministic=True).cpu().numpy()[0]
            elif self.env_name == 'carla':
                action = self.policy.act(to_torchify(self.last_obs), deterministic=True).cpu().numpy()
        # step: human action or policy action according to the intervention
        # show the variance of policy at the same time
        # net_std = torch.exp(self.policy.log_std.clamp(-5.0, 2.0)).sum().item()
        # obs, reward, done, info = self.env.step(action, net_std = net_std)
        obs, reward, done, info = self.env.step(action)

        # store data
        self._store_transition(self.last_obs, action, obs, reward, done, info)

        # update last_obs
        self.last_obs = obs

        if done:
            if self.env_name == 'metadrive':
                self.reset_env()
            # carla env will reset automatically

    def save_data(self, save_dir):
        # To be fixed
        assert 1 == 0, "Not implemented yet"

        sub_data = self.sub_buffer.get_all()
        human_data = self.human_data_buffer.get_all()
        print(f"Save data: {len(sub_data['actions'])} sub data, {len(human_data['actions'])} human data")
        data_dict = {
            'sub_data': sub_data, # right name
            'human_data': human_data,
            'sub_ptrs': self.sub_ptrs,
            'human_ptrs': self.human_ptrs
        }
        save_dir = save_dir + "replay_data/"
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + f"replay_data.pkl", 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"***save replay data to {save_dir}***")


    def update(self):
        self.isw_update()

    def isw_update(self):
        self.learning_steps_disc += 1
        self.learning_steps_policy += 1

        self.history_buffer = (self.sub_buffer, self.human_data_buffer)

        ## update
        idx, idx_prob, loss_policy, weights, rewards, loss_policy_bases, loss_disc, logits_pi, logits_exp = self.model.opt(self.history_buffer)

        ## log the loss
        self.writer.add_scalar('disc/loss', loss_disc, self.learning_steps_disc)
        self.writer.add_scalar('policy/loss', loss_policy, self.learning_steps_policy)
        for i, loss_base in enumerate(loss_policy_bases):
            self.writer.add_scalar(f'policy/loss_base_{i}', loss_base, self.learning_steps_policy)
        self.writer.add_scalar('disc/acc_pi', (logits_pi < 0).float().mean().item(), self.learning_steps_disc)
        self.writer.add_scalar('disc/acc_exp', (logits_exp > 0).float().mean().item(), self.learning_steps_disc)
        self.writer.add_scalar('Train/weight_max', weights.max(), self.learning_steps_policy)
        self.writer.add_scalar('Train/weight_mean', weights.mean(), self.learning_steps_policy)
        self.writer.add_scalar('Train/weight_min', weights.min(), self.learning_steps_policy)
        # log the weight and reward
        self.writer.add_scalar('Weight/human_half_mean', weights[self.batch_size//2:].mean(), self.learning_steps_policy)
        self.writer.add_scalar('Weight/sub_half_mean', weights[:self.batch_size//2].mean(), self.learning_steps_policy)
        if self.policy_algo == 'odice':
            self.writer.add_scalar('Reward/human_half_mean', rewards[self.batch_size//2:].mean(), self.learning_steps_policy)
            self.writer.add_scalar('Reward/sub_half_mean', rewards[:self.batch_size//2].mean(), self.learning_steps_policy)

        self.writer.add_scalar('Train/index', idx, self.learning_steps_policy)
        self.writer.add_scalar('Train/index_prob', idx_prob, self.learning_steps_policy)

        # log the loss bases
        self.loss_bases_list.append(loss_policy_bases)
        with open(self.loss_bases_dir + f"loss_bases.pkl", 'wb') as f:
            pickle.dump(self.loss_bases_list, f)
            

        

    def evaluate(self):
        # useless in online train
        self.policy = self.model.get_best_policy()
        self.policy.eval()
        ## new version with whole evaluation
        normalized_returns = np.array([evaluate_once(self.env, self.policy, seed) for seed in range(100, 150)])
        self.writer.add_scalar('Eval/normalized return mean', normalized_returns.mean(), self.step)

        print("---------------------------------------")
        print(f"Env: PVP, Evaluation over {self.step} episodes: score: {normalized_returns.mean():.3f}")
        print("---------------------------------------")



    # do some warm up
    def warm_up_policy(self):
        for _ in range(20): # self.epoch_policy
            exp_data = self.human_data_buffer.sample(self.batch_size)
            states = to_torchify(exp_data['observations'])
            actions = to_torchify(exp_data['actions'])

            # update the policy
            self.model.warm_up_policies(states, actions)


    def online_learn(self):

        # init something
        self.reset_env()

        # warm up the policy
        # self.policy = self.model.get_best_policy()
        self.policy = self.model.get_paticular_policy(0)
        for _ in range(200):
            self.collect_data()
        self.warm_up_policy()

        # begin real training
        print("Finishing warm up, Start real training!")
        self.reset_env()
        # self.policy = self.model.get_best_policy()
        self.policy = self.model.get_paticular_policy(0)

        # begin to learn
        while self.step < self.total_timesteps:
            # collect data
            self.collect_data()

            self.step += 1

            # save safety cost
            if self.env_name == 'metadrive':
                self.writer.add_scalar('Takeover/safety_cost', round(self.env.total_cost, 2), self.step)

            if self.step % self.train_freq == 0:
                if self.sub_buffer.size() > self.batch_size//2:
                    # update
                    self.update()
                    # self.policy = self.model.get_best_policy()
                    self.policy = self.model.get_paticular_policy(0)

                    ## log current human data size
                    self.writer.add_scalar('Buffer/human_data_size', self.human_data_buffer.size(), self.step)
                    
                else:
                    print("Buffer not enough, skip this update!")
            
                # save the model
                self.model.save_best_policy(self.writer.log_dir, self.step)
                self.model.save_paticular_policy(self.writer.log_dir, self.step, -1, 'last_policy')
                self.model.save_paticular_policy(self.writer.log_dir, self.step, 0, 'first_policy')
                

            
                

        # save the model and close
        self.writer.close()
        self.env.close()
        print("Finish the training!!!")

