import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from algos.networks import ValueFunction, TwinV, DeterministicPolicy, Discriminator, GaussianPolicy
import copy
import os


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXP_ADV_MAX = 100.


def f_star(residual, name="Pearson_chi_square"):
    if name == "Reverse_KL":
        return torch.exp(residual - 1)
    elif name == "Pearson_chi_square":
        omega_star = torch.max(residual / 2 + 1, torch.zeros_like(residual))
        return residual * omega_star - (omega_star - 1)**2


def f_prime_inverse(residual, name='Pearson_chi_square'):
    if name == "Reverse_KL":
        return torch.exp(residual - 1)
    elif name == "Pearson_chi_square":
        return torch.max(residual, torch.zeros_like(residual))
    

def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


class ODICE(nn.Module):
    def __init__(self, args, observation_space, action_space):
        super().__init__()
        # make policy and value function
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = args.features_dim
        self.hidden_dim = args.hidden_dim
        self.n_hidden = args.n_hidden
        self.layer_norm = args.layer_norm
        self.use_twin_v = args.use_twin_v
        self.value_lr = args.value_lr
        self.policy_lr = args.policy_lr

        # choose the policy
        # policy = DeterministicPolicy(observation_space, args.features_dim, action_space, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
        # vf = TwinV(observation_space,  args.features_dim, layer_norm=args.layer_norm, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden) if args.use_twin_v else ValueFunction(args.obs_dim, layer_norm=args.layer_norm, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
        policy = DeterministicPolicy(self.observation_space, self.features_dim, self.action_space, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
        vf = TwinV(self.observation_space,  self.features_dim, layer_norm=self.layer_norm, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden) if self.use_twin_v else ValueFunction(self.observation_space, layer_norm=self.layer_norm, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
        
        self.vf = vf.to(DEFAULT_DEVICE)
        self.vf_target = copy.deepcopy(vf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=self.value_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.state_feature = []
        self.Lambda = args.Lambda
        self.eta = args.eta
        self.f_name = args.f_name
        self.discount = args.discount
        self.beta = args.beta
        self.step = 0

        # other parameters
        self.update_type = args.type

    def update(self, observations, actions, next_observations, rewards, terminals, writer=None):
        if self.update_type == 'orthogonal_true_g':
            return self.orthogonal_true_g_update(observations, actions, next_observations, rewards, terminals, writer)


    def orthogonal_true_g_update(self, observations, actions, next_observations, rewards, terminals, writer=None):
        # the network will NOT update
        with torch.no_grad():
            target_v = self.vf_target(observations)
            target_v_next = self.vf_target(next_observations)

        v = self.vf.both(observations) if self.use_twin_v else self.vf(observations)
        v_next = self.vf.both(next_observations) if self.use_twin_v else self.vf(next_observations)

        forward_residual = rewards + (1. - terminals.float()) * self.discount * target_v_next - v
        backward_residual = rewards + (1. - terminals.float()) * self.discount * v_next - target_v
        forward_dual_loss = torch.mean(self.Lambda * f_star(forward_residual, self.f_name))
        backward_dual_loss = torch.mean(self.Lambda * self.eta * f_star(backward_residual, self.f_name))
        pi_residual = forward_residual.clone().detach()
        td_mean, td_min, td_max = torch.mean(forward_residual), torch.min(forward_residual), torch.max(forward_residual)

        self.v_optimizer.zero_grad(set_to_none=True)
        forward_grad_list, backward_grad_list = [], []
        forward_dual_loss.backward(retain_graph=True)
        for param in list(self.vf.parameters()):
            forward_grad_list.append(param.grad.clone().detach().reshape(-1))
        backward_dual_loss.backward()
        for i, param in enumerate(list(self.vf.parameters())):
            backward_grad_list.append(param.grad.clone().detach().reshape(-1) - forward_grad_list[i])
        forward_grad, backward_grad = torch.cat(forward_grad_list), torch.cat(backward_grad_list)
        parallel_coef = (torch.dot(forward_grad, backward_grad) / max(torch.dot(forward_grad, forward_grad), 1e-10)).item()  # avoid zero grad caused by f*
        forward_grad = (1 - parallel_coef) * forward_grad + backward_grad

        param_idx = 0
        for i, grad in enumerate(forward_grad_list):
            forward_grad_list[i] = forward_grad[param_idx: param_idx + grad.shape[0]]
            param_idx += grad.shape[0]

        self.v_optimizer.zero_grad(set_to_none=True)
        torch.mean((1 - self.Lambda) * v).backward()
        for i, param in enumerate(list(self.vf.parameters())):
            param.grad += forward_grad_list[i].reshape(param.grad.shape)

        self.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(self.vf_target, self.vf, self.beta)

        # Update policy
        weight = f_prime_inverse(pi_residual, self.f_name)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()

        # weight = torch.ones_like(weight).to(DEFAULT_DEVICE)
        # print(f"weight: {weight.mean()}")
        policy_out = self.policy(observations)

        # bc_losses = -policy_out.log_prob(actions)
        bc_losses = F.mse_loss(policy_out, actions, reduction='none').sum(-1)

        policy_loss = torch.mean(weight * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        # wandb
        # if (self.step + 1) % 100 == 0:
        #     # use the writer to log the values
        #     writer.add_scalar('Train/v_value', v.mean(), self.step)
        #     writer.add_scalar('Train/weight_max', weight.max(), self.step)
        #     writer.add_scalar('Train/weight_min', weight.min(), self.step)
        #     writer.add_scalar('Train/td_mean', td_mean, self.step)
        #     writer.add_scalar('Train/td_min', td_min, self.step)
        #     writer.add_scalar('Train/td_max', td_max, self.step)

        #     writer.add_scalar('Train/policy_loss', policy_loss, self.step)

        # self.step += 1

        return policy_loss.item()
    
    def just_get_loss(self, observations, actions, next_observations, rewards, terminals):
        # the network will NOT update
        with torch.no_grad():
            target_v_next = self.vf_target(next_observations)

            v = self.vf.both(observations) if self.use_twin_v else self.vf(observations)

            forward_residual = rewards + (1. - terminals.float()) * self.discount * target_v_next - v

            pi_residual = forward_residual.clone().detach()
        

            # Update policy
            weight = f_prime_inverse(pi_residual, self.f_name)
            weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()

            # weight = torch.ones_like(weight).to(DEFAULT_DEVICE)
            # print(f"weight: {weight.mean()}")
            policy_out = self.policy(observations)
            # bc_losses = -policy_out.log_prob(actions)
            bc_losses = F.mse_loss(policy_out, actions, reduction='none').sum(-1)
            policy_loss = torch.mean(weight * bc_losses)
        
        return policy_loss.item()
    
    def policy_bc_update(self, observations, actions):
        policy_out = self.policy(observations)
        bc_losses = F.mse_loss(policy_out, actions, reduction='none').sum(-1).mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        bc_losses.backward()
        self.policy_optimizer.step()


    def get_activation(self):
        def hook(model, input, output):
            self.state_feature.append(output.detach())
        return hook

    def save(self, model_dir, step, sub_dir = "models/"):
        checkpoint = {
            'step': step,
            'vf': self.vf.state_dict(),
            'vf_target': self.vf_target.state_dict(),
            'policy': self.policy.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
        }
        save_dir = model_dir + sub_dir
        os.makedirs(save_dir, exist_ok=True)
        torch.save(checkpoint, save_dir + f"/checkpoint_{step}.pth")
        print(f"***save models to {model_dir}***")


    def load(self, model_dir, step):
        checkpoint = torch.load(model_dir + f"/checkpoint_{step}.pth")
        # self.step = checkpoint['step']
        self.vf.load_state_dict(checkpoint['vf'])
        self.vf_target.load_state_dict(checkpoint['vf_target'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        print(f"***load the model from {model_dir}***")


    def reinit(self):
        policy = DeterministicPolicy(self.observation_space, self.features_dim, self.action_space, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
        vf = TwinV(self.observation_space,  self.features_dim, layer_norm=self.layer_norm, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden) if self.use_twin_v else ValueFunction(self.observation_space, layer_norm=self.layer_norm, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
        
        self.vf = vf.to(DEFAULT_DEVICE)
        self.vf_target = copy.deepcopy(vf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=self.value_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)































# GP for gussian policy
class ODICE_GP(nn.Module):
    def __init__(self, args, observation_space, action_space):
        super().__init__()
        # make policy and value function
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = args.features_dim
        self.hidden_dim = args.hidden_dim
        self.n_hidden = args.n_hidden
        self.layer_norm = args.layer_norm
        self.use_twin_v = args.use_twin_v
        self.value_lr = args.value_lr
        self.policy_lr = args.policy_lr

        # choose the policy
        # policy = DeterministicPolicy(observation_space, args.features_dim, action_space, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
        # vf = TwinV(observation_space,  args.features_dim, layer_norm=args.layer_norm, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden) if args.use_twin_v else ValueFunction(args.obs_dim, layer_norm=args.layer_norm, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
        policy = GaussianPolicy(self.observation_space, self.features_dim, self.action_space, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
        vf = TwinV(self.observation_space,  self.features_dim, layer_norm=self.layer_norm, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden) if self.use_twin_v else ValueFunction(self.observation_space, layer_norm=self.layer_norm, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
        
        self.vf = vf.to(DEFAULT_DEVICE)
        self.vf_target = copy.deepcopy(vf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=self.value_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.state_feature = []
        self.Lambda = args.Lambda
        self.eta = args.eta
        self.f_name = args.f_name
        self.discount = args.discount
        self.beta = args.beta
        self.step = 0

        # other parameters
        self.update_type = args.type

    def update(self, observations, actions, next_observations, rewards, terminals, writer=None):
        if self.update_type == 'orthogonal_true_g':
            return self.orthogonal_true_g_update(observations, actions, next_observations, rewards, terminals, writer)


    def orthogonal_true_g_update(self, observations, actions, next_observations, rewards, terminals, writer=None):
        # the network will NOT update
        with torch.no_grad():
            target_v = self.vf_target(observations)
            target_v_next = self.vf_target(next_observations)

        v = self.vf.both(observations) if self.use_twin_v else self.vf(observations)
        v_next = self.vf.both(next_observations) if self.use_twin_v else self.vf(next_observations)

        forward_residual = rewards + (1. - terminals.float()) * self.discount * target_v_next - v
        backward_residual = rewards + (1. - terminals.float()) * self.discount * v_next - target_v
        forward_dual_loss = torch.mean(self.Lambda * f_star(forward_residual, self.f_name))
        backward_dual_loss = torch.mean(self.Lambda * self.eta * f_star(backward_residual, self.f_name))
        pi_residual = forward_residual.clone().detach()
        td_mean, td_min, td_max = torch.mean(forward_residual), torch.min(forward_residual), torch.max(forward_residual)

        self.v_optimizer.zero_grad(set_to_none=True)
        forward_grad_list, backward_grad_list = [], []
        forward_dual_loss.backward(retain_graph=True)
        for param in list(self.vf.parameters()):
            forward_grad_list.append(param.grad.clone().detach().reshape(-1))
        backward_dual_loss.backward()
        for i, param in enumerate(list(self.vf.parameters())):
            backward_grad_list.append(param.grad.clone().detach().reshape(-1) - forward_grad_list[i])
        forward_grad, backward_grad = torch.cat(forward_grad_list), torch.cat(backward_grad_list)
        parallel_coef = (torch.dot(forward_grad, backward_grad) / max(torch.dot(forward_grad, forward_grad), 1e-10)).item()  # avoid zero grad caused by f*
        forward_grad = (1 - parallel_coef) * forward_grad + backward_grad

        param_idx = 0
        for i, grad in enumerate(forward_grad_list):
            forward_grad_list[i] = forward_grad[param_idx: param_idx + grad.shape[0]]
            param_idx += grad.shape[0]

        self.v_optimizer.zero_grad(set_to_none=True)
        torch.mean((1 - self.Lambda) * v).backward()
        for i, param in enumerate(list(self.vf.parameters())):
            param.grad += forward_grad_list[i].reshape(param.grad.shape)

        self.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(self.vf_target, self.vf, self.beta)

        # Update policy
        weight = f_prime_inverse(pi_residual, self.f_name)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        policy_out = self.policy(observations)

        bc_losses = -policy_out.log_prob(actions)
        # bc_losses = F.mse_loss(policy_out, actions, reduction='none').sum(-1)

        policy_loss = torch.mean(weight * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        # wandb
        # if (self.step + 1) % 100 == 0:
        #     # use the writer to log the values
        #     writer.add_scalar('Train/v_value', v.mean(), self.step)
        #     writer.add_scalar('Train/weight_max', weight.max(), self.step)
        #     writer.add_scalar('Train/weight_min', weight.min(), self.step)
        #     writer.add_scalar('Train/td_mean', td_mean, self.step)
        #     writer.add_scalar('Train/td_min', td_min, self.step)
        #     writer.add_scalar('Train/td_max', td_max, self.step)

        #     writer.add_scalar('Train/policy_loss', policy_loss, self.step)

        # self.step += 1

        return policy_loss.item()
    
    def just_get_loss(self, observations, actions, next_observations, rewards, terminals):
        # the network will NOT update
        with torch.no_grad():
            target_v_next = self.vf_target(next_observations)

            v = self.vf.both(observations) if self.use_twin_v else self.vf(observations)

            forward_residual = rewards + (1. - terminals.float()) * self.discount * target_v_next - v

            pi_residual = forward_residual.clone().detach()
        

            # Update policy
            weight = f_prime_inverse(pi_residual, self.f_name)
            weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
            policy_out = self.policy(observations)
            # bc_losses = -policy_out.log_prob(actions)
            bc_losses = F.mse_loss(policy_out, actions, reduction='none').sum(-1)
            policy_loss = torch.mean(weight * bc_losses)
        
        return policy_loss.item()
    
    def policy_bc_update(self, observations, actions):
        policy_out = self.policy(observations)
        bc_losses = F.mse_loss(policy_out, actions, reduction='none').sum(-1).mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        bc_losses.backward()
        self.policy_optimizer.step()


    def get_activation(self):
        def hook(model, input, output):
            self.state_feature.append(output.detach())
        return hook

    def save(self, model_dir, step, sub_dir = "models/"):
        checkpoint = {
            'step': step,
            'vf': self.vf.state_dict(),
            'vf_target': self.vf_target.state_dict(),
            'policy': self.policy.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
        }
        save_dir = model_dir + sub_dir
        os.makedirs(save_dir, exist_ok=True)
        torch.save(checkpoint, save_dir + f"/checkpoint_{step}.pth")
        print(f"***save models to {model_dir}***")


    def load(self, model_dir, step):
        checkpoint = torch.load(model_dir + f"/checkpoint_{step}.pth")
        # self.step = checkpoint['step']
        self.vf.load_state_dict(checkpoint['vf'])
        self.vf_target.load_state_dict(checkpoint['vf_target'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        print(f"***load the model from {model_dir}***")


    def reinit(self):
        policy = GaussianPolicy(self.observation_space, self.features_dim, self.action_space, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
        vf = TwinV(self.observation_space,  self.features_dim, layer_norm=self.layer_norm, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden) if self.use_twin_v else ValueFunction(self.observation_space, layer_norm=self.layer_norm, hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
        
        self.vf = vf.to(DEFAULT_DEVICE)
        self.vf_target = copy.deepcopy(vf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=self.value_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)

