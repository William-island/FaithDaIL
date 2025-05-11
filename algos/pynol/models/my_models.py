import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from algos.base_algos.odice import ODICE
from algos.pynol.schedule import UniformSSP
from algos.pynol.schedule import Schedule_for_NonConvex
from algos.pynol.meta import Hedge

from algos.networks import Discriminator






class iFTPL_Dp_policy:
    """Implementation of Adaptive Online Learning in Dynamic Environments.
    """

    def __init__(self,
                 T: int,
                 device = None,
                 args = None,
                 observation_space = None,
                 action_space = None):
        N = int(np.floor(np.log2(T)))+1
        if args.policy_algo == 'odice':
            NonConvex_algo = OGD_ODICE_for_NonConvex
        ssp = UniformSSP(
            NonConvex_algo,
            N,
            args = args,
            device = device,
            observation_space = observation_space,
            action_space = action_space)
        self.schedule = Schedule_for_NonConvex(ssp, len_history=args.len_history)

        lr = np.array([5 for _ in range(T)])
        self.meta = Hedge(prob=np.ones(len(ssp)) / len(ssp), lr=lr)

        self.t = 0

        # other parameters
        self.device = device
        
    def opt(self, history_buffer):
        """The optimization process of the base algorithm.

        All algorithms are divided into two parts:
        :meth:`~pynol.learner.models.Model.opt_by_optimism` at the beginning of
        current round and :meth:`~pynol.learner.models.Model.opt_by_gradient`
        at the end of current round.

        Args:
            env (Environment): Environment at the current round.

        Returns:
            tuple: tuple contains:
                x (numpy.ndarray): Decision at the current round. \n
                loss (float): Origin loss at the current round. \n
                surrogate_loss (float): the surrogate loss at the current round.
        """
        self.opt_by_optimism(None)
        return self.opt_by_gradient(history_buffer)
    
    def opt_by_optimism(self, optimism: Optional[np.ndarray]):
        """Optimize by the optimism.

        Args:
            optimism (numpy.ndarray, optional): External optimism at the beginning of the
                current round.

        Returns:
            None
        """
        # self.optimism_env = optimism
        # variables = vars(self)

        self.schedule.t = self.t
        self.meta.active_state = self.schedule.active_state
        # if self.optimism_base is not None and self.optimism_base.is_external:
        #     optimism_base = self.optimism_base.compute_optimism_base(variables)
        # else:
        #     optimism_base = self.internal_optimism_base
        self.schedule.opt_by_optimism(None)
        # if self.optimism_meta is not None and self.optimism_meta.is_external:
        #     optimism_meta = self.optimism_meta.compute_optimism_meta(variables)
        # else:
        #     optimism_meta = self.internal_optimism_meta
        self.meta.opt_by_optimism(None)

    
    def opt_by_gradient(self, history_buffer):
        """Optimize by the true gradient.

        Args:
            env (Environment): env as the data dictionary.

        Returns:
            tuple: tuple contains:
                x (numpy.ndarray): the decision at the current round. \n
                loss (float): the origin loss at the current round. \n
                surrogate_loss (float): the surrogate loss at the current round.
        """
        # combine bases
        self.x_bases = self.schedule.bases
        # idx, loss, weights, rewards, loss_disc, logits_pi, logits_exp = self._get_loss(self.meta.prob, history_buffer)
        idx, idx_prob, loss, weights, rewards, loss_disc, logits_pi, logits_exp = self._get_loss(self.meta.prob, history_buffer)
        
        # update bases
        self.loss_bases, self.surrogate_loss_bases = self.schedule.opt_by_gradient(history_buffer)

        # update meta
        prob = self.meta.opt_by_gradient(self.loss_bases, loss)
        # print(prob)

        self.t += 1
        # return idx, loss, weights, rewards, self.loss_bases, loss_disc, logits_pi, logits_exp
        return idx, idx_prob, loss, weights, rewards, self.loss_bases, loss_disc, logits_pi, logits_exp

    def get_best_policy(self):
        self.x_bases = self.schedule.bases
        idx = np.argmax(self.meta.prob)
        x_base = self.x_bases[idx]
        return x_base.policy
    
    def save_best_policy(self, logdir, step):
        idx = np.argmax(self.meta.prob)
        self.x_bases[idx].save(logdir, step)

    def get_paticular_policy(self, idx):
        self.x_bases = self.schedule.bases
        x_base = self.x_bases[idx]
        return x_base.policy
    
    def save_paticular_policy(self, logdir, step, idx):
        self.x_bases = self.schedule.bases
        x_base = self.x_bases[idx]
        x_base.save(logdir, step)

    def warm_up_policies(self, states, actions):
        self.x_bases = self.schedule.bases
        for i in range(len(self.x_bases)):
            x_base = self.x_bases[i]
            x_base.warm_up_policy(states, actions)
    
        
    def _get_loss(self, prob, history_buffer):
        latest = history_buffer.sample(self.t)
        '''define of latest:
           env = {'states': states, 'actions': actions, 'next_states': next_states, 'terminals': terminals,
           'states_exp': states_exp, 'actions_exp': actions_exp}
        '''
        states = latest['states']
        actions = latest['actions']
        next_states = latest['next_states']
        terminals = latest['terminals']
        states_exp = latest['states_exp']
        actions_exp = latest['actions_exp']

        ## choose one base learner according to the prob
        # idx = np.random.choice(len(self.x_bases), p=prob)
        idx = np.argmax(self.meta.prob)
        idx_prob = np.random.choice(len(self.x_bases), p=prob)

        # choose the biggest one
        # idx = np.argmax(prob)
        x_base = self.x_bases[idx]
        with torch.no_grad():
            loss_disc, logits_pi, logits_exp = x_base.disc_loss(states, actions, states_exp, actions_exp)
            loss_policy, weights, rewards = x_base.policy_loss(states, actions, next_states, terminals)
            
        
        # return idx, loss_policy, weights, rewards, loss_disc, logits_pi, logits_exp
        return idx, idx_prob, loss_policy, weights, rewards, loss_disc, logits_pi, logits_exp
    






class OGD_ODICE_for_NonConvex(ODICE):
    """Implementation of Online Gradient Descent. With discriminator as base learners

    ``OGD`` stands for Online Gradient Descent, the most popular algorithm for
    online learning. `OGD` updates the decision :math:`x_{t+1}` by

    """

    def __init__(self,
                 args = None,
                 device = None,
                 observation_space = None,
                 action_space = None):
        super().__init__(args,observation_space, action_space)
        self.t = 0
        self.device = device

        self.step_size = args.disc_lr
        self.disc_gradient_steps = args.disc_gradient_steps
        self.policy_gradient_steps = args.policy_gradient_steps

        ## init discriminator
        # self.obs_dim = args.obs_dim
        # self.act_dim = args.act_dim
        self.disc_lr = args.disc_lr
        self.disc = Discriminator(observation_space, args.features_dim, action_space).to(self.device)
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.disc_lr)

    def opt_by_optimism(self, optimism: Optional[np.ndarray] = None):
        """Optimize by the optimism.

        Args:
            optimism (numpy.ndarray, optional): External optimism at the beginning of the
                current round.

        Returns:
            None
        """
        pass

    def get_step_size(self):
        """Get the step size at each round.

        Returns:
            float: Step size at the current round.
        """
        return self.step_size[self.t] if hasattr(self.step_size,
                                                 '__len__') else self.step_size

    def disc_loss(self, states, actions, states_exp, actions_exp):
        with torch.no_grad():
            logits_pi = self.disc(states, actions)
            logits_exp = self.disc(states_exp, actions_exp)
            loss_pi = -F.logsigmoid(-logits_pi).mean()
            loss_exp = -F.logsigmoid(logits_exp).mean()
            loss_disc = loss_exp + loss_pi
        return loss_disc.item(), logits_pi, logits_exp
    
    def policy_loss(self, states, actions, next_states, terminals):
        # compute reward
        weights = self.disc.calculate_weight(states, actions).squeeze()
        rewards = torch.log(weights)
        loss =  self.just_get_loss(states, actions, next_states, rewards, terminals)
        return loss, weights, rewards



    def update_disc(self, states, actions, states_exp, actions_exp):
        # assert self.disc_gradient_steps == len(actions)-1
        assert self.disc_gradient_steps <= len(actions)-1

        self.disc.train()
        for i in range(self.disc_gradient_steps):
            state = states[i]
            action = actions[i]
            state_exp = states_exp[i]
            action_exp = actions_exp[i]

            self.disc_optimizer.zero_grad()
            logits_pi = self.disc(state, action)
            logits_exp = self.disc(state_exp, action_exp)
            loss_pi = -F.logsigmoid(-logits_pi).mean()
            loss_exp = -F.logsigmoid(logits_exp).mean()
            loss = loss_exp + loss_pi
            loss.backward()
            self.disc_optimizer.step()
    
    def update_policy(self, states, actions, next_states, terminals):
        assert self.policy_gradient_steps == len(actions)-1

        self.policy.train()
        for i in range(self.policy_gradient_steps):
            state = states[i]
            action = actions[i]
            next_state = next_states[i]
            terminal = terminals[i]

            weight = self.disc.calculate_weight(state, action).squeeze()
            reward = torch.log(weight)
            super().update(state, action, next_state, reward, terminal)

    def warm_up_policy(self, states, actions):
        self.policy.train()
        for _ in range(self.policy_gradient_steps):
            # performe bc once
            super().policy_bc_update(states, actions)


    def opt_by_gradient(self, his: dict):
        states = his['states']
        actions = his['actions']
        next_states = his['next_states']
        terminals = his['terminals']
        states_exp = his['states_exp']
        actions_exp = his['actions_exp']

        # save the updated loss
        # loss_disc = self.disc_loss(states[-1], actions[-1], states_exp[-1], actions_exp[-1])
        loss_policy, _, _ = self.policy_loss(states[-1], actions[-1], next_states[-1], terminals[-1])

        ## batch optimize mutiple times
        self.update_disc(states, actions, states_exp, actions_exp)
        self.update_policy(states, actions, next_states, terminals)

            

        return None , loss_policy, None