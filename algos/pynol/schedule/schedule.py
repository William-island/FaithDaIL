from copy import deepcopy
from typing import Optional, Union
import numpy as np
import random
import torch
from utils.utils import to_torchify, to_cat

from .cover import FullCover





class Schedule:
    """The class to schedule base-learners with problem-independent cover.

    Args:
        ssp (SSP): A ssp instance containing a bunch of initialized
            base-learners.
        cover (Cover, optional): A cover instance deciding the
            active state of base-learners.

    """

    def __init__(self, ssp, cover = None):
        self.bases = ssp.bases
        self.cover = cover if cover is not None else FullCover(len(self.bases))
        self._t = 0
        self._optimism = None
        self.active_state = np.ones(len(self.bases))
        self.active_index = np.where(self.active_state > 0)[0]

    def opt_by_optimism(self, optimism: np.ndarray):
        """Optimize by the optimism for all base-learners.

        Args:
            optimism (numpy.ndarray): External optimism for all alive base-learners.
        """
        for idx in self.active_index:
            self.bases[idx].opt_by_optimism(optimism)

    def opt_by_gradient(self, env: None):
        """Optimize by the gradient for all base-learners.

        Args:
            env (Environment): Environment at current round.

        Returns:
            tuple: tuple contains:
                loss (float): the origin loss of all alive base-learners. \n
                surrogate_loss (float): the surrogate loss of all alive base-learners.
        """
        loss = np.zeros(len(self.active_index))
        surrogate_loss = np.zeros_like(loss)
        for i, idx in enumerate(self.active_index):
            _, loss[i], surrogate_loss[i] = self.bases[idx].opt_by_gradient(
                env)
        return loss, surrogate_loss

    @property
    def t(self):
        """Set the number of current round, get active state from ``cover`` and
        reinitialize the base-learners whose active state is 2.
        """
        return self._t

    @t.setter
    def t(self, t):
        self._t = t
        self.cover.t = t
        self.active_state = self.cover.active_state
        self.active_index = np.where(self.active_state > 0)[0]
        self.reinit_bases()

    @property
    def x_active_bases(self):
        """Get the decisions of all alive base-learners.

        Returns:
            numpy.ndarray: Decisions of all alive base learners.
        """
        return np.array([self.bases[i].x for i in self.active_index])

    @property
    def optimism(self):
        """Get the optimisms of all alive base-learners.

        Returns:
            numpy.ndarray: Optimisms of all alive base learners.
        """
        self._optimism = np.zeros_like(self.x_active_bases)
        for i, idx in enumerate(self.active_index):
            self._optimism[i] = self.bases[idx].optimism
        return self._optimism

    def reinit_bases(self):
        """Reinitialize the base-learners whose active state is 2."""
        reinit_idx = np.where(self.active_state == 2)[0]
        for idx in reinit_idx:
            self.bases[idx].reinit()


    



class Schedule_for_NonConvex(Schedule):
    """The class to schedule base-learners with their unquie data cover."""

    def __init__(self, ssp, cover = None, len_history= None):
        super().__init__(ssp, cover)

        ## init the data cover
        self.gamma_pool = [2**i for i in range(len(self.bases))]
        self.time_checkpoint = np.zeros(len(self.bases), dtype=int)
        self.len_history = len_history


    def opt_by_gradient(self, history_buffer):
        """Optimize by the gradient for all base-learners.

        Args:
            env (Environment): Environment at current round.

        Returns:
            tuple: tuple contains:
                loss (float): the origin loss of all alive base-learners. \n
                surrogate_loss (float): the surrogate loss of all alive base-learners.
        """
        loss = np.zeros(len(self.active_index))
        surrogate_loss = np.zeros_like(loss)
        for i, idx in enumerate(self.active_index):
            ## check if need to restart
            if self.t % self.gamma_pool[i] == 0:
                # restart
                self.time_checkpoint[i] = self.t


            index = [random.sample(range(self.time_checkpoint[i], self.t+1), 1)[0] for _ in range(self.len_history)]
            index.append(self.t)
            his = history_buffer.sample(index)

            ## optimize
            _, loss[i], surrogate_loss[i] = self.bases[idx].opt_by_gradient(
                his)
        return loss, surrogate_loss
    























class Schedule_for_NonConvex_wob(Schedule):
    """The class to schedule base-learners with their unquie data cover."""

    def __init__(self, ssp, cover = None, epoch = None, len_loss = None, batch_size = None):
        super().__init__(ssp, cover)

        ## init the data cover
        self.gamma_pool = [2**i for i in range(len(self.bases))]
        self.time_checkpoint = np.zeros(len(self.bases), dtype=int)
        self.epoch = epoch

        self.sub_checkpoint = [0]
        self.human_checkpoint = [0]
        self.len_loss = len_loss
        self.batch_size = batch_size


    def opt_by_gradient(self, history_buffer):
        """Optimize by the gradient for all base-learners.

        Args:
            env (Environment): Environment at current round.

        Returns:
            tuple: tuple contains:
                loss (float): the origin loss of all alive base-learners. \n
                surrogate_loss (float): the surrogate loss of all alive base-learners.
        """
        sub_buffer = history_buffer[0]
        human_data_buffer = history_buffer[1]
        
        self.sub_checkpoint.append(sub_buffer.size())
        self.human_checkpoint.append(human_data_buffer.size())
        

        loss = np.zeros(len(self.active_index))
        surrogate_loss = np.zeros_like(loss)

        ## compute loss
        for _ in range(self.len_loss):
            # sample half half data
            sub_index =np.random.choice(range(self.sub_checkpoint[-2], self.sub_checkpoint[-1]), self.batch_size//2)
            sub_data = sub_buffer.sample(self.batch_size//2, sub_index)
            half_data = human_data_buffer.sample(self.batch_size//2)
            # states = to_torchify(np.concatenate([sub_data['observations'], half_data['observations']], axis=0))
            # actions = to_torchify(np.concatenate([sub_data['actions'], half_data['actions']], axis=0))
            states =  to_cat([to_torchify(sub_data['observations']), to_torchify(half_data['observations'])], dim=0)
            actions = to_cat([to_torchify(sub_data['actions']), to_torchify(half_data['actions'])], dim=0)

            # next_states = to_torchify(np.concatenate([sub_data['next_observations'], half_data['next_observations']], axis=0))
            # terminals = to_torchify(np.concatenate([sub_data['terminals'], half_data['terminals']], axis=0))
            next_states = to_cat([to_torchify(sub_data['next_observations']), to_torchify(half_data['next_observations'])], dim=0)
            terminals = to_cat([to_torchify(sub_data['terminals']), to_torchify(half_data['terminals'])], dim=0)

            for i, idx in enumerate(self.active_index):
                l, sl = self.bases[idx].get_latest_loss(states, actions, next_states, terminals)
                loss[i] += l
                surrogate_loss[i] += sl
        loss /= self.len_loss
        surrogate_loss /= self.len_loss


        for i, idx in enumerate(self.active_index):
            print(f'Updating base learner {idx}!')
            ## check if need to restart
            if self.t % self.gamma_pool[i] == 0:
                # restart
                self.time_checkpoint[i] = self.t
                ## important change (not working)
                # self.bases[idx].reinit()


            sub_left_ptr = self.sub_checkpoint[self.time_checkpoint[i]]
            sub_right_ptr = self.sub_checkpoint[self.t+1]

            # reinit code just for metadrive
            if self.t % 12 == 0 and self.t > 0:
                # reinit
                print(f"Reinit base learner {idx}!")
                self.bases[idx].reinit()
                epoch_train = 3000
            else:
                epoch_train = self.epoch
        
            # epoch_train = self.epoch

            # if self.t % 12 == 0 and self.t > 0:
            #     # reinit
            #     print(f"Reinit base learner {idx}!")
            #     self.bases[idx].reinit()
            #     epoch_train = 3000
            # else:
            #     epoch_train = self.epoch

            for ep in range(epoch_train):
                rnd = np.random.randint(self.time_checkpoint[i]+1, self.t+2)
                human_right_ptr = self.human_checkpoint[rnd]

                ## sample data
                exp_index = np.random.choice(range(0, human_right_ptr), self.batch_size)
                exp_data = human_data_buffer.sample(self.batch_size, exp_index)
                states_exp, actions_exp = to_torchify(exp_data['observations']), to_torchify(exp_data['actions'])

                # sample half half data
                sub_index = np.random.choice(range(sub_left_ptr, sub_right_ptr), self.batch_size//2)
                sub_data = sub_buffer.sample(self.batch_size//2, sub_index)

                half_index = np.random.choice(range(0, human_right_ptr), self.batch_size//2)
                half_data = human_data_buffer.sample(self.batch_size//2, half_index)

                # states = to_torchify(np.concatenate([sub_data['observations'], half_data['observations']], axis=0))
                # actions = to_torchify(np.concatenate([sub_data['actions'], half_data['actions']], axis=0))
                states =  to_cat([to_torchify(sub_data['observations']), to_torchify(half_data['observations'])], dim=0)
                actions = to_cat([to_torchify(sub_data['actions']), to_torchify(half_data['actions'])], dim=0)

                # next_states = to_torchify(np.concatenate([sub_data['next_observations'], half_data['next_observations']], axis=0))
                # terminals = to_torchify(np.concatenate([sub_data['terminals'], half_data['terminals']], axis=0))
                next_states = to_cat([to_torchify(sub_data['next_observations']), to_torchify(half_data['next_observations'])], dim=0)
                terminals = to_cat([to_torchify(sub_data['terminals']), to_torchify(half_data['terminals'])], dim=0)

                his = {
                    'states': states,
                    'actions': actions,
                    'next_states': next_states,
                    'terminals': terminals,
                    'states_exp': states_exp,
                    'actions_exp': actions_exp,
                }
                ## optimize
                self.bases[idx].opt_by_gradient(his, ep)
        return loss, surrogate_loss
    