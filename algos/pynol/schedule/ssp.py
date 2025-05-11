import copy

import numpy as np


class SSP:
    """The simplest class to initialize base-learners, which accepts the base
    instances as the input directly.

    Args:
        bases (list): List of base-learners.
    """
    def __init__(self, bases: list = None):
        self.bases = bases

    def __add__(self, ssp):
        new_ssp = copy.deepcopy(self)
        new_ssp.bases = self.bases + ssp.bases if self.bases is not None else ssp.bases
        return new_ssp

    def __len__(self):
        return len(self.bases)
    


class UniformSSP(SSP):
    def __init__(self,
                 base_class: None,
                 num_bases: int,
                 **kwargs_base):
        bases = [
            base_class(**kwargs_base) for _ in range(num_bases)
        ]
        super().__init__(bases)