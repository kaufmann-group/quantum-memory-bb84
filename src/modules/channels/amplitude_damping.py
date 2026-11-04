"""
amplitude damping quantum channel

source:
    https://quantumchannelzoo.org/channel/amplitude-damping

description:
    relaxation of |1> to |0> with probability gamma in [0, 1]
    K0 = |0><0| + sqrt(1-gamma)|1><1|,  K1 = sqrt(gamma)|0><1|

usage:
    AmplitudeDamping(gamma=0.2).apply(register, targets="E")
"""

from math import sqrt
from numbers import Real

from qutip import Qobj

from .channels import QuantumChannel

__all__ = ["AmplitudeDamping"]


class AmplitudeDamping(QuantumChannel):
    """A_gamma, with gamma in [0, 1]. Construct a new instance for new parameters."""

    num_targets = 1
    input_dim = output_dim = 2

    def __init__(self, gamma):
        if isinstance(gamma, bool) or not isinstance(gamma, Real) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be a finite real number in [0, 1]")
        self.gamma = float(gamma)

    def kraus_operators(self):
        return [
            Qobj([[1, 0], [0, sqrt(1 - self.gamma)]]),  # no decay: |1> amplitude shrinks
            Qobj([[0, sqrt(self.gamma)], [0, 0]]),      # decay: |1> -> |0>
        ]
