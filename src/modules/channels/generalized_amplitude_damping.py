"""Generalized amplitude damping channel.

Relaxes a qubit at rate gamma toward the thermal state diag(1-N, N), i.e.
amplitude damping into a finite-temperature environment.

A_{gamma,N}, using all four Kraus operators from the source.

https://quantumchannelzoo.org/channel/generalized_amplitude

Usage:
    GeneralizedAmplitudeDamping(gamma=0.2, N=0.1).apply(register, "E")
    bb84.key_generation(FGGNPAttack(), memory_channel=GeneralizedAmplitudeDamping(0.2, 0.1))
"""

from math import sqrt
from numbers import Real

from qutip import Qobj

from .channels import QuantumChannel

__all__ = ["GeneralizedAmplitudeDamping"]


class GeneralizedAmplitudeDamping(QuantumChannel):
    """
    Generalized amplitude damping channel.
    """

    num_targets = 1
    input_dim = output_dim = 2

    def __init__(self, gamma, N):
        for name, value in (("gamma", gamma), ("N", N)):
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite real number in [0, 1]")
        self.gamma = float(gamma)
        self.N = float(N)

    def kraus_operators(self):
        survival = sqrt(1 - self.gamma)
        return [
            # With weight 1-N: amplitude damping toward |0>.
            sqrt(1 - self.N) * Qobj([[1, 0], [0, survival]]),
            Qobj([[0, sqrt(self.gamma * (1 - self.N))], [0, 0]]),  # |1> -> |0>
            # With weight N: the mirror image, pumping toward |1>.
            sqrt(self.N) * Qobj([[survival, 0], [0, 1]]),
            Qobj([[0, 0], [sqrt(self.gamma * self.N), 0]]),  # |0> -> |1>
        ]
