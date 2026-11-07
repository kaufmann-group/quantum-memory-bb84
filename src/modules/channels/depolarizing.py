"""Qubit depolarizing channel, using the Zoo's Pauli-error parameter p.

Source: https://quantumchannelzoo.org/channel/depolarizing
Action: N(rho) = (1-p)rho + (p/3)(XrhoX + YrhoY + ZrhoZ).
p is in [0, 1]. In (1-q)rho + q*Tr(rho)*I/2, q = 4*p/3.
Consequently, complete depolarization is p=3/4, NOT p=1.

Usage: Depolarizing(p=0.15).apply(register, targets="E")
"""

from math import sqrt
from numbers import Real

from qutip import qeye, sigmax, sigmay, sigmaz

from .channels import QuantumChannel

__all__ = ["Depolarizing"]


class Depolarizing(QuantumChannel):
    """Depolarizing channel.

    Replaces the qubit with a random Pauli error: each of X, Y and Z occurs
    with probability p/3, so complete depolarization is p = 3/4, not p = 1.

    Source:
        https://quantumchannelzoo.org/channel/depolarizing

    Usage:
        Depolarizing(p=0.15).apply(register, targets="E")
        bb84.key_generation(signal_channel=Depolarizing(0.15))
        bb84.qber()   # about 2p/3 = 0.10
    """

    num_targets = 1
    input_dim = output_dim = 2

    def __init__(self, p):
        if isinstance(p, bool) or not isinstance(p, Real) or not 0 <= p <= 1:
            raise ValueError("p must be a finite real number in [0, 1]")
        self.p = float(p)

    def kraus_operators(self):
        # Kraus weights are square roots of the probabilities 1-p and p/3 (for each of X, Y, Z).
        weight = sqrt(self.p / 3)
        return [sqrt(1 - self.p) * qeye(2), weight * sigmax(), weight * sigmay(), weight * sigmaz()]
