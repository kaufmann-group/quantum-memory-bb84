"""
Bit-flip Quantum Channel

Apply X with probability p where p is the flip probability
N(rho) = (1-p) rho + p X rho X, with p in [0, 1]

https://quantumchannelzoo.org/channel/bit-flip

Usage:
    BitFlip(p=0.2).apply(register, targets="B")
    bb84.key_generation(signal_channel=BitFlip(0.05))
    bb84.qber()   # about p/2: X-basis states are unaffected
"""

from math import sqrt
from numbers import Real

from qutip import qeye, sigmax

from .channels import QuantumChannel

__all__ = ["BitFlip"]


class BitFlip(QuantumChannel):
    """
    Bit flip quantum channel
    """

    num_targets = 1
    input_dim = output_dim = 2

    def __init__(self, p):
        if isinstance(p, bool) or not isinstance(p, Real) or not 0 <= p <= 1:
            raise ValueError("p must be a finite real number in [0, 1]")
        self.p = float(p)

    def kraus_operators(self):
        return [sqrt(1 - self.p) * qeye(2), sqrt(self.p) * sigmax()]
