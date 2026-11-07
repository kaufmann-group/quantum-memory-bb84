"""Phase-flip (Z-dephasing) channel.

Source: https://quantumchannelzoo.org/channel/phase-flip
Action: N(rho) = (1-p) rho + p Z rho Z, with p in [0, 1].
Coherences are multiplied by 1-2p: p=1/2 completely dephases;
p=1 applies the unitary Z and does not completely dephase.
Usage: PhaseFlip(p=0.2).apply(register, targets="E")
"""

from math import sqrt
from numbers import Real

from qutip import qeye, sigmaz

from .channels import QuantumChannel

__all__ = ["PhaseFlip"]


class PhaseFlip(QuantumChannel):
    """Phase-flip channel.

    Applies Z with probability p in [0, 1], which multiplies the coherences by
    1-2p (Z-dephasing).

    Source:
        https://quantumchannelzoo.org/channel/phase-flip

    Usage:
        PhaseFlip(p=0.2).apply(register, targets="E")
        bb84.key_generation(signal_channel=PhaseFlip(0.1))
        bb84.qber()   # about p/2: Z-basis states are unaffected
    """

    num_targets = 1
    input_dim = output_dim = 2

    def __init__(self, p):
        if isinstance(p, bool) or not isinstance(p, Real) or not 0 <= p <= 1:
            raise ValueError("p must be a finite real number in [0, 1]")
        self.p = float(p)

    def kraus_operators(self):
        return [sqrt(1 - self.p) * qeye(2), sqrt(self.p) * sigmaz()]
