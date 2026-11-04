"""Joint damping-dephasing qubit channel."""

from math import sqrt
from numbers import Real

from qutip import Qobj

from .channels import QuantumChannel

__all__ = ["DampingDephasing"]


class DampingDephasing(QuantumChannel):
    """
    Joint damping-dephasing quantum channel, using three Kraus operators.

    source:
        https://quantumchannelzoo.org/channel/damping_dephasing
        https://arxiv.org/abs/2405.06231 (eq. 14)

    Parameters:
        p in [0, 1/2], the dephasing probability
        g in [0, 1] the amplitude-damping probability
    """

    num_targets = 1
    input_dim = output_dim = 2

    def __init__(self, p, g):
        """
        Define the channel from its dephasing probability p and damping probability g.
        """
        if isinstance(p, bool) or not isinstance(p, Real) or not 0 <= p <= 0.5:
            raise ValueError("p must be a finite real number in [0, 0.5]")
        if isinstance(g, bool) or not isinstance(g, Real) or not 0 <= g <= 1:
            raise ValueError("g must be a finite real number in [0, 1]")
        self.p = float(p)
        self.g = float(g)

    def kraus_operators(self):
        survival = sqrt(1 - self.g)

        return [
            # No phase flip, no decay.
            sqrt(1 - self.p) * Qobj([
                [1, 0],
                [0, survival]
            ]),

            # Decay |1> -> |0>.
            Qobj([
                [0, sqrt(self.g)],
                [0, 0]
            ]),

            # Phase flip (Z) together with the no-decay part.
            sqrt(self.p) * Qobj([
                [1, 0],
                [0, -survival]
            ])
        ]
