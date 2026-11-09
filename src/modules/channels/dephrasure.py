"""
dephrasure quantum channel

Z-dephasing p followed by erasure q, both in [0, 1]. Output is a qutrit:
N(rho) = (1-q) [(1-p) rho + p Z rho Z] + q |e><e|,  with |e> = |2>
The shared register applies the map and recognizes the erasure flag;
measurement returns 0, 1, or 2 (erasure).

https://quantumchannelzoo.org/channel/dephrasure
https://arxiv.org/abs/1806.08327

Usage:
    Dephrasure(p=0.1, q=0.2).apply(register, targets="B")
    bb84.key_generation(signal_channel=Dephrasure(p=0.05, q=0.1))
    bb84.loss_rate()   # about q
"""

from math import sqrt
from numbers import Real

from qutip import Qobj

from .channels import QuantumChannel
from .qubit_erasure import _erasure_measurement_projectors

__all__ = ["Dephrasure"]


class Dephrasure(QuantumChannel):
    """
    Dephrasure channel.        
    """

    num_targets = 1
    input_dim = 2
    output_dim = 3
    erasure_outcome = 2  # measurement outcome that means "erased"

    def __init__(self, p, q):
        for name, value in (("p", p), ("q", q)):
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite real number in [0, 1]")
        self.p = float(p)
        self.q = float(q)

    def kraus_operators(self):
        dims = [[3], [2]]  # each operator maps a qubit (2) into a qutrit (3)
        return [
            # Not erased, no phase flip: the qubit is embedded unchanged.
            sqrt((1 - self.p) * (1 - self.q))
            * Qobj([[1, 0], [0, 1], [0, 0]], dims=dims),
            # Not erased, phase flip: the qubit is embedded after Z.
            sqrt(self.p * (1 - self.q))
            * Qobj([[1, 0], [0, -1], [0, 0]], dims=dims),
            # Erased: |0> and |1> both go to the erasure level |2>.
            sqrt(self.q) * Qobj([[0, 0], [0, 0], [1, 0]], dims=dims),
            sqrt(self.q) * Qobj([[0, 0], [0, 0], [0, 1]], dims=dims),
        ]

    def measurement_projectors(self, basis="Z"):
        """Return projectors for register.measure(); outcome 2 means erasure."""
        return _erasure_measurement_projectors(basis)
