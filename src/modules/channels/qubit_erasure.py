"""
Qubit erasure channel.

Loses the photon with probability epsilon in [0, 1]; the output is always
a qutrit whose level |e> = |2> flags the loss, and measuring it returns 2.

https://quantumchannelzoo.org/channel/erasure

Usage:
    QubitErasure(epsilon=0.1).apply(register, targets="B")
    bb84.key_generation(alice_eve_channel=QubitErasure(0.1))  # loss before Eve
    bb84.loss_rate()   # about epsilon
"""


from math import sqrt
from numbers import Real

from qutip import Qobj, basis, ket2dm

from .channels import QuantumChannel

__all__ = ["QubitErasure"]


def _erasure_measurement_projectors(basis_name):
    """
    Qutrit projectors in bit-0, bit-1, erasure order.
    """
    zero, one, erased = (basis(3, i) for i in range(3))
    if basis_name == "Z":
        states = [zero, one, erased]
    elif basis_name == "X":
        # |+> and |-> inside the qubit block, plus the erasure level.
        states = [(zero + one) / sqrt(2), (zero - one) / sqrt(2), erased]
    else:
        raise ValueError("basis must be 'Z' or 'X'")
    return [ket2dm(state) for state in states]


class QubitErasure(QuantumChannel):
    """
    Qubit erasure channel.
    """

    num_targets = 1
    input_dim = 2
    output_dim = 3
    erasure_outcome = 2  # measurement outcome that means "erased"

    def __init__(self, epsilon):
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be a finite real number in [0, 1]")
        self.epsilon = float(epsilon)

    def kraus_operators(self):
        dims = [[3], [2]]  # each operator maps a qubit (2) into a qutrit (3)
        return [
            # Not erased: the qubit is embedded unchanged.
            sqrt(1 - self.epsilon) * Qobj([[1, 0], [0, 1], [0, 0]], dims=dims),
            # Erased: |0> and |1> both go to the erasure level |2>.
            sqrt(self.epsilon) * Qobj([[0, 0], [0, 0], [1, 0]], dims=dims),
            sqrt(self.epsilon) * Qobj([[0, 0], [0, 0], [0, 1]], dims=dims),
        ]

    def measurement_projectors(self, basis="Z"):
        """Return projectors for register.measure(); outcome 2 means erasure."""
        return _erasure_measurement_projectors(basis)



