"""
qubit erasure quantum channel

source:
    https://quantumchannelzoo.org/channel/erasure

The channel maps a qubit to a qutrit with |e> = basis(3, 2). The shared
QuantumRegister applies its rectangular (3x2) Kraus operators. QuantumChannel.apply
marks the output flag, so register.measure(target, "Z" or "X") recognizes it
and returns 2 for erasure. The optional measurement_projectors method remains
available for explicit measurements. Applying it again to an already-erased
output acts only on the surviving qubit block, so erasures compose:
P(erased) = 1 - (1 - epsilon1)(1 - epsilon2).
"""

from math import sqrt
from numbers import Real

from qutip import Qobj, basis, ket2dm

from .channels import QuantumChannel

__all__ = ["QubitErasure"]


def _erasure_measurement_projectors(basis_name):
    """Qutrit projectors in bit-0, bit-1, erasure order."""
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
    """Erase with probability epsilon in [0, 1]; output is always a qutrit."""

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



