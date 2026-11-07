"""General qubit Pauli channel, with the Zoo's probability ordering.

Source: https://quantumchannelzoo.org/channel/pauli
Action: N(rho) = p1*rho + p2*XrhoX + p3*YrhoY + p4*ZrhoZ.
The four probabilities correspond to I, X, Y, Z and sum to one.
Only roundoff within 1e-12 of a unit sum is normalized.
Usage: Pauli(p1=0.7, p2=0.1, p3=0.1, p4=0.1).apply(register, "E")
"""

from math import fsum, isclose, sqrt
from numbers import Real

from qutip import qeye, sigmax, sigmay, sigmaz

from .channels import QuantumChannel

__all__ = ["Pauli"]


class Pauli(QuantumChannel):
    """Pauli channel.

    Applies I, X, Y or Z with probabilities p1, p2, p3 and p4, which must
    sum to one.

    Four explicit probabilities in I, X, Y, Z order.

    Source:
        https://quantumchannelzoo.org/channel/pauli

    Usage:
        Pauli(p1=0.7, p2=0.1, p3=0.1, p4=0.1).apply(register, "E")
        e91.key_generation(alice_channel=Pauli(0.97, 0.01, 0.01, 0.01))
    """

    num_targets = 1
    input_dim = output_dim = 2

    def __init__(self, p1, p2, p3, p4):
        values = (p1, p2, p3, p4)
        for name, value in zip(("p1", "p2", "p3", "p4"), values):
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite real number in [0, 1]")

        # fsum returns a correctly rounded sum (no accumulated roundoff).
        total = fsum(float(value) for value in values)
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("p1 + p2 + p3 + p4 must equal 1")

        # Divide by the total to remove that small roundoff from the stored probabilities.
        self.p1, self.p2, self.p3, self.p4 = (float(value) / total for value in values)

    def kraus_operators(self):
        return [
            sqrt(self.p1) * qeye(2),
            sqrt(self.p2) * sigmax(),
            sqrt(self.p3) * sigmay(),
            sqrt(self.p4) * sigmaz(),
        ]
