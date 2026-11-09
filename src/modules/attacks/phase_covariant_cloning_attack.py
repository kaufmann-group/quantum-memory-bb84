"""
Phase-covariant cloning attack.

Eve runs an economical, asymmetric 1-to-2 phase-covariant cloner on each
signal: Bob receives one approximate copy and Eve keeps the other, which
she measures in Alice's announced basis after sifting.

theta controls the asymmetric tradeoff between Bob's and Eve's fidelities:
theta=0 gives Bob a perfect copy; theta=pi/2 gives Eve a perfect copy.

Source:
    https://doi.org/10.1103/PhysRevA.60.2764
    https://doi.org/10.1103/PhysRevA.62.012302

Usage:
    bb84 = BB84(seed=1, number_bits=1000)
    bb84.key_generation(PhaseCovariantCloningAttack(theta=np.pi/4))
    bb84.qber()                    # about (1 - cos(theta))/2 ≈ 0.15
    guesses = bb84.eve_guesses()
    (guesses == bb84.alice_key).mean()   # about (1 + sin(theta))/2 ≈ 0.85
"""


from numbers import Real
import secrets

import numpy as np
from qutip import Qobj, basis, qeye, sigmax, tensor

from .attack import Attack, _is_basis
from ..utils.quantum_register import ERASURE, UNKNOWN


class PhaseCovariantCloningAttack(Attack):
    """
    Phase-covariant cloning attack.
    """

    probe_labels = ("E",)

    def __init__(self, theta=np.pi / 4, seed=secrets.randbits(128), probe_label="E"):
        if (isinstance(theta, bool) or not isinstance(theta, Real)
                or not np.isfinite(theta) or not 0 <= theta <= np.pi / 2):
            raise ValueError("theta must be a finite real number in [0, pi/2]")
        super().__init__(seed)
        # Per-instance label, so this attack can be chained with another one.
        self.probe_labels = self._check_labels(probe_label, 1)
        self.theta = float(theta)

        # With a blank |0> probe, the equatorial cloning unitary acts as:
        #   |00> -> |00>
        #   |10> -> cos(theta)|10> + sin(theta)|01>.
        # Its action on |01> completes an orthonormal two-dimensional rotation.
        c, s = np.cos(self.theta), np.sin(self.theta)
        clone_xy = Qobj([[1, 0, 0, 0],
                         [0, c, s, 0],
                         [0, -s, c, 0],
                         [0, 0, 0, 1]], dims=[[2, 2], [2, 2]])

        # Rotate BB84's X-Z great circle to the equator, clone, then rotate
        # both output copies back so Bob and Eve use the original BB84 bases.
        rotate = (qeye(2) - 1j*sigmax()) / np.sqrt(2)
        self.unitary = (tensor(rotate.dag(), rotate.dag())
                        * clone_xy * tensor(rotate, qeye(2)))

    def apply(self, register):
        (e,) = self.probe_labels
        register.add_ancilla(e, basis(2, 0), preparation="Init |0>")
        register.apply_unitary(self.unitary, [register.signal, e],
                               name=f"PhaseClone({self.theta:.3g})")
        return register

    def _read_probe(self, register, alice_basis):
        if not _is_basis(alice_basis):
            raise ValueError("Alice's basis must be X, Z, or a finite measurement angle")
        # Both clones were rotated back into the original BB84 coordinate system.
        bit = register.measure(self.probe_labels[0], alice_basis, self.rng)
        return UNKNOWN if bit == ERASURE else bit
