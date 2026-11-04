"""Economical asymmetric 1-to-2 phase-covariant cloning attack for BB84.

An equatorial (X-Y plane) cloner is conjugated by a single-qubit X rotation
so that it clones the actual BB84 states in the X-Z plane. Bob receives the
first approximate clone; Eve stores the second as her probe. The interaction
uses no additional ancilla beyond Eve's blank qubit.

For the four BB84 states, the ideal clone fidelities are
    F_B = (1 + cos(theta))/2, F_E = (1 + sin(theta))/2.
The symmetric cloner uses theta=pi/4. This is not a universal cloning machine.

References:
    Bruß et al., Physical Review A 62, 012302 (2000).
    Niu and Griffiths, Physical Review A 60, 2764 (1999).
"""

from numbers import Real

import numpy as np
from qutip import Qobj, basis, qeye, sigmax, tensor

from .attack import Attack, _is_basis
from ..utils.quantum_register import ERASURE, UNKNOWN


class PhaseCovariantCloningAttack(Attack):
    """Approximately copy each BB84 state into Bob's signal and Eve's probe.

    theta controls the asymmetric tradeoff between Bob's and Eve's fidelities:
    theta=0 gives Bob a perfect copy; theta=pi/2 gives Eve a perfect copy.
    """

    probe_labels = ("E",)

    def __init__(self, theta=np.pi / 4, seed=43, probe_label="E"):
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
