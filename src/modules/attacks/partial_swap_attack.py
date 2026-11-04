"""Partial-SWAP probe attack with a tunable signal-probe interaction.

The interaction U(theta) = cos(theta)*I - i*sin(theta)*SWAP transfers quantum
information from Bob's signal to Eve's initially blank |0> probe. This is a
parameterized attack construction, not a claim of an optimal BB84 strategy.

After sifting Eve measures her probe in Alice's announced basis. The optional
BB84.eve_guesses(readout="optimal") instead gives the noise-adapted Helstrom
measurement on the stored probe.
"""

from numbers import Real

import numpy as np
from qutip import Qobj, basis, qeye

from .attack import Attack
from ..utils.quantum_register import ERASURE, UNKNOWN


class PartialSwapAttack(Attack):
    """Interact with Eve's |0> probe using exp(-i*theta*SWAP).

    theta=0 leaves the signal unchanged; theta=pi/2 swaps the signal into
    Eve's probe, up to phases. Intermediate angles continuously interpolate.
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

        # Two-qubit basis order is |00>, |01>, |10>, |11> (signal first).
        swap = Qobj([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dims=[[2, 2], [2, 2]])
        self.unitary = np.cos(self.theta)*qeye([2, 2]) - 1j*np.sin(self.theta)*swap

    def apply(self, register):
        (e,) = self.probe_labels
        register.add_ancilla(e, basis(2, 0), preparation="Init |0>")
        # Apply the two-qubit unitary to the signal and Eve's probe, in that order.
        register.apply_unitary(self.unitary, [register.signal, e],
                               name=f"PartialSWAP({self.theta:.3g})")
        return register

    def _read_probe(self, register, alice_basis):
        # Measuring in the announced basis is a fixed readout, not generally optimal.
        bit = register.measure(self.probe_labels[0], alice_basis, self.rng)
        return UNKNOWN if bit == ERASURE else bit
