"""
The "optimal Eve" individual attack via Fuchs, Gisin, Griffiths, Niu and Peres
(FGGNP), built as a three-qubit circuit on the signal and two probes E and F

Eve's two probes E and F interact with the signal through two CNOTs, tuned
so that Bob sees the chosen error rate `disturbance` while Eve gains the
most information about each bit that an attack on one signal at a time
allows. After sifting she guesses with E's Z outcome (Z basis) or the parity of
E's and F's X outcomes (X basis).

https://journals.aps.org/pra/pdf/10.1103/PhysRevA.56.1163
https://journals.aps.org/pra/pdf/10.1103/PhysRevA.56.1173 Figure 1

Usage:
    bb84 = BB84(seed=1, number_bits=1000)
    bb84.key_generation(FGGNPAttack(disturbance=0.10))
    bb84.qber()                    # about 0.10
    eve_key = bb84.eve_guesses()   # fixed readout described above
    bb84.eve_optimal_success()     # exact Helstrom bound per basis

    # FGGNP's fixed readout needs Z/X key bases, so use E91's "zx" settings.
    e91 = E91(seed=1, number_pairs=1000, settings="zx")
    e91.key_generation(FGGNPAttack())
"""

from numbers import Real
import secrets

import numpy as np
from qutip import basis
from qutip_qip.circuit import QubitCircuit

from .attack import Attack
from ..utils.quantum_register import ERASURE, UNKNOWN


class FGGNPAttack(Attack):
    """
    Fuchs-Gisin-Griffiths-Niu-Peres optimal individual attack.
    """

    probe_labels = ("E", "F")

    def __init__(self, disturbance=0.10, seed=secrets.randbits(128), probe_labels=("E", "F")):
        if isinstance(disturbance, bool) or not isinstance(disturbance, Real) or not 0 <= disturbance <= .5:
            raise ValueError("Disturbance must be a finite real number in [0, 0.5]")
        super().__init__(seed)
        # Per-instance labels, so this attack can be chained with another one.
        self.probe_labels = self._check_labels(probe_labels, 2)

        # D is the error rate the attack causes in Bob's sifted bits.
        self.D = float(disturbance)

        # Probe states, tuned by D:
        #   |e0> = sqrt(1 - delta)|0> + sqrt(delta)|1>, with delta = 1/2 - sqrt(D(1 - D))
        #   |f0> = sqrt(1 - D)|0> + sqrt(D)|1>
        delta = .5 - np.sqrt(self.D*(1-self.D))
        zero, one = basis(2, 0), basis(2, 1)
        self.e0 = np.sqrt(1-delta)*zero + np.sqrt(delta)*one
        self.f0 = np.sqrt(1-self.D)*zero + np.sqrt(self.D)*one

        # Wire 0 is the signal, wire 1 is probe E and wire 2 is probe F:
        # a CNOT from the signal onto E, then a CNOT from F onto the signal.
        self.circuit = QubitCircuit(3)
        self.circuit.add_gate("CNOT", controls=[0], targets=[1])
        self.circuit.add_gate("CNOT", controls=[2], targets=[0])
        self.unitary = self.circuit.compute_unitary()

    def apply(self, register):
        E, F = self.probe_labels
        register.add_ancilla(E, self.e0, preparation="Init |e0>")
        register.add_ancilla(F, self.f0, preparation="Init |f0>")
        # This note lets the display combine both probe reads into one guess.
        register.annotate("FGGNP probe register", self.probe_labels)
        register.apply_circuit(self.circuit, [register.signal, E, F])
        return register

    def _read_probe(self, register, alice_basis):
        E, F = self.probe_labels
        if not (isinstance(alice_basis, str) and alice_basis in ("Z", "X")):
            raise ValueError(f"FGGNP's fixed readout is defined only for the Z and X bases (got {alice_basis!r}); use readout='optimal' instead")

        # Z basis: Eve's guess is E's Z outcome.
        if alice_basis == "Z":
            e = register.measure(E, "Z", self.rng)
            return UNKNOWN if e == ERASURE else e

        # X basis: Eve's guess is the parity of E's and F's X outcomes.
        e = register.measure(E, "X", self.rng)
        f = register.measure(F, "X", self.rng)
        if f == ERASURE:
            return UNKNOWN  # E alone holds no information about Alice's bit.
        if e == ERASURE:
            return f  # F alone is still right with probability > 1/2.
        return e ^ f
