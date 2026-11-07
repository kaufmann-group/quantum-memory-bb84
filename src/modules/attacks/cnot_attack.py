"""
CNOT attack.

Eve applies a CNOT from the signal (control) onto her blank probe |0>
(target) and, after sifting, measures the probe in Alice's announced basis.
Z-basis signals are copied perfectly, while X-basis signals become
entangled with the probe, so Bob's sifted QBER is about 1/4 and Eve
guesses about 3/4 of the sifted bits.

https://doi.org/10.1103/PhysRevA.71.042312 Section 3 and Figure 1/3

Usage:
    bb84 = BB84(seed=1, number_bits=1000)
    bb84.key_generation(CNOTAttack())
    bb84.qber()                            # about 0.25
    eve_key = bb84.eve_guesses()           # probe read in Alice's basis
    bb84.eve_guesses(readout="optimal")    # Helstrom readout instead

    # Chain with another attack by giving the probe its own label and seed.
    bb84.key_generation([InterceptResendAttack(seed=1), CNOTAttack(seed=2, probe_label="E2")])
"""

import secrets

from qutip import basis
from qutip_qip.circuit import QubitCircuit

from .attack import Attack
from ..utils.quantum_register import ERASURE, UNKNOWN


class CNOTAttack(Attack):
    """
    CNOT Attack.
    """

    probe_labels = ("E",)

    def __init__(self, seed=secrets.randbits(128), probe_label="E"):
        super().__init__(seed)
        # Per-instance label, so this attack can be chained with another one.
        self.probe_labels = self._check_labels(probe_label, 1)

        # Wire 0 is the signal (control) and wire 1 is Eve's probe (target).
        self.circuit = QubitCircuit(2)
        self.circuit.add_gate("CNOT", controls=[0], targets=[1])
        self.unitary = self.circuit.compute_unitary()

    def apply(self, register):
        (e,) = self.probe_labels  # The single probe label.
        register.add_ancilla(e, basis(2, 0), preparation="Init |0>")
        register.apply_circuit(self.circuit, [register.signal, e])
        return register

    def _read_probe(self, register, alice_basis):
        # Measure the probe in Alice's public basis; a lost probe gives no guess.
        bit = register.measure(self.probe_labels[0], alice_basis, self.rng)
        return UNKNOWN if bit == ERASURE else bit
