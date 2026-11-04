"""CNOT attack: Eve entangles a fresh probe with the signal and reads it after sifting."""

from qutip import basis
from qutip_qip.circuit import QubitCircuit

from .attack import Attack
from ..utils.quantum_register import ERASURE, UNKNOWN


class CNOTAttack(Attack):
    """Eve applies a CNOT from the signal (control) onto her probe |0> (target).

    After sifting she measures the probe in Alice's announced basis.
    """

    probe_labels = ("E",)

    def __init__(self, seed=43, probe_label="E"):
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
