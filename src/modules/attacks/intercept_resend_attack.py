"""Intercept-resend attack: Eve measures the signal and forwards the collapsed state."""

import numpy as np
from qutip import basis

from .attack import Attack, _is_basis
from ..utils.quantum_register import ERASURE, UNKNOWN, QuantumRegister


class InterceptResendAttack(Attack):
    """Eve measures the signal in a basis chosen uniformly from `bases` and resends.

    bases defaults to BB84's ("Z", "X"). For E91, Eve may instead use the
    protocol's key angles, e.g. InterceptResendAttack(bases=(np.pi/4, np.pi/2)).
    """

    def __init__(self, seed=43, bases=("Z", "X")):
        bases = tuple(bases)
        if not bases or not all(_is_basis(b) for b in bases):
            raise ValueError("bases must be a nonempty list of 'X', 'Z' or finite angles in radians")
        self.bases = bases
        super().__init__(seed)

    def reset(self):
        super().reset()
        self.eve_bits = []   # Eve's measured bit in each round (ERASURE if the photon was lost)
        self.eve_bases = []  # the basis Eve chose in each round

    def apply(self, register):
        # Draws exactly as rng.choice(["Z", "X"]) did, so seeded BB84 runs are unchanged.
        eve_basis = self.bases[int(self.rng.integers(len(self.bases)))]
        bit = register.measure(register.signal, eve_basis, self.rng)
        self.eve_bits.append(bit)
        self.eve_bases.append(eve_basis)
        return register

    def average_apply(self, register):
        """Measure-and-resend averaged over Eve's basis and outcome (no RNG used)."""
        # A scratch qubit register, used only to build the measurement projectors.
        qubit = QuantumRegister().load(basis(2, 0))

        # Each of the n bases is chosen with probability 1/n, so every projector P
        # becomes the Kraus operator sqrt(1/n)*P of one averaged channel.
        weight = np.sqrt(1 / len(self.bases))
        kraus = [weight*P for b in self.bases for P in qubit.measurement_projectors("B", b)]
        return register.apply_kraus(kraus, register.signal, name="Intercept-resend (avg)")

    def measure_after_sifting(self, alice_bases, sifted_mask, guess_on_erasure=False,
                              reference_states=None):
        """Eve's recorded bits; reference_states is ignored (no quantum probe to optimize)."""
        _, mask = self._sifting_inputs(alice_bases, sifted_mask)

        # Eve's guess for each sifted round is simply the bit she measured.
        guesses = np.array(self.eve_bits, dtype=int)[mask]

        # Where Eve's own measurement found the photon lost, she has no bit.
        missing = guesses == ERASURE
        if guess_on_erasure:
            guesses[missing] = self.rng.integers(0, 2, size=missing.sum())
        else:
            guesses[missing] = UNKNOWN

        self.guesses = guesses
        return guesses.copy()
