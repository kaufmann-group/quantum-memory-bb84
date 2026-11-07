import secrets

import numpy as np
from qutip import basis

from .attack import Attack, _is_basis
from ..utils.quantum_register import ERASURE, UNKNOWN, QuantumRegister


class InterceptResendAttack(Attack):
    """Intercept-resend attack.

    Eve measures each signal in a basis chosen uniformly from `bases` and
    resends the state she found, for both BB84 and E91.

    bases defaults to BB84's ("Z", "X"). For E91 Eve may instead use the
    protocol's key angles, e.g. (np.pi/4, np.pi/2). In BB84 a random-basis
    intercept gives Bob a sifted QBER of 1/4 and Eve 3/4 of the sifted bits.

    Source:
        C. H. Bennett and G. Brassard, "Quantum cryptography: Public key
        distribution and coin tossing", Proc. IEEE Int. Conf. on Computers,
        Systems and Signal Processing, Bangalore, 175-179 (1984); reprinted in
        Theor. Comput. Sci. 560, 7-11 (2014).
        https://doi.org/10.1016/j.tcs.2014.05.025
        For E91: A. K. Ekert, "Quantum cryptography based on Bell's theorem",
        Phys. Rev. Lett. 67, 661-663 (1991).
        https://doi.org/10.1103/PhysRevLett.67.661

    Usage:
        bb84 = BB84(seed=1, number_bits=1000)
        bb84.key_generation(InterceptResendAttack(seed=43))
        bb84.qber()                   # about 0.25
        eve_key = bb84.eve_guesses()  # Eve's measured bit for every sifted round

        e91 = E91(seed=1, number_pairs=1000)
        e91.key_generation(InterceptResendAttack(bases=(np.pi/4, np.pi/2)))
        e91.chsh()                    # |S| <= 2: the Bell test reveals Eve
    """

    def __init__(self, seed=secrets.randbits(128), bases=("Z", "X")): # bases defaults to BB84's ("Z", "X")
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

    def measure_after_sifting(self, alice_bases, sifted_mask, guess_on_erasure=False, reference_states=None):
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
