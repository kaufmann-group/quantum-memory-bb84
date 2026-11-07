"""
Attack abstract base class.

Runs Eve's interaction with each signal, stores her probes after Bob's
measurement, and reads them out once Alice announces her bases.
Subclasses implement apply() (Eve's interaction with the signal) and, if
they keep quantum probes, _read_probe() (Eve's measurement after sifting).
"""

from abc import ABC, abstractmethod
from numbers import Real

import numpy as np

from ..utils.discrimination import helstrom_projectors
from ..utils.quantum_register import QuantumRegister, UNKNOWN


def _is_basis(b):
    """
    A public basis is "X", "Z", or a measurement angle in the x-z plane (radians).
    """

    if isinstance(b, str):
        return b in ("X", "Z")
    return isinstance(b, Real) and not isinstance(b, bool) and bool(np.isfinite(b))


def _same_state(a, b):
    """
    Compare RNG states, which may hold NumPy arrays (e.g. MT19937).
    """

    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return False
        return all(_same_state(a[k], b[k]) for k in a)
    return bool(np.array_equal(a, b))


class Attack(ABC):
    """
    Attack abstract base class.
    """

    probe_labels = ()  # register labels of Eve's quantum probes (none for intercept-resend)

    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """
        Clear the previous batch while keeping the RNG stream advancing.
        """

        self.probe_states = []           # Eve's reduced probe state for every round
        self.probe_erasure_levels = []   # erasure flags of those probes for every round
        self.round_records = {}          # round index -> register to display
        self._stored_round_records = {}  # round index -> full register copy at capture time
        self.round_count = 0
        self.detected = []  # Bob's per-round detection, reported by the protocol.
        self.guesses = np.array([], dtype=int)

    @staticmethod
    def _check_labels(labels, count):
        """
        Validate per-instance probe labels (when attacks are chained).
        """

        labels = (labels,) if isinstance(labels, str) else tuple(labels)
        if (len(labels) != count or len(set(labels)) != count
                or not all(isinstance(label, str) and label for label in labels)):
            raise ValueError(f"Supply {count} distinct, nonempty string probe label(s)")
        return labels

    @abstractmethod
    def apply(self, register: QuantumRegister):
        pass

    def __call__(self, register):
        return self.apply(register)

    def average_apply(self, register):
        """
        Apply the attack averaged over Eve's own random choices.

        Used to compute the reference probe states for the optimal readout.
        The default suits attacks whose apply() draws no random numbers (the
        CNOT and FGGNP attacks). If apply() draws from self.rng, the RNG is
        restored and NotImplementedError asks the subclass to override this.
        """
        
        state = self.rng.bit_generator.state
        self.apply(register)

        # A changed RNG state means apply() drew random numbers.
        if not _same_state(self.rng.bit_generator.state, state):
            self.rng.bit_generator.state = state
            raise NotImplementedError(f"{type(self).__name__}.apply uses randomness; override average_apply")
        return register

    def capture(self, register, round_index, record=False, detected=True):
        """
        Called by BB84 after memory noise and Bob's measurement.

        Conditioning the simulated probe state on Bob's sampled result retains
        joint measurement statistics. Eve's measurement policy receives only
        the public basis, not Bob's bit. `detected` is public (Bob announces
        losses) and is used only to reject masks that include lost rounds.
        """
        if round_index != self.round_count:
            raise ValueError("Capture rounds sequentially, beginning at zero")
        self.detected.append(bool(detected))

        if self.probe_labels:
            # Keep only Eve's probes (everything else is traced out) and their erasure flags.
            self.probe_states.append(register.reduced(self.probe_labels).copy())
            self.probe_erasure_levels.append({label: register.erasure_levels[label] for label in self.probe_labels if label in register.erasure_levels})

        # Recorded rounds keep the whole register so they can be displayed later.
        if record:
            self._stored_round_records[round_index] = register.copy()
            self.round_records[round_index] = self._stored_round_records[round_index]
        self.round_count += 1

    def _probe_register(self, index):
        """
        Return a fresh register holding round `index`'s stored probes, ready to measure.
        """

        # A recorded round reuses its full register, so its drawing shows the whole round.
        if index in self._stored_round_records:
            return self._stored_round_records[index].copy()
        return QuantumRegister().load(self.probe_states[index], labels=self.probe_labels, signal=self.probe_labels[0], erasure_levels=self.probe_erasure_levels[index], name="Stored probe")

    def _read_probe(self, register, alice_basis):
        raise NotImplementedError("This attack has no delayed quantum readout")

    def _read_optimal(self, register, projectors):
        """
        Joint Helstrom measurement on all probes; inconclusive gives UNKNOWN.
        """
        
        outcome = register.measure(list(self.probe_labels), projectors, self.rng, name="Helstrom")
        # Outcome 2 is helstrom_projectors' inconclusive projector.
        return UNKNOWN if outcome == 2 else outcome

    def _sifting_inputs(self, alice_bases, sifted_mask):
        """Validate the public bases and sifting mask; return them as arrays."""
        bases = np.asarray(alice_bases, dtype=object)  # Keep "Z"/"X" strings and float angles as they are.
        mask = np.asarray(sifted_mask, dtype=bool)
        if bases.ndim != 1 or mask.shape != bases.shape or len(bases) != self.round_count:
            raise ValueError("Basis and sifted-mask arrays must match the captured rounds")
        if not all(_is_basis(b) for b in bases):
            raise ValueError("Alice's bases must be 'X', 'Z', or finite measurement angles in radians")
        if (mask & ~np.asarray(self.detected, dtype=bool)).any():
            raise ValueError("The mask includes rounds Bob lost to erasure, so Eve's guesses would not align with the sifted key. Pass BB84.sifted_mask (not matching_bases), or call BB84.eve_guesses().")
        return bases, mask

    def measure_after_sifting(self, alice_bases, sifted_mask, guess_on_erasure=False, reference_states=None):
        """Return Eve's guesses for the rounds in BB84.sifted_mask.

        The result aligns element-by-element with BB84.alice_key. Prefer
        BB84.eve_guesses(), which passes the correct arrays automatically.
        Missing required probe outcomes give UNKNOWN=-1. With
        guess_on_erasure=True they give a fair random guess.

        By default Eve uses this attack's fixed readout, which is not optimized
        for each noise channel. With reference_states={basis: (rho0, rho1)}
        (supplied by BB84.eve_guesses(readout="optimal")), she instead makes
        the Helstrom measurement: the best single-round guess of Alice's bit.
        """
        bases, mask = self._sifting_inputs(alice_bases, sifted_mask)

        # One Helstrom measurement per public basis (None: Bob never detects in that basis).
        optimal = None
        if reference_states is not None:
            optimal = {b: helstrom_projectors(*states) for b, states in reference_states.items() if states is not None}

        guesses = []
        for i in np.flatnonzero(mask):  # indices of the sifted rounds
            probe = self._probe_register(i)
            probe.annotate(f"Public basis: {bases[i]}", self.probe_labels)

            if optimal is None:
                bit = self._read_probe(probe, bases[i])
            else:
                bit = self._read_optimal(probe, optimal[bases[i]])

            if bit == UNKNOWN and guess_on_erasure:
                bit = int(self.rng.integers(0, 2))
                probe.annotate(f"Random guess: {bit}", self.probe_labels)
            guesses.append(bit)

            # For recorded rounds, keep the measured register so display() shows the readout.
            if i in self._stored_round_records:
                probe.annotate(f"Eve guess: {bit}", self.probe_labels)
                self.round_records[i] = probe

        self.guesses = np.array(guesses, dtype=int)
        return self.guesses.copy()
