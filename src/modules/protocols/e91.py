"""
E91 entanglement-based quantum key distribution with a CHSH Bell test.

Settings use Bloch-sphere angles (halve them for photon polarizer angles):
  "ekert" (default, Ekert's 1991 choice):
      Alice 0, pi/4, pi/2; Bob pi/4, pi/2, 3pi/4.
      Key: pairs where both chose pi/4 or both chose pi/2.
      CHSH: Alice {0, pi/2} x Bob {pi/4, 3pi/4}.
  "zx" (E91-type variant with Z/X keys, so the BB84 attacks' fixed Z/X
  readouts and FGGNP's tuned disturbance apply unchanged):
      Alice 0 (Z), pi/2 (X); Bob 0, pi/2, pi/4, 3pi/4.
      Key: Z-Z and X-X pairs. CHSH: Alice {0, pi/2} x Bob {pi/4, 3pi/4}.
  A dict {"alice": angles, "bob": angles, "chsh": ((i, j), (k, l))} gives
  custom settings; key pairs are those where both angles are equal and the
  setting pair is not used by the Bell test (its outcomes are made public).

The singlet gives opposite outcomes for equal angles, so Bob flips his bit
to form the key. The Bell test uses
    S = E(a, b) - E(a, b') + E(a', b) + E(a', b'),   E = <(-1)^(x + y)>,
with |S| <= 2 for any local model (for example after intercept-resend) and
S = -2*sqrt(2) for an ideal singlet. Eve's interaction and channel noise
pull |S| down. The test uses detected pairs only (fair sampling).

Produces sifted keys and the Bell test only; error correction,
authentication, and privacy amplification are not implemented here.
"""

from copy import deepcopy
from numbers import Integral

import numpy as np
from qutip import Qobj, basis, tensor
from qutip_qip.circuit import QubitCircuit

from .protocol import Protocol
from ..utils.discrimination import helstrom_success
from ..utils.quantum_register import ERASURE, UNKNOWN, QuantumRegister

# Preset measurement settings (see the module docstring). chsh=((i, j), (k, l))
# holds the indices of Alice's settings a, a' and Bob's settings b, b'.
SETTINGS = {
    "ekert": dict(alice=(0.0, np.pi/4, np.pi/2), bob=(np.pi/4, np.pi/2, 3*np.pi/4),
                  chsh=((0, 2), (0, 2))),
    "zx": dict(alice=(0.0, np.pi/2), bob=(0.0, np.pi/2, np.pi/4, 3*np.pi/4),
               chsh=((0, 1), (2, 3))),
}


def _basis_label(angle):
    """Return "Z" for angle 0 and "X" for pi/2, otherwise the angle as a float.

    The names let the Z/X readouts apply and make displays read M_Z and M_X.
    """
    if np.isclose(angle, 0.0, rtol=0, atol=1e-12):
        return "Z"
    if np.isclose(angle, np.pi/2, rtol=0, atol=1e-12):
        return "X"
    return float(angle)


class E91(Protocol):
    """Simulate number_pairs E91 rounds: singlet source, measurements, sifting and CHSH test."""

    party_labels = ("A", "B")  # Alice's and Bob's qubits

    def __init__(self, seed=42, number_pairs=1000, settings="ekert"):
        super().__init__(seed)

        # bool is a subclass of int, so it is rejected explicitly.
        if isinstance(number_pairs, bool) or not isinstance(number_pairs, Integral) or number_pairs < 1:
            raise ValueError("number_pairs must be a positive integer")

        self.number_pairs = int(number_pairs)
        self._set_settings(settings)
        self._has_run = False      # key_generation has been called at least once
        self._completed = False    # the latest key_generation call finished
        self.round_registers = {}  # pair index -> register copy, for record_rounds
        self._run_config = {}      # channels and attacks of the last run (for the optimal readout)
        self._new_batch()

    def _set_settings(self, settings):
        """Validate a preset name or custom dict and store the measurement settings."""
        if isinstance(settings, str):
            if settings not in SETTINGS:
                raise ValueError(f"settings must be one of {sorted(SETTINGS)} or a dict")
            name, spec = settings, SETTINGS[settings]
        elif isinstance(settings, dict):
            name, spec = "custom", settings
        else:
            raise TypeError("settings must be a preset name or a dict")

        try:
            alice = tuple(float(a) for a in spec["alice"])
            bob = tuple(float(b) for b in spec["bob"])
            (i, j), (k, l) = spec["chsh"]  # Alice's (a, a') and Bob's (b, b') setting indices
        except (KeyError, TypeError, ValueError):
            raise ValueError("Custom settings need 'alice' and 'bob' angle lists and 'chsh'=((i, j), (k, l))")

        if not alice or not bob or not np.isfinite(alice + bob).all():
            raise ValueError("Setting angles must be finite and nonempty")

        # The CHSH indices must be integers naming two different settings of each party.
        if (not all(isinstance(x, Integral) for x in (i, j, k, l)) or i == j or k == l
                or not {i, j} <= set(range(len(alice))) or not {k, l} <= set(range(len(bob)))):
            raise ValueError("chsh must name two distinct Alice settings and two distinct Bob settings by index")

        # Key pairs: equal angles on a setting pair the Bell test does not use.
        key_angles = []
        for x, a in enumerate(alice):
            for y, b in enumerate(bob):
                used_by_bell_test = x in (i, j) and y in (k, l)
                if np.isclose(a, b, rtol=0, atol=1e-12) and not used_by_bell_test:
                    key_angles.append(a)
        if not key_angles:
            raise ValueError("Alice and Bob need at least one common angle outside the Bell-test "
                             "settings to form a key")

        self.settings = name
        self.alice_setting_angles = alice
        self.bob_setting_angles = bob
        self.chsh_settings = ((int(i), int(j)), (int(k), int(l)))
        # Remove repeated angles while keeping their first-seen order.
        self.key_angles = tuple(dict.fromkeys(key_angles))

    def _new_batch(self):
        """Draw both parties' random settings. The draw order fixes every seeded result."""
        n = self.number_pairs
        self.alice_settings = self.rng.integers(0, len(self.alice_setting_angles), size=n)
        self.bob_settings = self.rng.integers(0, len(self.bob_setting_angles), size=n)
        self.alice_angles = np.asarray(self.alice_setting_angles)[self.alice_settings]
        self.bob_angles = np.asarray(self.bob_setting_angles)[self.bob_settings]
        # Public basis per pair, as Eve's readout sees it after sifting.
        self.alice_basis_labels = np.array([_basis_label(a) for a in self.alice_angles], dtype=object)
        self.alice_bits = np.full(n, -1, dtype=int)  # -1 until measured
        self.bob_bits = np.full(n, -1, dtype=int)
        self.alice_key = self.bob_key = np.array([], dtype=int)

    @staticmethod
    def _prepare(register, attacked_arm):
        """Emit the singlet (|01> - |10>)/sqrt(2): |00> -> X, X -> H on A -> CNOT A->B."""
        register.load(tensor(basis(2, 0), basis(2, 0)), labels=("A", "B"),
                      signal=attacked_arm, name="Source |00>")
        source = QubitCircuit(2)
        source.add_gate("X", targets=[0])                   # |00> -> |10>
        source.add_gate("X", targets=[1])                   # -> |11>
        source.add_gate("SNOT", targets=[0])                # -> (|0> - |1>)|1>/sqrt(2)
        source.add_gate("CNOT", controls=[0], targets=[1])  # -> (|01> - |10>)/sqrt(2)
        register.apply_circuit(source, ["A", "B"])

    def key_generation(self, attack=None, alice_channel=None, bob_channel=None, memory_channel=None, memory_targets=None, record_rounds=(0,), source_eve_channel=None, attacked_arm="B"):
        """Distribute number_pairs singlets, measure them, sift the key and run the Bell test.

        Order per pair: source -> communication circuit on the attacked arm ->
        source_eve_channel (attacked arm, before Eve) -> Eve interaction(s)
        on the attacked arm -> alice_channel on A and bob_channel on B (on
        the attacked arm, the stretch after Eve) -> memory noise on Eve's
        probes -> Alice and Bob measure -> save Eve's probes.

        attacked_arm is "B" (Bob's arm, default) or "A" (Alice's arm); Eve's
        attacks act on that qubit. Without an attack, source_eve_channel is
        simply the first stretch of that arm. A pair is used only if both
        parties detect it. Returns (alice_key, bob_key) with Bob's bits
        already flipped. Each later call generates a fresh batch.
        """
        if attacked_arm not in ("A", "B"):
            raise ValueError("attacked_arm must be 'A' (Alice's arm) or 'B' (Bob's arm)")
        records = self._validate_records(record_rounds, self.number_pairs)
        attacks = self._as_attacks(attack)
        targets = self._memory_targets(attacks, memory_channel, memory_targets)

        # The first batch was drawn in __init__; every later run draws a new one.
        if self._has_run:
            self._new_batch()
        self._has_run = True
        self._completed = False
        self.round_registers = {}
        self.attack = attack
        self.attacks = attacks
        for a in attacks:
            a.reset()

        # A copy, so later changes to these objects cannot alter this run's analysis.
        self._run_config = deepcopy(dict(alice_channel=alice_channel, bob_channel=bob_channel,
                                         memory_channel=memory_channel, memory_targets=targets,
                                         source_eve_channel=source_eve_channel, attacked_arm=attacked_arm,
                                         circuit=self.register.circuit, attacks=attacks))

        for i in range(self.number_pairs):
            # The source emits a singlet; the attacked arm passes the channel and Eve.
            self._prepare(self.register, attacked_arm)
            self.register.transmit(attacks, channel=source_eve_channel)
            self._apply_channel(alice_channel, self.register, ["A"])
            self._apply_channel(bob_channel, self.register, ["B"])
            self._apply_channel(memory_channel, self.register, targets)

            # Both parties measure; the outcome ERASURE (2) means the photon was lost.
            self.alice_bits[i] = self.register.measure("A", _basis_label(self.alice_angles[i]), self.rng)
            self.bob_bits[i] = self.register.measure("B", _basis_label(self.bob_angles[i]), self.rng)
            detected = self.alice_bits[i] != ERASURE and self.bob_bits[i] != ERASURE

            # Each attack stores its probe state for the delayed readout after sifting.
            for a in attacks:
                a.capture(self.register, i, record=i in records, detected=detected)
            if i in records:
                self.round_registers[i] = self.register.copy()

        # Pairs whose settings belong to the Bell test.
        (i, j), (k, l) = self.chsh_settings
        bell_test = np.isin(self.alice_settings, (i, j)) & np.isin(self.bob_settings, (k, l))

        # A pair is used only if both parties detected their photon.
        self.detected_mask = (self.alice_bits != ERASURE) & (self.bob_bits != ERASURE)
        self.matching_settings = np.isclose(self.alice_angles, self.bob_angles, rtol=0, atol=1e-12)

        # Bell-test outcomes are announced publicly, so those rounds never become key.
        self.key_mask = self.matching_settings & ~bell_test & self.detected_mask
        self.sifted_mask = self.key_mask  # Same role as BB84.sifted_mask.
        self.alice_key = self.alice_bits[self.key_mask]
        # Equal angles on a singlet give opposite outcomes, so Bob flips his bits.
        self.bob_key = 1 - self.bob_bits[self.key_mask]
        self.chsh_mask = bell_test & self.detected_mask
        self._completed = True
        return self.alice_key, self.bob_key

    def _require_run(self):
        if not self._completed:
            raise RuntimeError("Run key_generation first")

    def qber(self):
        """Quantum bit error rate: the fraction of key bits where Alice and Bob differ."""
        self._require_run()
        if not self.alice_key.size:
            return np.nan  # No key bits.
        return float(np.mean(self.alice_key != self.bob_key))

    def loss_rate(self):
        """Fraction of pairs lost by Alice, Bob, or both."""
        self._require_run()
        return float(np.mean(~self.detected_mask))

    def correlation(self, alice_setting, bob_setting):
        """Return (E, n): E = <(-1)^(x+y)> over n detected pairs with these setting indices."""
        self._require_run()
        sel = ((self.alice_settings == alice_setting) & (self.bob_settings == bob_setting)
               & self.detected_mask)
        n = int(sel.sum())
        if n == 0:
            return np.nan, 0

        # Bits 0/1 become signs +1/-1, so the product of the signs is (-1)^(x + y).
        signs = (1 - 2*self.alice_bits[sel]) * (1 - 2*self.bob_bits[sel])
        return float(signs.mean()), n

    def chsh(self, with_error=False):
        """S = E(a, b) - E(a, b') + E(a', b) + E(a', b') from the CHSH settings.

        |S| > 2 certifies entanglement; an ideal singlet gives -2*sqrt(2).
        with_error=True returns (S, standard error), using Var(E) = (1 - E^2)/n.
        """
        self._require_run()
        (i, j), (k, l) = self.chsh_settings

        # Each term is (sign in S, E, n).
        terms = [(+1, *self.correlation(i, k)), (-1, *self.correlation(i, l)),
                 (+1, *self.correlation(j, k)), (+1, *self.correlation(j, l))]
        S = float(sum(sign*E for sign, E, _ in terms))
        if not with_error:
            return S

        # The four terms use disjoint pairs, so their variances add.
        # The error is undefined if any term has no pairs.
        if all(n for *_, n in terms):
            error = float(np.sqrt(sum((1 - E**2)/n for _, E, n in terms)))
        else:
            error = np.nan
        return S, error

    def _reference_probe_states(self, attack, angle):
        """Eve's weighted probe states (2*p0*rho0, 2*p1*rho1) for Alice's key bit at this angle.

        rho_b is Eve's probe state given Alice's outcome b, and p_b its
        probability, both conditioned on both parties detecting the pair.
        The factor 2 lets the equal-prior Helstrom helpers return the
        correct prior-weighted result. Returns None if no pair is detected.
        """
        # Replay a pair without sampling: Eve's random choices are averaged over
        # and every channel acts on the full density matrix.
        cfg = self._run_config
        reg = QuantumRegister()
        reg.circuit = cfg["circuit"]
        self._prepare(reg, cfg["attacked_arm"])
        reg.transmit(channel=cfg["source_eve_channel"])
        for a in cfg["attacks"]:
            a.average_apply(reg)
        self._apply_channel(cfg["alice_channel"], reg, ["A"])
        self._apply_channel(cfg["bob_channel"], reg, ["B"])
        self._apply_channel(cfg["memory_channel"], reg, cfg["memory_targets"])

        # Condition on both parties detecting by projecting out each erasure level |2>.
        try:
            for label in ("A", "B"):
                if label in reg.erasure_levels:
                    reg.postselect(label, Qobj(np.diag([1, 1, 0])), name=f"{label} detects")
        except ValueError:
            return None

        # Condition on Alice's outcome; p is that outcome's probability.
        weighted = []
        for bit in (0, 1):
            branch = reg.copy()
            projector = branch.measurement_projectors("A", _basis_label(angle))[bit]
            try:
                p = branch.postselect("A", projector, name=f"Alice outcome {bit}")
            except ValueError:
                p = 0.0
            weighted.append(2*p*branch.reduced(list(attack.probe_labels)))
        return tuple(weighted)

    def eve_guesses(self, attack=None, guess_on_erasure=False, readout="fixed"):
        """Eve's delayed guesses of the key bits, aligned element-by-element with alice_key.

        readout="fixed" uses the attack's own readout, which estimates the bit
        on the qubit she attacked. On Bob's arm she then flips it, since the
        singlet makes Bob's outcome the opposite of Alice's. FGGNP's fixed
        readout covers only Z and X, so use "zx" settings or readout="optimal"
        with it. readout="optimal" makes the Helstrom measurement for Alice's
        key bit directly (attacks with quantum probes only; intercept-resend
        returns its recorded, flipped bits either way). Rounds where the probe
        gives no information (erased, or an inconclusive optimal outcome)
        return UNKNOWN=-1, or a random bit with guess_on_erasure=True.
        """
        self._require_run()
        if readout not in ("fixed", "optimal"):
            raise ValueError('readout must be "fixed" or "optimal"')

        chosen = self._select_attack(attack)

        # The optimal readout needs Eve's reference probe states for each key basis.
        references = None
        if readout == "optimal" and chosen.probe_labels:
            references = {_basis_label(a): self._reference_probe_states(chosen, a) for a in self.key_angles}

        guesses = chosen.measure_after_sifting(self.alice_basis_labels, self.key_mask,
                                               guess_on_erasure=guess_on_erasure,
                                               reference_states=references)

        # A fixed readout on Bob's arm estimates Bob's bit, the opposite of Alice's.
        if references is None and self._run_config["attacked_arm"] == "B":
            known = guesses != UNKNOWN
            guesses[known] = 1 - guesses[known]
        return guesses

    def eve_optimal_success(self, attack=None):
        """Exact best probability that Eve guesses Alice's key bit, per key basis.

        This is the Helstrom bound for her stored probes under the last run's
        channels. Keys are "Z", "X", or the angle in radians; nan if no pair
        can be detected.
        """
        self._require_run()
        chosen = self._select_attack(attack)
        if not chosen.probe_labels:
            raise ValueError("eve_optimal_success needs an attack with quantum probes")

        result = {}
        for angle in self.key_angles:
            states = self._reference_probe_states(chosen, angle)
            result[_basis_label(angle)] = np.nan if states is None else helstrom_success(*states)
        return result

    def display(self, round_index=None, attack=None, **kwargs):
        """Display the last pair, or a pair selected by record_rounds.

        For a selected pair, Eve's delayed measurements appear after calling
        eve_guesses(), if that pair was retained. With several attacks, the
        last attack's readout is shown unless attack= selects another.
        Returns a Matplotlib Figure.
        """
        self._require_run()

        if round_index is None:
            register = self.register
        else:
            if round_index not in self.round_registers:
                raise ValueError("Include this pair in key_generation(record_rounds=...)")
            register = self.round_registers[round_index]
            # Prefer an attack's copy of the round, which also contains Eve's readout.
            chosen = self.attacks if attack is None else (self._select_attack(attack),)
            for a in chosen:
                register = a.round_records.get(round_index, register)

        index = self.number_pairs - 1 if round_index is None else round_index
        kwargs.setdefault("title", f"E91 pair {index} (zero-based)")
        return register.display(**kwargs)
