"""BB84 prepare-and-measure quantum key distribution.

Alice sends each random bit in a random basis: |0>/|1> (Z) or |+>/|-> (X).
Bob measures in his own random basis, and the rounds where the bases match
and Bob detected the photon form the sifted key. Optional attacks and noise
channels act on the signal and on Eve's probes. Error correction and privacy
amplification are not implemented here.
"""

from copy import deepcopy
from numbers import Integral
import secrets

import numpy as np
from qutip import Qobj, basis
from qutip_qip.circuit import QubitCircuit

from .protocol import Protocol
from ..utils.discrimination import helstrom_success
from ..utils.quantum_register import ERASURE, QuantumRegister


class BB84(Protocol):
    """
    Simulate `number_bits` BB84 rounds, optionally with attacks and noise channels.
    """

    def __init__(self, seed=secrets.randbits(128), number_bits=1000):
        super().__init__(seed)

        if isinstance(number_bits, bool) or not isinstance(number_bits, Integral) or number_bits < 1:
            raise ValueError("number_bits must be a positive integer")

        self.number_bits = int(number_bits)
        self._has_run = False      # key_generation has been called at least once
        self._completed = False    # the latest key_generation call finished
        self.round_registers = {}  # round index -> register copy, for record_rounds
        self._run_config = {}      # channels and attacks of the last run (for the optimal readout)
        self._new_batch()

    def _new_batch(self):
        """
        Draw Alice's bits and both parties' bases. The draw order fixes every seeded result.
        """

        self.alice_bits = self.rng.integers(0, 2, size=self.number_bits)
        self.alice_bases = self.rng.choice(["Z", "X"], size=self.number_bits)
        self.bob_bases = self.rng.choice(["Z", "X"], size=self.number_bits)
        self.bob_bits = np.full(self.number_bits, -1, dtype=int)  # -1 until Bob measures
        self.alice_key = self.bob_key = np.array([], dtype=int)

    @staticmethod
    def _prepare(register, bit, alice_basis):
        """Load Alice's BB84 state for one round into the register."""
        register.load(basis(2, 0), name="Alice |0>")
        preparation = QubitCircuit(1)
        if bit == 1:
            preparation.add_gate("X", targets=[0])     # |0> -> |1>
        if alice_basis == "X":
            preparation.add_gate("SNOT", targets=[0])  # Hadamard: |0> -> |+>, |1> -> |->
        register.apply_circuit(preparation, "B")

    def key_generation(self, attack=None, signal_channel=None, memory_channel=None, memory_targets=None, record_rounds=(0,), alice_eve_channel=None):
        """Run one batch; one attack (or a list applied in order) and optional channels.

        Order: Alice -> communication circuit -> alice_eve_channel -> Eve
        interaction(s) -> signal_channel (Eve to Bob) -> memory noise -> Bob
        measurement -> save Eve's probe. Without an attack the two channel
        segments simply compose. A loss (erasure) before Eve leaves her
        interaction with nothing to act on, and the signal stays lost.
        Independent probe noise commutes with Bob's local measurement.
        Each later call generates a fresh batch and clears attack records.
        """
        records = self._validate_records(record_rounds, self.number_bits)
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
        self._run_config = deepcopy(dict(alice_eve_channel=alice_eve_channel, signal_channel=signal_channel, memory_channel=memory_channel, memory_targets=targets, circuit=self.register.circuit, attacks=attacks))

        for i in range(self.number_bits):
            # Alice prepares the signal, which then travels through the channels and Eve.
            self._prepare(self.register, self.alice_bits[i], self.alice_bases[i])
            self.register.transmit(attacks, channel=alice_eve_channel, mark_regions=True)
            if signal_channel is not None or memory_channel is not None:
                self.register.mark_region("Quantum channels")
            self._apply_channel(signal_channel, self.register, ["B"])
            self._apply_channel(memory_channel, self.register, targets)

            # Bob measures; the outcome ERASURE (2) means the photon was lost.
            self.register.mark_region("Measurement")
            self.bob_bits[i] = self.register.measure("B", self.bob_bases[i], self.rng)

            # Each attack stores its probe state for the delayed readout after sifting.
            for a in attacks:
                a.capture(self.register, i, record=i in records,
                          detected=self.bob_bits[i] != ERASURE)
            if i in records:
                self.round_registers[i] = self.register.copy()

        # Sifting: keep the rounds where the bases match and Bob detected the photon.
        self.matching_bases = self.alice_bases == self.bob_bases
        self.detected_mask = self.bob_bits != ERASURE
        self.sifted_mask = self.matching_bases & self.detected_mask
        self.alice_key = self.alice_bits[self.sifted_mask]
        self.bob_key = self.bob_bits[self.sifted_mask]
        self._completed = True
        return self.alice_key, self.bob_key

    def qber(self):
        """
        Quantum bit error rate: the fraction of sifted bits where Alice and Bob differ.
        """

        if not self._completed:
            raise RuntimeError("Run key_generation first")
        if not self.alice_key.size:
            return np.nan  # No sifted bits.
        return float(np.mean(self.alice_key != self.bob_key))

    def loss_rate(self):
        """
        Fraction of rounds in which Bob's photon was lost.
        """

        if not self._completed:
            raise RuntimeError("Run key_generation first")
        
        return float(np.mean(self.bob_bits == ERASURE))

    def _reference_probe_states(self, attack, alice_basis):
        """
        Eve's weighted probe states (2*p0*rho0, 2*p1*rho1) for Alice's bit in one basis.

        rho_b is Eve's probe state for Alice's bit b, averaged over everything
        Eve does not know and conditioned on Bob detecting the signal, as in
        every sifted round. p_b is the share of bit b among detected rounds;
        it differs from 1/2 when loss depends on the bit. The factor 2 lets the
        equal-prior Helstrom helpers return the prior-weighted result, as in
        E91. Returns None if Bob can never detect the signal.
        """
        cfg = self._run_config
        states, detected = [], []
        for bit in (0, 1):
            # Replay the round without sampling: Eve's random choices are averaged
            # over and every channel acts on the full density matrix.
            reg = QuantumRegister()
            reg.circuit = cfg["circuit"]
            self._prepare(reg, bit, alice_basis)
            reg.transmit(channel=cfg["alice_eve_channel"])
            for a in cfg["attacks"]:
                a.average_apply(reg)
            self._apply_channel(cfg["signal_channel"], reg, ["B"])
            self._apply_channel(cfg["memory_channel"], reg, cfg["memory_targets"])

            # Condition on Bob detecting the photon by projecting out the erasure level |2>.
            p_detect = 1.0
            if "B" in reg.erasure_levels:
                try:
                    p_detect = reg.postselect("B", Qobj(np.diag([1, 1, 0])), name="Bob detects")
                except ValueError:
                    p_detect = 0.0  # Bob never detects this bit.

            states.append(reg.reduced(list(attack.probe_labels)))
            detected.append(p_detect)

        total = sum(detected)
        if total <= 1e-12:
            return None
        # p_b = detected[b] / total, so each entry is 2*p_b*rho_b.
        return tuple(2*p/total*rho for p, rho in zip(detected, states))

    def eve_guesses(self, attack=None, guess_on_erasure=False, readout="fixed"):
        """Eve's delayed guesses, aligned element-by-element with alice_key.

        With several attacks, choose one by instance or list index.
        readout="fixed" uses the attack's own readout rule. readout="optimal"
        makes the Helstrom measurement, the best single-round guess of
        Alice's bit for the probes' actual (noisy) states. It applies to
        attacks with quantum probes; intercept-resend has no probe to
        optimize, so its guesses are the same either way. Rounds where the
        probe gives no information (erased, or an inconclusive optimal
        outcome) return UNKNOWN=-1, or a random bit with guess_on_erasure=True.
        """
        if not self._completed:
            raise RuntimeError("Run key_generation first")
        if readout not in ("fixed", "optimal"):
            raise ValueError('readout must be "fixed" or "optimal"')

        chosen = self._select_attack(attack)

        # The optimal readout needs Eve's reference probe states for each basis.
        references = None
        if readout == "optimal" and chosen.probe_labels:
            references = {b: self._reference_probe_states(chosen, b) for b in ("Z", "X")}

        return chosen.measure_after_sifting(self.alice_bases, self.sifted_mask, guess_on_erasure=guess_on_erasure, reference_states=references)

    def eve_optimal_success(self, attack=None):
        """Exact best probability that Eve guesses a sifted bit, per basis.

        This is the Helstrom bound for her stored probes under the last run's
        channels. It needs no sampling, so it is a noise-free benchmark for
        eve_guesses(). Returns {"Z": p, "X": p}; nan if Bob never detects.
        """
        if not self._completed:
            raise RuntimeError("Run key_generation first")
        chosen = self._select_attack(attack)
        if not chosen.probe_labels:
            raise ValueError("eve_optimal_success needs an attack with quantum probes")

        result = {}
        for b in ("Z", "X"):
            states = self._reference_probe_states(chosen, b)
            result[b] = np.nan if states is None else helstrom_success(*states)
        return result

    def display(self, round_index=None, attack=None, **kwargs):
        """Display the last round, or a round selected by record_rounds.

        For a selected round, delayed Eve measurements appear after calling
        eve_guesses() (or attack.measure_after_sifting), if that round was
        retained. With several attacks, the last attack's readout is shown
        unless attack= selects another. Returns a Matplotlib Figure.
        """
        if not self._completed:
            raise RuntimeError("Run key_generation first")

        if round_index is None:
            register = self.register
        else:
            if round_index not in self.round_registers:
                raise ValueError("Include this round in key_generation(record_rounds=...)")
            register = self.round_registers[round_index]
            # Prefer an attack's copy of the round, which also contains Eve's readout.
            chosen = self.attacks if attack is None else (self._select_attack(attack),)
            for a in chosen:
                register = a.round_records.get(round_index, register)

        index = self.number_bits - 1 if round_index is None else round_index
        kwargs.setdefault("title", f"BB84 round {index} (zero-based)")
        return register.display(**kwargs)
