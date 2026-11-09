"""
Noise-adaptive Helstrom attack.

Wraps a probe-based attack and replaces Eve's fixed probe readout with the
Helstrom measurement: the minimum-error measurement for the probes' actual
noisy states, given the public basis and Bob's detection.

The default interaction is FGGNPAttack. An existing CNOTAttack or another
probe-based Attack can be provided instead, including custom probe labels.
It is a readout strategy, not a new signal interaction, so its guesses
come only from eve_guesses(readout="optimal") (BB84 or E91).

Source:
    https://doi.org/10.1007/BF01007479

Usage:
    bb84 = BB84(seed=1, number_bits=1000)
    attack = NoiseAdaptiveHelstromAttack(CNOTAttack())
    bb84.key_generation(attack, memory_channel=Depolarizing(0.1))
    eve_key = bb84.eve_guesses(readout="optimal")
    bb84.eve_optimal_success()   # exact Helstrom success probability per basis
"""

import secrets

from .attack import Attack
from .fggnp_attack import FGGNPAttack


class NoiseAdaptiveHelstromAttack(Attack):
    """
    Noise-adaptive Helstrom attack.
    """

    probe_labels = ("E", "F")

    def __init__(self, interaction=None, seed=secrets.randbits(128)):
        if interaction is None:
            interaction = FGGNPAttack(seed=seed)
        if not isinstance(interaction, Attack) or not interaction.probe_labels:
            raise ValueError("interaction must be a probe-based Attack instance")
        if isinstance(interaction, NoiseAdaptiveHelstromAttack):
            raise ValueError("Do not wrap another NoiseAdaptiveHelstromAttack")

        self.interaction = interaction
        self.probe_labels = self._check_labels(interaction.probe_labels, len(interaction.probe_labels))
        super().__init__(seed)

    def apply(self, register):
        # Reuse exactly the underlying probe interaction; it remains Eve's attack.
        return self.interaction.apply(register)

    def average_apply(self, register):
        # BB84 replays the same interaction for each Alice bit to obtain the
        # conditional noisy probe states. Random interactions must implement
        # their own correct average_apply() method.
        return self.interaction.average_apply(register)

    def _read_probe(self, register, alice_basis):
        raise ValueError("NoiseAdaptiveHelstromAttack requires reference states; "
                         "use BB84.eve_guesses(readout='optimal')")

    def measure_after_sifting(self, alice_bases, sifted_mask, guess_on_erasure=False,
                              reference_states=None):
        """Use the noise-conditioned states supplied by BB84's optimal readout."""
        if reference_states is None:
            raise ValueError("Noise-adaptive readout needs BB84's conditional probe "
                             "states; call BB84.eve_guesses(readout='optimal')")
        return super().measure_after_sifting(alice_bases, sifted_mask, guess_on_erasure=guess_on_erasure, reference_states=reference_states)
