"""Noise-adaptive Helstrom readout of an existing quantum-probe attack.

This is a measurement strategy, NOT a distinct signal-interaction attack.
It wraps an attack such as FGGNP or CNOT and gives Eve the best single-round
minimum-error readout of its actual noisy probe states, conditional on the
public BB84 basis and Bob's announced detection event.

The reference states depend on the whole BB84 channel configuration. They are
computed by BB84.eve_guesses(readout="optimal"), not by the attack in isolation.
Calling the fixed readout therefore raises instead of silently claiming that
a non-adaptive measurement is noise-adaptive.
"""

from .attack import Attack
from .fggnp_attack import FGGNPAttack


class NoiseAdaptiveHelstromAttack(Attack):
    """Wrap a deterministic probe interaction and require optimized readout.

    The default interaction is FGGNPAttack. An existing CNOTAttack or another
    probe-based Attack can be provided instead, including custom probe labels.
    """

    probe_labels = ("E", "F")

    def __init__(self, interaction=None, seed=43):
        if interaction is None:
            interaction = FGGNPAttack(seed=seed)
        if not isinstance(interaction, Attack) or not interaction.probe_labels:
            raise ValueError("interaction must be a probe-based Attack instance")
        if isinstance(interaction, NoiseAdaptiveHelstromAttack):
            raise ValueError("Do not wrap another NoiseAdaptiveHelstromAttack")

        self.interaction = interaction
        self.probe_labels = self._check_labels(interaction.probe_labels,
                                               len(interaction.probe_labels))
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
        return super().measure_after_sifting(alice_bases, sifted_mask,
                                             guess_on_erasure=guess_on_erasure,
                                             reference_states=reference_states)
