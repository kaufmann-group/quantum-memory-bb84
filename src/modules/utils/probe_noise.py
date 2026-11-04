"""Independent memory noise on Eve's probes, with one channel per probe label."""

from dataclasses import dataclass

from ..channels.channels import QuantumChannel


@dataclass
class ProbeNoise:
    """Apply a different single-target channel to each probe.

    Example: ProbeNoise({"E": AmplitudeDamping(0.2), "F": QubitErasure(0.25)})
    """

    channels: dict[str, QuantumChannel]  # probe label -> channel applied to that probe

    def __post_init__(self):
        # Runs right after the __init__ that @dataclass generates.
        if not self.channels:
            raise ValueError("Provide at least one probe channel.")
        if any(channel.num_targets != 1 for channel in self.channels.values()):
            raise ValueError("Each probe needs a single-target channel.")

    @property
    def num_targets(self):
        # One target per probe, so the protocol passes every probe in a single call.
        return len(self.channels)

    def apply(self, register, targets):
        targets = [targets] if isinstance(targets, str) else list(targets)

        # The targets must be exactly the probes this object has channels for.
        if len(targets) != self.num_targets or set(targets) != self.channels.keys():
            raise ValueError(f"Expected these probe targets: {list(self.channels)}")

        for target in targets:
            self.channels[target].apply(register, target)
        return register

    def apply_each(self, register, targets):
        # Protocol._apply_channel calls apply_each instead of apply when
        # num_targets == 1 (a single probe), so both must do the same thing.
        return self.apply(register, targets)
