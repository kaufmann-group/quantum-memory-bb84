"""
QuantumChannel (abstract base class for noise channels)

Defines a channel N(rho) = sum_K K rho K^dag through its Kraus operators
and applies it to labeled subsystems of a QuantumRegister.

Subclasses implement kraus_operators(). A channel whose output is an
erasure qutrit also sets erasure_outcome = 2 (example see QubitErasure).

https://quantumchannelzoo.org/about

Usage:
    class Identity(QuantumChannel):
        num_targets = 1

        def kraus_operators(self):
            return [qeye(2)]

    Identity().apply(register, "B")               # one target
    Identity().apply_each(register, ["E", "F"])   # each target independently
    bb84.key_generation(signal_channel=Identity())
"""


from abc import ABC, abstractmethod

from ..utils.quantum_register import QuantumRegister

class QuantumChannel(ABC):
    """
    Quantum Channel abstract base class
    """

    num_targets = 1 # Number of subsystems the channel acts on jointly, each channel here acts on one qubit (2x2 Kraus operator). 

    @abstractmethod
    def kraus_operators(self):
        """
        Returns the Kraus operators for this channel.
        """

        pass

    def apply(self, register: QuantumRegister, targets):
        """
        Applies a quantum channel to one or more subsystems in a register.

        Arguments:
            register: the quantum register being modified
            targets: which subsystems to apply the channel to
        """

        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if len(targets) != self.num_targets:
            raise ValueError(f"Channel requires {self.num_targets} target(s)")

        # Channels whose output has an erasure level define erasure_outcome,
        # so the register flags the output as a possibly-lost photon.
        flag = getattr(self, "erasure_outcome", None)

        # This automatic flag is only supported for single-target channels.
        if flag is not None and self.num_targets != 1:
            raise ValueError("Declare multi-target erasure flags explicitly in the register")

        flags = {targets[0]: flag} if flag is not None else None
        register.apply_kraus(self.kraus_operators(), targets=targets, name=type(self).__name__, erasure_levels=flags)

        return register

    def apply_each(self, register: QuantumRegister, targets):
        """
        Applies a single-subsystem channel independently to each target.

        Arguments:
            register: the quantum register being modified
            targets: one label or a list of distinct labels
        """

        if self.num_targets != 1:
            raise ValueError("apply_each requires a single-subsystem channel")

        targets = [targets] if isinstance(targets, str) else list(targets)

        if len(set(targets)) != len(targets):
            raise ValueError("apply_each targets must be distinct")

        for target in targets:
            self.apply(register, target)

        return register # returns the register
