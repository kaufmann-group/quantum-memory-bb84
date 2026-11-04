from abc import ABC, abstractmethod
from numbers import Integral

import numpy as np

from ..utils.quantum_register import QuantumRegister
from ..attacks.attack import Attack


class Protocol(ABC):
    """
    Base class for QKD protocols that share attacks, channels and probes.

    Subclasses set party_labels to the register labels their legitimate parties use; attack probes may not reuse them.
    """

    party_labels = ("B",)

    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.register = QuantumRegister() # Register shared by Alice's and Bob's qubits (and Eve's probes during an attack)
        self.attack = None  # The last `attack` argument of last `key_generation` run
        self.attacks = ()   # the same attack(s) as a tuple

    @abstractmethod
    def key_generation(self, attack=None):
        pass

    @staticmethod
    def _apply_channel(channel, register, targets):
        """
        Apply an optional channel to the targets.
        """

        if channel is None:
            return
        if channel.num_targets == 1: 
            channel.apply_each(register, targets) # Single target channel
        else:
            channel.apply(register, targets) # Multi target channel (with ProbeNoise)

    def _validate_records(self, record_rounds, count):
        """
        Return the set of round indices to record, checking each is in range.
        """
        records = set(record_rounds)
        if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < count for i in records):
            raise ValueError("record_rounds must contain valid zero-based round indices")
        return records

    def _as_attacks(self, attack):
        """
        Prepares the `attack` argument for a protocol run. 
        """
        
        if attack is None:
            return ()
        attacks = tuple(attack) if isinstance(attack, (list, tuple)) else (attack,)

        # Removed this, code below just checks for `Attack` object.
        #for a in attacks:
        #    if not callable(a) or not all(callable(getattr(a, m, None)) for m in ("apply", "reset", "capture")):
        #        raise TypeError("Each attack must be callable and provide apply, reset and capture (subclass Attack)")
        #if len({id(a) for a in attacks}) != len(attacks):
        #    raise ValueError("Each attack instance may appear only once per run")

        for a in attacks:
            if not isinstance(a, Attack):
                raise TypeError("Each attack must be an instance of an `Attack` subclass")
            
        if len({id(a) for a in attacks}) != len(attacks):
            raise ValueError("Each attack instance may appear only once per run")

        # Probe labels must not clash with each other or with Alice's and Bob's labels.
        labels = [label for a in attacks for label in getattr(a, "probe_labels", ())]
        if len(set(labels)) != len(labels) or set(labels) & set(self.party_labels):
            raise ValueError(f"Probe labels must be unique across attacks and differ from {list(self.party_labels)} (got {labels}); e.g. CNOTAttack(probe_label='E2')")

        # Equal seeds give identical random streams, so chained Eves would make identical choices.
        seeds = [getattr(a, "seed", None) for a in attacks]
        keys = []
        for s in seeds:
            if s is None or isinstance(s, np.random.Generator): # Generator seeds are skipped.
                continue
            keys.append(repr(int(s)) if isinstance(s, Integral) else repr(s))
        if len(set(keys)) != len(keys):
            raise ValueError("Chained attacks need distinct seeds.")
        return attacks

    @staticmethod
    def _memory_targets(attacks, memory_channel, memory_targets):
        """
        Resolve and validate which probes the memory channel acts on.
        """
        probes = tuple(label for a in attacks for label in getattr(a, "probe_labels", ()))
        if memory_targets is not None and memory_channel is None:
            raise ValueError("memory_targets was given without a memory_channel")

        # Default to every probe; accept one label or a list of labels.
        if memory_targets is None:
            targets = list(probes)
        elif isinstance(memory_targets, str):
            targets = [memory_targets]
        else:
            targets = list(memory_targets)

        if memory_channel is not None and (not targets or any(t not in probes for t in targets)):
            raise ValueError(f"Memory noise requires targets belonging to the attack's quantum probes {list(probes)}")
        return targets

    def _select_attack(self, attack):
        """Choose one attack of the last run by index or instance; None means the only one."""
        # By list index (negative indices count from the end, as for lists).
        if isinstance(attack, Integral) and not isinstance(attack, bool):
            if not -len(self.attacks) <= attack < len(self.attacks):
                raise ValueError(f"Attack index {attack} is out of range; the last run had "
                                 f"{len(self.attacks)} attack(s)")
            return self.attacks[attack]

        # None is only unambiguous when the run had exactly one attack.
        if attack is None:
            if len(self.attacks) != 1:
                if self.attacks:
                    raise ValueError("Pass attack= (an instance or index) to choose one of several attacks")
                raise ValueError("This run had no attack")
            return self.attacks[0]

        # By instance (the same object, not just an equal one).
        if not any(attack is a for a in self.attacks):
            raise ValueError("That attack did not take part in the last run")
        return attack
