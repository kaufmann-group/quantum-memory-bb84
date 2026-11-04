"""BB84 and E91 quantum key distribution simulations with eavesdropping attacks."""

from .protocols.bb84 import BB84
from .protocols.e91 import E91

from .attacks.intercept_resend_attack import InterceptResendAttack
from .attacks.cnot_attack import CNOTAttack
from .attacks.fggnp_attack import FGGNPAttack

from .attacks.breidbart_attack import BreidbartAttack
from .attacks.partial_swap_attack import PartialSwapAttack
from .attacks.phase_covariant_cloning_attack import PhaseCovariantCloningAttack
from .attacks.noise_adaptive_helstrom_attack import NoiseAdaptiveHelstromAttack

from .utils.quantum_register import QuantumRegister, ERASURE, UNKNOWN
from .utils.probe_noise import ProbeNoise

__all__ = [
    "BB84",
    "E91",
    "InterceptResendAttack",
    "CNOTAttack",
    "FGGNPAttack",
    "BreidbartAttack",
    "PartialSwapAttack",
    "PhaseCovariantCloningAttack",
    "NoiseAdaptiveHelstromAttack",
    "QuantumRegister",
    "ERASURE",
    "UNKNOWN",
    "ProbeNoise",
]