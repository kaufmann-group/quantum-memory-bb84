"""Breidbart intercept-resend attack: measure halfway between the BB84 bases.

The register's angle specifies a Bloch-sphere direction, not the angle in
|b0> = cos(pi/8)|0> + sin(pi/8)|1>. Therefore the Breidbart measurement uses
pi/4 radians in QuantumRegister.measure(). The resulting projectors select
|b0> and its orthogonal complement.

This is an intercept-resend attack, not an attack with stored quantum probes.
"""

import numpy as np

from .intercept_resend_attack import InterceptResendAttack


class BreidbartAttack(InterceptResendAttack):
    """Eve intercepts BB84 signals and measures at 45 degrees between X and Z.

    The recorded outcome is her bit guess after sifting. On ideal BB84 signals,
    Eve guesses correctly with probability cos(pi/8)**2; Bob's sifted QBER
    is 1/4. Measurement outcomes and resends use the parent attack's machinery.
    """

    def __init__(self, seed=43):
        # Measurement uses the angle of the observable cos(a)*Z + sin(a)*X.
        # Its +1 eigenstate is cos(a/2)|0> + sin(a/2)|1>.
        super().__init__(seed=seed, bases=(np.pi / 4,))
