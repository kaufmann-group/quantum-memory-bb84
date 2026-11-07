"""
Breidbart attack.

Intercept-resend attack in which Eve always measures in the Breidbart
basis, halfway between the BB84 Z and X bases, and resends what she found.
This maximizes her chance of guessing each bit, with probability 
cos(pi/8)**2 ≈ 0.854 with Bob's sifted QBER is 1/4.

https://doi.org/10.1007/BF00191318

Usage:
    bb84 = BB84(seed=1, number_bits=1000)
    bb84.key_generation(BreidbartAttack(seed=43))
    bb84.qber()                          # about 0.25
    guesses = bb84.eve_guesses()
    (guesses == bb84.alice_key).mean()   # about 0.85
"""

import secrets

import numpy as np

from .intercept_resend_attack import InterceptResendAttack

class BreidbartAttack(InterceptResendAttack):
    """
    Breidbart attack.
    """

    def __init__(self, seed=secrets.randbits(128)):
        super().__init__(seed=seed, bases=(np.pi / 4,)) # Measurement uses the angle of the observable cos(a)*Z + sin(a)*x
