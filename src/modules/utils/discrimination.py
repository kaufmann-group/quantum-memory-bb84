"""Optimal (Helstrom) discrimination of two equally likely states.

Used to compare Eve's fixed probe readout with the best single-round guess
of Alice's bit that any measurement on the same (noisy) probe allows.
"""

import numpy as np
from qutip import Qobj


def helstrom_projectors(rho0, rho1, tol=1e-10):
    """Return [P_guess0, P_guess1, P_inconclusive] for equal priors.

    P_guess0 projects onto the positive part of rho0 - rho1, P_guess1 onto the
    negative part. The inconclusive projector covers the null space, where the
    two states are identical (for example an erased probe); guessing at random
    there still attains the Helstrom bound.
    """
    if rho0.dims != rho1.dims:
        raise ValueError("Both states must have the same dimensions")

    # rho0 - rho1 is Hermitian, so eigh gives real eigenvalues and orthonormal eigenvectors.
    values, vectors = np.linalg.eigh((rho0 - rho1).full())

    def projector(select):
        # Sum of |v><v| over the eigenvectors picked by the boolean mask `select`.
        block = vectors[:, select]
        return Qobj(block @ block.conj().T, dims=rho0.dims)

    return [projector(values > tol), projector(values < -tol), projector(np.abs(values) <= tol)]


def helstrom_success(rho0, rho1):
    """Best probability of identifying which of two equally likely states was sent.

    P = 1/2 + ||rho0 - rho1||_1 / 4, where the trace norm ||.||_1 is the sum of
    the absolute eigenvalues of the Hermitian difference.
    """
    eigenvalues = np.linalg.eigvalsh((rho0 - rho1).full())
    trace_norm = float(np.abs(eigenvalues).sum())
    return 0.5 + 0.25*trace_norm
