"""Generalized dephasing in arbitrary finite dimension.

Source: https://quantumchannelzoo.org/channel/general-dephase

This implements the page's Kraus/isometry definition:
    |i> -> |i> tensor |phi_i>,
    K_a = sum_i <a|phi_i> |i><i|.
Supply d normalized QuTiP environment kets |phi_i> of a common dimension >= 2.
An optional orthonormal basis_vectors list selects the preserved system basis;
otherwise the computational basis is used. One target is one d-level subsystem.

Complex-overlap convention: these Kraus operators multiply rho_ij by
<phi_j|phi_i>. The page's displayed action reverses these inner-product
indices; here the Kraus representation and isometry determine the convention.

Example reproducing PhaseFlip(p):
    phi0 = sqrt(1-p)*basis(2, 0) + sqrt(p)*basis(2, 1)
    phi1 = sqrt(1-p)*basis(2, 0) - sqrt(p)*basis(2, 1)
    channel = GeneralizedDephasing([phi0, phi1])
Alternatively use from_correlation_matrix(C), where N(rho)_ij=C_ij*rho_ij
in the preserved basis. C must be positive semidefinite with unit diagonal.
"""

import numpy as np
from qutip import Qobj

from .channels import QuantumChannel

__all__ = ["GeneralizedDephasing"]


class GeneralizedDephasing(QuantumChannel):
    """Generalized dephasing channel.

    Keeps the populations in a chosen basis of a d-level system and multiplies
    each coherence rho_ij by the overlap <phi_j|phi_i> of environment states.

    A general dephasing channel, not restricted to one real qubit parameter.

    Source:
        https://quantumchannelzoo.org/channel/general-dephase

    Usage:
        p = 0.2
        phi0 = np.sqrt(1-p)*basis(2, 0) + np.sqrt(p)*basis(2, 1)
        phi1 = np.sqrt(1-p)*basis(2, 0) - np.sqrt(p)*basis(2, 1)
        GeneralizedDephasing([phi0, phi1]).apply(register, "E")   # same as PhaseFlip(p)

        C = np.array([[1, 0.6], [0.6, 1]])   # N(rho)_01 = 0.6 * rho_01
        bb84.key_generation(signal_channel=GeneralizedDephasing.from_correlation_matrix(C))
    """

    num_targets = 1

    def __init__(self, environment_states, basis_vectors=None):
        environment_states = list(environment_states)
        d = len(environment_states)  # system dimension: one environment state per basis vector
        if d < 2:
            raise ValueError("Supply at least two environment states (one per system basis vector)")
        for state in environment_states:
            if not isinstance(state, Qobj) or not state.isket:
                raise TypeError("Every environment state must be a QuTiP ket")

        env_dim = environment_states[0].shape[0]
        if env_dim < 2 or any(state.shape != (env_dim, 1) for state in environment_states):
            raise ValueError("Environment kets must have a common dimension >= 2")

        # Column i holds the amplitudes of |phi_i>, so row a holds <a|phi_i> for every i.
        amplitudes = np.column_stack([state.full().ravel() for state in environment_states])
        norms = np.linalg.norm(amplitudes, axis=0)
        if not np.isfinite(amplitudes).all() or not np.allclose(norms, 1, atol=1e-12, rtol=0):
            raise ValueError("Environment kets must be finite and normalized")
        self._amplitudes = amplitudes / norms  # Correct accepted floating-point roundoff.

        # The preserved system basis, as the columns of a unitary matrix.
        if basis_vectors is None:
            matrix = np.eye(d, dtype=complex)
        else:
            basis_vectors = list(basis_vectors)
            if len(basis_vectors) != d:
                raise ValueError("There must be d system basis vectors")
            if any(not isinstance(v, Qobj) or not v.isket or v.shape != (d, 1)
                   for v in basis_vectors):
                raise ValueError("Each system basis vector must be a d-dimensional QuTiP ket")
            matrix = np.column_stack([v.full().ravel() for v in basis_vectors])
            if not np.isfinite(matrix).all() or not np.allclose(
                matrix.conj().T @ matrix, np.eye(d), atol=1e-12, rtol=0
            ):
                raise ValueError("System basis vectors must be finite and orthonormal")

        self.input_dim = self.output_dim = d
        self.environment_dim = env_dim
        self._basis = Qobj(matrix, dims=[[d], [d]])

    def kraus_operators(self):
        # K_a = B diag(<a|phi_0>, ..., <a|phi_{d-1}>) B^dag, where B's columns
        # are the preserved basis vectors; there is one K_a per environment level a.
        d = self.input_dim
        return [
            self._basis * Qobj(np.diag(row), dims=[[d], [d]]) * self._basis.dag()
            for row in self._amplitudes
        ]

    @classmethod
    def from_correlation_matrix(cls, correlation_matrix, basis_vectors=None):
        """Construct from C with N(rho)_ij = C_ij*rho_ij in the chosen basis.

        C is a d-by-d Hermitian positive-semidefinite matrix with C_ii=1.
        Numerical tolerance is 1e-12 (relative to spectral scale for eigenvalues).
        """
        raw = correlation_matrix.full() if isinstance(correlation_matrix, Qobj) else correlation_matrix
        C = np.array(raw, dtype=complex, copy=True)
        if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 2:
            raise ValueError("C must be a square matrix of dimension >= 2")
        if not np.isfinite(C).all():
            raise ValueError("C must contain finite entries")
        if not np.allclose(C, C.conj().T, atol=1e-12, rtol=0):
            raise ValueError("C must be Hermitian")
        if not np.allclose(np.diag(C), 1, atol=1e-12, rtol=0):
            raise ValueError("C must have unit diagonal")

        # Symmetrize to remove roundoff, then check that no eigenvalue is negative.
        values, vectors = np.linalg.eigh((C + C.conj().T) / 2)
        tolerance = 1e-12 * max(1.0, float(np.max(np.abs(values))))
        if values.min() < -tolerance:
            raise ValueError("C must be positive semidefinite")

        # C = A A^dagger; row i of A contains the amplitudes of |phi_i>.
        A = vectors * np.sqrt(np.maximum(values, 0))[None, :]
        # C_ii = 1 already makes each row a unit vector; this only removes roundoff.
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        d = C.shape[0]
        environment_states = [Qobj(A[i, :].reshape(d, 1), dims=[[d], [1]]) for i in range(d)]
        return cls(environment_states, basis_vectors=basis_vectors)
