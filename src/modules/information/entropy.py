import numpy as np

def shannon_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def von_neumann_entropy(rho):
    rho = (rho + rho.conj().T) / 2
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.clip(eigenvalues, 0, None)
    eigenvalues /= eigenvalues.sum()
    return shannon_entropy(eigenvalues)

def holevo_binary(rho0, rho1):
    avg = 0.5 * (rho0 + rho1)
    return (von_neumann_entropy(avg)
            - 0.5 * von_neumann_entropy(rho0)
            - 0.5 * von_neumann_entropy(rho1))
