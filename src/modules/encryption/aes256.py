"""AES-256-GCM authenticated encryption (exactly 32-byte symmetric key)."""

from ._helpers import _require_key
from .aes_gcm import encrypt_aes_gcm, decrypt_aes_gcm


def encrypt_aes256(symmetric_key: bytes, message: str | bytes) -> bytes:
    # Enforce the exact AES-256 key size, then reuse the general AES-GCM code.
    _require_key(symmetric_key, (32,))
    return encrypt_aes_gcm(symmetric_key, message)


def decrypt_aes256(symmetric_key: bytes, packet: bytes) -> bytes:
    _require_key(symmetric_key, (32,))
    return decrypt_aes_gcm(symmetric_key, packet)
