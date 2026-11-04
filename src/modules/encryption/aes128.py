"""AES-128-GCM authenticated encryption (exactly 16-byte symmetric key)."""

from ._helpers import _require_key
from .aes_gcm import encrypt_aes_gcm, decrypt_aes_gcm


def encrypt_aes128(symmetric_key: bytes, message: str | bytes) -> bytes:
    # Enforce the exact AES-128 key size, then reuse the general AES-GCM code.
    _require_key(symmetric_key, (16,))
    return encrypt_aes_gcm(symmetric_key, message)


def decrypt_aes128(symmetric_key: bytes, packet: bytes) -> bytes:
    _require_key(symmetric_key, (16,))
    return decrypt_aes_gcm(symmetric_key, packet)
