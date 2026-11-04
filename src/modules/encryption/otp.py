"""
One-Time Pad encryption/decryption.

The key must be exactly as long as the message and must never be reused.
"""

from ._helpers import _message_bytes, _require_key


def encrypt_otp(symmetric_key: bytes, message: str | bytes) -> bytes:
    plaintext = _message_bytes(message)
    key = _require_key(symmetric_key, (len(plaintext),))

    # XOR each message byte with the matching key byte.
    return bytes(m ^ k for m, k in zip(plaintext, key))


def decrypt_otp(symmetric_key: bytes, ciphertext: bytes) -> bytes:
    # XOR is its own inverse, (m ^ k) ^ k == m, so decryption is the same operation.
    return encrypt_otp(symmetric_key, ciphertext)
