"""AES-GCM authenticated encryption (128-, 192-, or 256-bit keys).

Packet format: 12-byte random nonce || ciphertext || 16-byte tag.
Never reuse a nonce with the same key; authentication failures raise InvalidTag.
"""

import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ._helpers import _message_bytes, _require_key, _require_packet


def encrypt_aes_gcm(symmetric_key: bytes, message: str | bytes) -> bytes:
    key = _require_key(symmetric_key, (16, 24, 32))
    cipher = AESGCM(key)

    # A fresh random 12-byte nonce for every message; it is sent in the clear.
    nonce = secrets.token_bytes(12)

    # encrypt() returns the ciphertext with the 16-byte tag appended.
    # The final None means there is no extra associated data to authenticate.
    ciphertext_and_tag = cipher.encrypt(nonce, _message_bytes(message), None)
    return nonce + ciphertext_and_tag


def decrypt_aes_gcm(symmetric_key: bytes, packet: bytes) -> bytes:
    key = _require_key(symmetric_key, (16, 24, 32))
    cipher = AESGCM(key)

    # The shortest valid packet is a 12-byte nonce plus a 16-byte tag (empty message).
    packet = _require_packet(packet, 28)
    nonce, ciphertext_and_tag = packet[:12], packet[12:]

    # Raises InvalidTag if the key is wrong or the packet was modified.
    return cipher.decrypt(nonce, ciphertext_and_tag, None)
