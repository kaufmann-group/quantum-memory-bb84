"""AES-CTR encryption (128-, 192-, or 256-bit keys).

Packet format: 16-byte random IV || ciphertext.
WARNING: CTR does NOT authenticate ciphertext or detect tampering.
Prefer AES-GCM or ChaCha20-Poly1305 for real communications.
Never reuse an IV with the same key.
"""

import secrets

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from ._helpers import _message_bytes, _require_key, _require_packet


def encrypt_aes_ctr(symmetric_key: bytes, message: str | bytes) -> bytes:
    key = _require_key(symmetric_key, (16, 24, 32))

    # A fresh random 16-byte IV (the starting counter block); it is sent in the clear.
    iv = secrets.token_bytes(16)
    encryptor = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor()

    # CTR is a stream mode, so the ciphertext is exactly as long as the message.
    ciphertext = encryptor.update(_message_bytes(message)) + encryptor.finalize()
    return iv + ciphertext


def decrypt_aes_ctr(symmetric_key: bytes, packet: bytes) -> bytes:
    key = _require_key(symmetric_key, (16, 24, 32))

    # A packet must contain at least the 16-byte IV (the message may be empty).
    packet = _require_packet(packet, 16)
    iv, ciphertext = packet[:16], packet[16:]

    decryptor = Cipher(algorithms.AES(key), modes.CTR(iv)).decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()
