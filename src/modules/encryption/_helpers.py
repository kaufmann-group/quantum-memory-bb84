"""Shared validation and BB84/E91 key conversion utilities."""


def _message_bytes(message: str | bytes) -> bytes:
    """Encode text as UTF-8; leave raw message bytes unchanged."""
    if isinstance(message, str):
        return message.encode("utf-8")
    if isinstance(message, bytes):
        return message
    raise TypeError("message must be str or bytes")


def _require_key(symmetric_key: bytes, allowed_lengths: tuple[int, ...]) -> bytes:
    """Return the key unchanged after checking its type and length in bytes."""
    if not isinstance(symmetric_key, bytes):
        raise TypeError("symmetric_key must be bytes")
    if len(symmetric_key) not in allowed_lengths:
        raise ValueError(f"symmetric_key must contain {allowed_lengths} bytes")
    return symmetric_key


def _require_packet(packet: bytes, minimum_length: int) -> bytes:
    """Return the packet unchanged after checking its type and minimum length."""
    if not isinstance(packet, bytes):
        raise TypeError("encrypted packet must be bytes")
    if len(packet) < minimum_length:
        raise ValueError("encrypted packet is too short")
    return packet


def qkd_bits_to_bytes(bits: str) -> bytes:
    """Convert a final privacy-amplified QKD bit string to bytes, with no padding.

    Example: "0100100011111111" -> bytes([0b01001000, 0b11111111]).
    """
    if (
        not isinstance(bits, str)
        or not bits                  # empty string
        or len(bits) % 8             # not a whole number of bytes
        or set(bits) - {"0", "1"}    # contains a character other than 0 or 1
    ):
        raise ValueError("bits must be a nonempty binary string with length divisible by 8")

    # Read the string 8 characters at a time and parse each chunk as a base-2 number.
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
