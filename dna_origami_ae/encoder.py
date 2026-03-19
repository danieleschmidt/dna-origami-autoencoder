"""
DNAEncoder: convert arbitrary binary data to DNA nucleotide sequences.

Encoding scheme
---------------
1. The input bytes are expanded with Hamming(7,4) error-correction so that
   any single-bit error per 7-bit code-word can be corrected later.
2. The resulting bit-stream is mapped to nucleotides:
       00 → A
       01 → T
       10 → G
       11 → C
   Each nucleotide therefore carries 2 bits (base-4 encoding).
3. A fixed 8-nucleotide primer is prepended so that the decoder can find
   the start of the payload and verify synchronisation.

Decoding is the exact inverse: strip primer → map nucleotides → run Hamming
decoder → reassemble original bytes.

References
----------
- Church et al. Science 337, 1628–1630 (2012)
- Goldman et al. Nature 494, 77–80 (2013)
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

# ── Nucleotide ↔ bit-pair maps ─────────────────────────────────────────────

_BASE_TO_BITS: dict[str, str] = {"A": "00", "T": "01", "G": "10", "C": "11"}
_BITS_TO_BASE: dict[str, str] = {v: k for k, v in _BASE_TO_BITS.items()}

# Start primer – identifies the payload (8 bases = 16 bits, always the same)
_PRIMER: str = "ATCGATCG"

# ── Hamming(7,4) helpers ──────────────────────────────────────────────────

def _hamming_encode_nibble(nibble: int) -> list[int]:
    """Encode a 4-bit nibble as a 7-bit Hamming code-word.

    Bit positions are 1-indexed (Hamming convention):
        p1 p2 d1 p3 d2 d3 d4
    where pN are parity bits and dN are data bits.
    """
    d = [(nibble >> i) & 1 for i in range(3, -1, -1)]  # d1 d2 d3 d4
    p1 = d[0] ^ d[1] ^ d[3]
    p2 = d[0] ^ d[2] ^ d[3]
    p3 = d[1] ^ d[2] ^ d[3]
    return [p1, p2, d[0], p3, d[1], d[2], d[3]]


def _hamming_decode_word(word: list[int]) -> int | None:
    """Decode a 7-bit Hamming code-word, correcting single-bit errors.

    Returns the 4-bit nibble, or None if an uncorrectable error is detected.
    """
    if len(word) != 7:
        return None
    p1, p2, d1, p3, d2, d3, d4 = word
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = s1 * 1 + s2 * 2 + s3 * 4  # error bit position (1-indexed)
    if syndrome != 0:
        pos = syndrome - 1          # 0-indexed
        if pos < 7:
            word = list(word)
            word[pos] ^= 1         # flip the erroneous bit
    _, _, d1, _, d2, d3, d4 = word
    return (d1 << 3) | (d2 << 2) | (d3 << 1) | d4


# ── Main class ────────────────────────────────────────────────────────────

class DNAEncoder:
    """Encode/decode arbitrary bytes ↔ DNA nucleotide sequences.

    Parameters
    ----------
    use_hamming : bool
        Whether to apply Hamming(7,4) error-correction coding (default True).
        When False a plain base-4 encoding is used.
    primer : str
        8-nucleotide start primer prepended to every encoded sequence.
    """

    def __init__(self, use_hamming: bool = True, primer: str = _PRIMER) -> None:
        self.use_hamming = use_hamming
        self.primer = primer

    # ── Encoding ────────────────────────────────────────────────────────

    def encode(self, data: bytes | str) -> str:
        """Encode *data* to a DNA nucleotide string.

        Parameters
        ----------
        data : bytes or str
            The payload to encode.  Strings are UTF-8 encoded.

        Returns
        -------
        str
            DNA sequence (A/T/G/C characters only).
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        bits = self._bytes_to_bits(data)

        if self.use_hamming:
            bits = self._apply_hamming_encoding(bits)

        nucleotides = self._bits_to_nucleotides(bits)
        return self.primer + nucleotides

    def decode(self, sequence: str) -> bytes:
        """Decode a DNA nucleotide string back to the original bytes.

        Parameters
        ----------
        sequence : str
            A DNA sequence previously produced by :meth:`encode`.

        Returns
        -------
        bytes
            Reconstructed payload.

        Raises
        ------
        ValueError
            If the primer is missing or the sequence is malformed.
        """
        sequence = sequence.upper().strip()

        if not sequence.startswith(self.primer):
            raise ValueError(
                f"Primer '{self.primer}' not found at start of sequence "
                f"(got '{sequence[:len(self.primer)]}')"
            )

        payload_seq = sequence[len(self.primer):]
        bits = self._nucleotides_to_bits(payload_seq)

        if self.use_hamming:
            bits = self._apply_hamming_decoding(bits)

        return self._bits_to_bytes(bits)

    # ── Bit / nucleotide conversions ─────────────────────────────────────

    @staticmethod
    def _bytes_to_bits(data: bytes) -> list[int]:
        bits: list[int] = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def _bits_to_bytes(bits: list[int]) -> bytes:
        # Trim padding zeros from the right if needed
        # We expect the number of bits to be a multiple of 8 after decoding
        n = (len(bits) // 8) * 8
        result = bytearray()
        for i in range(0, n, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)
        return bytes(result)

    @staticmethod
    def _bits_to_nucleotides(bits: list[int]) -> str:
        # Pad to even length
        if len(bits) % 2:
            bits = bits + [0]
        nucleotides = []
        for i in range(0, len(bits), 2):
            pair = str(bits[i]) + str(bits[i + 1])
            nucleotides.append(_BITS_TO_BASE[pair])
        return "".join(nucleotides)

    @staticmethod
    def _nucleotides_to_bits(seq: str) -> list[int]:
        bits: list[int] = []
        for base in seq:
            if base not in _BASE_TO_BITS:
                raise ValueError(f"Invalid nucleotide: '{base}'")
            bits.extend(int(b) for b in _BASE_TO_BITS[base])
        return bits

    # ── Hamming ECC ───────────────────────────────────────────────────────

    @staticmethod
    def _apply_hamming_encoding(bits: list[int]) -> list[int]:
        """Encode a flat bit list using Hamming(7,4), nibble by nibble."""
        # Pad to a multiple of 4
        remainder = len(bits) % 4
        if remainder:
            bits = bits + [0] * (4 - remainder)

        encoded: list[int] = []
        for i in range(0, len(bits), 4):
            nibble = (bits[i] << 3) | (bits[i + 1] << 2) | (bits[i + 2] << 1) | bits[i + 3]
            encoded.extend(_hamming_encode_nibble(nibble))
        return encoded

    @staticmethod
    def _apply_hamming_decoding(bits: list[int]) -> list[int]:
        """Decode a Hamming(7,4)-encoded bit list, correcting single-bit errors."""
        # Trim to a multiple of 7
        trim = len(bits) - (len(bits) % 7)
        bits = bits[:trim]

        decoded: list[int] = []
        for i in range(0, len(bits), 7):
            word = bits[i:i + 7]
            nibble = _hamming_decode_word(word)
            if nibble is None:
                nibble = 0  # uncorrectable – substitute 0
            decoded.extend([(nibble >> j) & 1 for j in range(3, -1, -1)])
        return decoded

    # ── Utility ──────────────────────────────────────────────────────────

    def encoding_ratio(self) -> float:
        """Overhead factor relative to plain base-4 encoding.

        Base-4: 1 nucleotide per 2 bits → 4 bits/base (wait, ×0.25 bytes/base)
        Hamming(7,4): 7 nucleotides per 4 data bits → 3.5× more nucleotides
        """
        return 7 / 4 if self.use_hamming else 1.0

    def __repr__(self) -> str:
        return (
            f"DNAEncoder(use_hamming={self.use_hamming}, "
            f"primer='{self.primer}')"
        )
