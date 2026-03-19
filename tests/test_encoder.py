"""Tests for DNAEncoder."""

import pytest
from dna_origami_ae.encoder import (
    DNAEncoder,
    _hamming_encode_nibble,
    _hamming_decode_word,
)


# ── Hamming helpers ───────────────────────────────────────────────────────

class TestHammingHelpers:
    def test_encode_decode_all_nibbles(self):
        """Round-trip for all 16 possible nibbles with no errors."""
        for nibble in range(16):
            word = _hamming_encode_nibble(nibble)
            assert len(word) == 7
            recovered = _hamming_decode_word(word)
            assert recovered == nibble, f"Failed for nibble={nibble}"

    def test_single_bit_error_correction(self):
        """Hamming(7,4) should correct any single-bit flip."""
        for nibble in range(16):
            word = _hamming_encode_nibble(nibble)
            for bit_pos in range(7):
                corrupted = list(word)
                corrupted[bit_pos] ^= 1  # flip one bit
                recovered = _hamming_decode_word(list(corrupted))
                assert recovered == nibble, (
                    f"Failed to correct bit {bit_pos} for nibble={nibble}"
                )


# ── DNAEncoder ────────────────────────────────────────────────────────────

class TestDNAEncoder:
    def test_encode_returns_valid_bases(self):
        enc = DNAEncoder()
        seq = enc.encode("Hi")
        assert all(b in "ATGC" for b in seq), f"Invalid base in: {seq}"

    def test_primer_prepended(self):
        enc = DNAEncoder()
        seq = enc.encode("test")
        assert seq.startswith(enc.primer)

    def test_roundtrip_simple(self):
        enc = DNAEncoder()
        for msg in ["", "A", "Hello", "Hello World", "DNA storage rocks!"]:
            encoded = enc.encode(msg)
            decoded = enc.decode(encoded).decode("utf-8")
            assert decoded == msg, f"Round-trip failed for {msg!r}"

    def test_roundtrip_bytes(self):
        enc = DNAEncoder()
        data = bytes(range(256))
        assert enc.decode(enc.encode(data)) == data

    def test_roundtrip_no_hamming(self):
        enc = DNAEncoder(use_hamming=False)
        msg = "plain encoding"
        assert enc.decode(enc.encode(msg)).decode("utf-8") == msg

    def test_missing_primer_raises(self):
        enc = DNAEncoder()
        with pytest.raises(ValueError, match="Primer"):
            enc.decode("AAAAAAAAAAA")

    def test_invalid_base_raises(self):
        enc = DNAEncoder()
        with pytest.raises(ValueError, match="Invalid nucleotide"):
            enc.decode("ATCGATCGXXXX")  # X is not a valid base

    def test_custom_primer(self):
        enc = DNAEncoder(primer="GCGCGCGC")
        seq = enc.encode("custom primer")
        assert seq.startswith("GCGCGCGC")
        assert enc.decode(seq).decode("utf-8") == "custom primer"

    def test_encoding_ratio_hamming(self):
        enc = DNAEncoder(use_hamming=True)
        assert enc.encoding_ratio() == 7 / 4

    def test_encoding_ratio_no_hamming(self):
        enc = DNAEncoder(use_hamming=False)
        assert enc.encoding_ratio() == 1.0
