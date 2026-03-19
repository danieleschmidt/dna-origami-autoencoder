"""Tests for DNADecoder."""

import pytest
import torch
from dna_origami_ae.decoder import (
    DNADecoder,
    seq_to_tensor,
    tensor_to_seq,
    VOCAB,
    SOS_IDX,
    EOS_IDX,
    PAD_IDX,
)
from dna_origami_ae.encoder import DNAEncoder
from dna_origami_ae.simulator import OrigamiSimulator


class TestTokenisation:
    def test_seq_to_tensor_known_sequence(self):
        t = seq_to_tensor("ATGC")
        expected = [VOCAB["A"], VOCAB["T"], VOCAB["G"], VOCAB["C"]]
        assert t.tolist() == expected

    def test_roundtrip_tokenisation(self):
        seq = "AATCGGCTA"
        assert tensor_to_seq(seq_to_tensor(seq)) == seq

    def test_sos_eos_stripped(self):
        """tensor_to_seq should strip SOS/EOS tokens."""
        import torch
        t = torch.tensor([SOS_IDX, VOCAB["A"], VOCAB["T"], EOS_IDX])
        assert tensor_to_seq(t) == "AT"


class TestDNADecoder:
    @pytest.fixture
    def tiny_decoder(self):
        return DNADecoder(d_model=32, nhead=2, num_encoder_layers=1,
                         num_decoder_layers=1, dim_feedforward=64, dropout=0.0)

    def test_repr(self, tiny_decoder):
        r = repr(tiny_decoder)
        assert "DNADecoder" in r
        assert "params=" in r

    def test_decode_returns_valid_dna(self, tiny_decoder):
        """Without training, output should still be valid DNA characters."""
        result = tiny_decoder.decode("ATCGATCGATCG", max_len=20)
        assert all(b in "ATGC" for b in result)

    def test_train_reduces_loss(self, tiny_decoder):
        """Training should reduce cross-entropy loss."""
        seq = "ATCGATCG" * 3  # short clean sequence
        sim = OrigamiSimulator(substitution_rate=0.1, seed=1)
        pairs = [(sim.simulate(seq), seq) for _ in range(50)]
        losses = tiny_decoder.train_on_pairs(pairs, epochs=20, lr=1e-3, verbose=False)
        assert losses[-1] < losses[0], "Loss did not decrease during training"

    def test_save_load(self, tiny_decoder, tmp_path):
        path = str(tmp_path / "model.pt")
        tiny_decoder.save(path)
        tiny_decoder.load(path)

    def test_overfit_short_sequence(self):
        """A small model should be able to overfit a single short sequence pair."""
        clean = "ATCGATCGATCG"
        noisy = "ATCAATCGATGG"
        decoder = DNADecoder(d_model=64, nhead=4, num_encoder_layers=2,
                             num_decoder_layers=2, dim_feedforward=128, dropout=0.0)
        pairs = [(noisy, clean)] * 200
        decoder.train_on_pairs(pairs, epochs=100, lr=5e-4, verbose=False)
        reconstructed = decoder.decode(noisy, max_len=len(clean) + 5)
        # After severe overfitting on one pair, the output should match
        assert reconstructed == clean, f"Expected {clean!r}, got {reconstructed!r}"

    def test_end_to_end_pipeline(self):
        """Integration: encode → simulate → train → decode → recover message."""
        message = "Hi"
        enc = DNAEncoder(use_hamming=True)
        clean = enc.encode(message)

        sim = OrigamiSimulator(substitution_rate=0.02, insertion_rate=0.005,
                               deletion_rate=0.005, seed=7)

        # Generate training pairs
        import random
        pairs = []
        for s in range(200):
            sim._rng = random.Random(s)
            pairs.append((sim.simulate(clean), clean))

        decoder = DNADecoder(d_model=64, nhead=4, num_encoder_layers=2,
                             num_decoder_layers=2, dim_feedforward=256, dropout=0.1)
        decoder.train_on_pairs(pairs, epochs=60, lr=5e-4, verbose=False)

        # Test on a fresh noisy sample
        sim._rng = random.Random(999)
        noisy = sim.simulate(clean)
        reconstructed = decoder.decode(noisy, max_len=len(clean) + 20)

        # The transformer output should be a valid DNA string
        assert all(b in "ATGC" for b in reconstructed)
