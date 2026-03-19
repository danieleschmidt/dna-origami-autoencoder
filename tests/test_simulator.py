"""Tests for OrigamiSimulator."""

import pytest
from dna_origami_ae.simulator import OrigamiSimulator


class TestOrigamiSimulator:
    def test_zero_errors_returns_identical(self):
        sim = OrigamiSimulator(substitution_rate=0.0, insertion_rate=0.0, deletion_rate=0.0)
        seq = "ATGCATGCATGC"
        assert sim.simulate(seq) == seq

    def test_all_bases_valid(self):
        sim = OrigamiSimulator(substitution_rate=0.3, insertion_rate=0.1, deletion_rate=0.1, seed=1)
        seq = "AAAAAAAAAAAAAAAA"
        result = sim.simulate(seq)
        assert all(b in "ATGC" for b in result), f"Invalid base in: {result}"

    def test_substitution_only(self):
        """With only substitutions the length must stay the same."""
        sim = OrigamiSimulator(substitution_rate=0.5, insertion_rate=0.0, deletion_rate=0.0, seed=99)
        seq = "ATGCATGCATGCATGC"
        result = sim.simulate(seq)
        assert len(result) == len(seq)

    def test_reproducible_with_seed(self):
        seq = "AAATTTGGGCCC"
        sim1 = OrigamiSimulator(substitution_rate=0.1, seed=7)
        sim2 = OrigamiSimulator(substitution_rate=0.1, seed=7)
        assert sim1.simulate(seq) == sim2.simulate(seq)

    def test_different_seeds_differ(self):
        seq = "AAATTTGGGCCC" * 5
        sim1 = OrigamiSimulator(substitution_rate=0.3, seed=1)
        sim2 = OrigamiSimulator(substitution_rate=0.3, seed=2)
        # Very unlikely to be equal with 30% substitution rate
        assert sim1.simulate(seq) != sim2.simulate(seq)

    def test_batch_simulation(self):
        sim = OrigamiSimulator(substitution_rate=0.05, seed=42)
        seqs = ["ATGCATGC", "CCCCCCCC", "TTTTTTTT"]
        results = sim.simulate_batch(seqs)
        assert len(results) == 3
        for r in results:
            assert all(b in "ATGC" for b in r)

    def test_error_summary_keys(self):
        sim = OrigamiSimulator(seed=0)
        summary = sim.error_summary("ATGC", "ATGCATGC")
        assert "original_length" in summary
        assert "degraded_length" in summary
        assert "length_delta" in summary
        assert "mismatch_rate" in summary

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError):
            OrigamiSimulator(substitution_rate=1.5)
        with pytest.raises(ValueError):
            OrigamiSimulator(insertion_rate=-0.1)
        with pytest.raises(ValueError):
            OrigamiSimulator(deletion_rate=2.0)
