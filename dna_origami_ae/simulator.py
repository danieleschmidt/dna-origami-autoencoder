"""
OrigamiSimulator: model the noise introduced by DNA synthesis and sequencing.

Real DNA storage systems must contend with three primary error types:

Substitution  (base replacement):  A → G, T → C, etc.
    Sources: polymerase errors, chemical damage, sequencing miscalls.
    Typical rate: ~0.1–1% per base in Oxford Nanopore, lower for Illumina.

Insertion     (extra base added):
    Sources: homopolymer slippage, DNA synthesis overstep.
    Typical rate: ~0.5–2% for long-read sequencers.

Deletion      (base lost):
    Sources: polymerase skip, exonuclease activity, homopolymer compression.
    Typical rate: ~1–3% for long-read sequencers.

This simulator applies these errors independently per base with configurable
per-type rates, producing realistic degraded sequences for downstream decoding.

References
----------
- Organick et al. Nat. Biotechnol. 36, 242–248 (2018)
- Ceze et al. Nat. Rev. Genet. 20, 456–466 (2019)
"""

from __future__ import annotations

import random
from typing import Optional

_BASES = ["A", "T", "G", "C"]


class OrigamiSimulator:
    """Introduce realistic synthesis / sequencing errors into DNA sequences.

    Parameters
    ----------
    substitution_rate : float
        Probability that each base is replaced by a random different base.
    insertion_rate : float
        Probability that a random extra base is inserted *before* each base.
    deletion_rate : float
        Probability that each base is deleted (dropped entirely).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        substitution_rate: float = 0.01,
        insertion_rate: float = 0.005,
        deletion_rate: float = 0.005,
        seed: Optional[int] = None,
    ) -> None:
        if not (0.0 <= substitution_rate <= 1.0):
            raise ValueError(f"substitution_rate must be in [0,1], got {substitution_rate}")
        if not (0.0 <= insertion_rate <= 1.0):
            raise ValueError(f"insertion_rate must be in [0,1], got {insertion_rate}")
        if not (0.0 <= deletion_rate <= 1.0):
            raise ValueError(f"deletion_rate must be in [0,1], got {deletion_rate}")

        self.substitution_rate = substitution_rate
        self.insertion_rate = insertion_rate
        self.deletion_rate = deletion_rate
        self._rng = random.Random(seed)

    def simulate(self, sequence: str) -> str:
        """Apply errors to *sequence* and return the degraded sequence.

        Parameters
        ----------
        sequence : str
            Clean DNA sequence (A/T/G/C).

        Returns
        -------
        str
            Degraded DNA sequence with substitutions, insertions, and deletions.
        """
        result: list[str] = []
        for base in sequence.upper():
            # Insertion: add a random base before this position
            if self._rng.random() < self.insertion_rate:
                result.append(self._rng.choice(_BASES))

            # Deletion: skip this base
            if self._rng.random() < self.deletion_rate:
                continue

            # Substitution: replace with a different base
            if self._rng.random() < self.substitution_rate:
                result.append(self._rng.choice([b for b in _BASES if b != base]))
            else:
                result.append(base)

        return "".join(result)

    def simulate_batch(self, sequences: list[str]) -> list[str]:
        """Apply errors independently to each sequence in a list."""
        return [self.simulate(s) for s in sequences]

    def error_summary(self, original: str, degraded: str) -> dict:
        """Compute simple statistics comparing original vs degraded sequence.

        Note: because insertions and deletions change the length this is a
        rough character-level count, not a full alignment.
        """
        n_orig = len(original)
        n_deg = len(degraded)
        n_compare = min(n_orig, n_deg)
        mismatches = sum(
            1 for a, b in zip(original[:n_compare], degraded[:n_compare]) if a != b
        )
        return {
            "original_length": n_orig,
            "degraded_length": n_deg,
            "length_delta": n_deg - n_orig,
            "positional_mismatches": mismatches,
            "mismatch_rate": mismatches / n_compare if n_compare else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"OrigamiSimulator("
            f"sub={self.substitution_rate:.3f}, "
            f"ins={self.insertion_rate:.3f}, "
            f"del={self.deletion_rate:.3f})"
        )
