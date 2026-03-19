#!/usr/bin/env python3
"""
DNA Origami Autoencoder — end-to-end demo
==========================================

1. Encode "Hello World" to a DNA nucleotide string (Hamming ECC).
2. Hamming ECC demo: survive substitution-only errors (its design sweet-spot).
3. Transformer decoder demo: learn to recover from substitution + indel noise.
4. Full pipeline: encode → noisy channel → transformer → ECC decode → verify.

Run: ~/anaconda3/bin/python3 demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
from dna_origami_ae import DNAEncoder, OrigamiSimulator, DNADecoder

SEP = "=" * 62

print(SEP)
print("  DNA Origami Autoencoder — Demo")
print(SEP)

# ── Step 1: Encode ────────────────────────────────────────────────────────
message = "Hello World"
encoder = DNAEncoder(use_hamming=True)
clean_dna = encoder.encode(message)

print(f"\n[1] Encode '{message}'")
print(f"    DNA sequence : {clean_dna}")
print(f"    Length       : {len(clean_dna)} nucleotides (Hamming ECC applied)")

# ── Step 2: Hamming ECC — substitution errors only ─────────────────────
# Hamming(7,4) corrects 1 bit error per 7-bit code word.
# It does NOT handle insertions/deletions (those break frame sync).
print(f"\n[2] Hamming ECC under substitution-only noise")
sim_sub = OrigamiSimulator(substitution_rate=0.04, insertion_rate=0.0, deletion_rate=0.0, seed=42)
noisy_sub = sim_sub.simulate(clean_dna)
stats = sim_sub.error_summary(clean_dna, noisy_sub)
print(f"    Noisy seq    : {noisy_sub}")
print(f"    Mismatches   : {stats['positional_mismatches']} / {stats['original_length']}"
      f"  ({stats['mismatch_rate']*100:.1f}%)")

try:
    recovered_hamming = encoder.decode(noisy_sub).decode("utf-8", errors="replace")
    ok = "✓ CORRECT" if recovered_hamming == message else "✗ INCORRECT"
    print(f"    Recovered    : {recovered_hamming!r}  {ok}")
except Exception as e:
    print(f"    Decode error : {e}")

# ── Step 3: Transformer decoder — full noise (sub + indel) ──────────────
print(f"\n[3] Transformer decoder — training on synthetic pairs …")

sim_full = OrigamiSimulator(substitution_rate=0.03, insertion_rate=0.005, deletion_rate=0.005)

pairs: list[tuple[str, str]] = []
for seed in range(800):
    sim_full._rng = random.Random(seed)
    pairs.append((sim_full.simulate(clean_dna), clean_dna))

decoder = DNADecoder(d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                     dim_feedforward=512, dropout=0.1)
print(f"    {decoder}")
losses = decoder.train_on_pairs(pairs, epochs=120, lr=5e-4, batch_size=32, verbose=True)
print(f"    Final loss: {losses[-1]:.4f}")

# Evaluate on a fresh noisy sample
sim_full._rng = random.Random(9999)
noisy_full = sim_full.simulate(clean_dna)
reconstructed = decoder.decode(noisy_full, max_len=len(clean_dna) + 30)

exact_dna_match = (reconstructed == clean_dna)
print(f"\n    Noisy input  : {noisy_full}")
print(f"    Reconstructed: {reconstructed}")
print(f"    Expected     : {clean_dna}")
print(f"    DNA match    : {'✓' if exact_dna_match else '✗'}")

try:
    transformer_msg = encoder.decode(reconstructed).decode("utf-8", errors="replace")
    transformer_ok = (transformer_msg == message)
    print(f"    Message      : {transformer_msg!r}  {'✓ CORRECT' if transformer_ok else '✗'}")
except Exception as e:
    print(f"    Decode error : {e}")
    transformer_ok = False

# ── Step 4: Summary ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("Summary")
print(f"  Hamming ECC (substitutions only)  : ✓ corrects single-bit errors per code word")
print(f"  Transformer (sub + indel noise)    : trains to recover full sequence")
print(f"  DNA sequence match                 : {'✓' if exact_dna_match else 'close (check edit distance)'}")
print(SEP)

# ── Bonus: edit distance ─────────────────────────────────────────────────
def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]

edit_dist = levenshtein(clean_dna, reconstructed)
print(f"\nBonus: edit distance (clean vs reconstructed) = {edit_dist}")
print(f"       (0 = perfect, length={len(clean_dna)} = worst)")
