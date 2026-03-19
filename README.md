# DNA Origami Autoencoder

> **Encode digital data as DNA nucleotide sequences, simulate synthesis and sequencing noise, and decode back using a transformer.**

---

## What Is DNA Data Storage?

DNA is nature's hard drive. A single gram of DNA can theoretically store ~215 petabytes of data — orders of magnitude denser than any silicon medium. Unlike magnetic or flash storage, DNA is stable for thousands of years under the right conditions.

Real-world DNA storage systems are being built today:
- **Church et al. (2012, *Science*)** — first practical DNA data storage (650 kB in synthetic DNA)
- **Goldman et al. (2013, *Nature*)** — 739 kB stored in synthetic oligonucleotides
- **Organick et al. (2018, *Nature Biotechnology*)** — random access across 200 MB stored in a DNA pool
- **Microsoft & Twist Bioscience** — fully automated DNA read/write pipeline
- **Catalog** — world record 16 GB stored in DNA (2021)

The challenge: DNA synthesis and sequencing introduce errors. This project implements the core encoding/decoding pipeline and trains a transformer to denoise corrupted sequences.

---

## Architecture

```
 digital data (bytes)
        │
        ▼
 ┌─────────────┐
 │ DNAEncoder  │  base-4 encoding + Hamming(7,4) ECC
 └─────────────┘
        │
        ▼
 DNA sequence (ATGC…)
        │
        ▼  (real-world: synthesis + storage + sequencing)
 ┌──────────────────┐
 │ OrigamiSimulator │  substitution / insertion / deletion errors
 └──────────────────┘
        │
        ▼
 noisy DNA sequence
        │
        ▼
 ┌───────────────┐
 │  DNADecoder   │  encoder–decoder transformer (denoising)
 └───────────────┘
        │
        ▼
 clean DNA sequence  →  DNAEncoder.decode()  →  original bytes
```

### `DNAEncoder`

Converts arbitrary bytes to a nucleotide string.

**Encoding steps:**
1. Convert bytes to a bit-stream.
2. Apply **Hamming(7,4)** error-correcting code: every 4 data bits → 7 bits (can correct 1-bit errors per code word).
3. Map pairs of bits to nucleotides: `00→A`, `01→T`, `10→G`, `11→C`.
4. Prepend an 8-nucleotide start primer (`ATCGATCG`) for synchronisation.

```python
from dna_origami_ae import DNAEncoder

enc = DNAEncoder(use_hamming=True)
dna = enc.encode("Hello World")
# "ATCGATCGGTGTCAACACAGTTCACACCACACACCACACTCCCTTTAAA..."

original = enc.decode(dna).decode("utf-8")
# "Hello World"
```

### `OrigamiSimulator`

Models the three primary error types in real DNA storage pipelines:

| Error type     | Physical cause                             | Typical rate  |
|----------------|--------------------------------------------|---------------|
| Substitution   | Polymerase errors, sequencing miscalls     | 0.1–1%        |
| Insertion      | Homopolymer slippage, synthesis overstep   | 0.5–2%        |
| Deletion       | Polymerase skip, homopolymer compression   | 1–3%          |

```python
from dna_origami_ae import OrigamiSimulator

sim = OrigamiSimulator(
    substitution_rate=0.02,
    insertion_rate=0.005,
    deletion_rate=0.005,
    seed=42
)
noisy = sim.simulate(dna)
```

### `DNADecoder`

A standard **encoder–decoder transformer** trained as a denoising autoencoder. The input is the noisy nucleotide sequence; the output is the reconstructed clean sequence.

- Vocabulary: `{A, T, G, C, PAD, SOS, EOS}` (7 tokens)
- Architecture: configurable depth (default: 2 encoder layers, 2 decoder layers, d_model=128, 4 attention heads)
- Training: teacher-forced cross-entropy on (noisy, clean) sequence pairs
- Inference: greedy decoding

```python
from dna_origami_ae import DNADecoder, OrigamiSimulator
import random

# Generate training pairs
sim = OrigamiSimulator(substitution_rate=0.03, insertion_rate=0.005, deletion_rate=0.005)
pairs = []
for seed in range(800):
    sim._rng = random.Random(seed)
    pairs.append((sim.simulate(dna), dna))

# Train
decoder = DNADecoder(d_model=128, nhead=4)
decoder.train_on_pairs(pairs, epochs=120, lr=5e-4)

# Decode
recovered_dna = decoder.decode(noisy)
```

---

## Quick Start

```bash
git clone https://github.com/danieleschmidt/dna-origami-autoencoder
cd dna-origami-autoencoder
pip install torch numpy
python demo.py
```

### Expected output

```
[1] Encode 'Hello World'
    DNA sequence : ATCGATCGGTGTCAACACAGTTCACACCACACACCACACTCCCTTTAAA...
    Length       : 85 nucleotides

[2] Hamming ECC under substitution-only noise
    Mismatches   : 0 / 85  (0.0%)
    Recovered    : 'Hello World'  ✓ CORRECT

[3] Transformer decoder — training on synthetic pairs …
    Final loss: 0.1066
    Reconstructed: ATCGATCGGTGTCAACACAGTTCACACCACACACCACACT...
    DNA match    : ✗  (edit distance = 2 from clean)
    Message      : 'Hello Wor█d'

Bonus: edit distance (clean vs reconstructed) = 2
```

---

## Installation

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

```bash
pip install -e ".[dev]"
```

---

## Running Tests

```bash
pytest
```

29 tests covering Hamming(7,4) encoding, round-trip correctness, error simulation, and transformer training/inference.

---

## Project Structure

```
dna_origami_ae/
├── __init__.py       # Public API: DNAEncoder, OrigamiSimulator, DNADecoder
├── encoder.py        # Binary ↔ DNA conversion with Hamming ECC
├── simulator.py      # Substitution / insertion / deletion noise model
└── decoder.py        # Transformer-based sequence denoiser

tests/
├── test_encoder.py
├── test_simulator.py
└── test_decoder.py

demo.py               # End-to-end "Hello World" demonstration
```

---

## Research Context

This project is directly inspired by active research in molecular data storage:

- **Ceze et al. (2019, *Nature Reviews Genetics*)** — comprehensive review of DNA digital data storage
- **Yazdi et al. (2015, *Nucleic Acids Research*)** — rewritable DNA-based storage
- **Antkowiak et al. (2020)** — low-cost DNA data storage using error-correcting codes and neural networks
- **Heckel et al. (2019, *Nat. Commun.*)** — characterizing errors in synthetic DNA for storage

The transformer denoising approach extends this line of work by framing error correction as a learned sequence-to-sequence problem, analogous to how neural machine translation corrects OCR output.

---

## License

MIT © Daniel Schmidt
