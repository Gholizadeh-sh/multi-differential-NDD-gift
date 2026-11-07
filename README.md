# Corrected GIFT-64 neural-distinguisher experiments

Start with [START_HERE_FA.md](START_HERE_FA.md) for Persian instructions.

This package replaces the **cipher and experiment implementation**, not the old
experimental numbers. Do not reuse datasets or trained models produced by the
uploaded generators. The corrected 4/6/7/8-round results must be measured afresh.

## Install (Linux, Python 3.10+)

Extract the ZIP, open a terminal in `gift64_corrected`, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_reference.py
```

`verify_reference.py` additionally requires `g++`; on Debian/Ubuntu install
`build-essential` if it is missing. The 10 Python correctness tests do not need a
compiler. The vendored official files make verification possible without GitHub
access. Training needs PyTorch; GPU use is optional. For a CPU-only machine,
install the CPU build of PyTorch first to avoid unnecessary CUDA packages:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The code was checked with the versions recorded in
`verification/environment_versions.json`. `requirements.txt` gives compatible
ranges rather than pretending to guarantee bitwise identity across platforms.

## Run the paper configurations

First perform a small pipeline check (these are NOT paper results):

```bash
python run_experiments.py --out results/smoke --n 2048 --epochs 2 --seeds 0
```

Then run all eight distinct configurations in Tables III--VI at full size:

```bash
python run_experiments.py --out results/paper_seed0 --n 1048576 --epochs 15 --seeds 0
```

For the revised manuscript, use five repetitions with independently changed data
and training seeds under the **same fixed master key**:

```bash
python run_experiments.py --out results/paper_5seeds --n 1048576 --epochs 15 --seeds 0 1 2 3 4
```

The latter is still a single-key experiment, not a generalization-to-new-keys
experiment. The suite uses the original numerical key
`0x0123456789ABCDEF0123456789ABCDEF` throughout. Use `--key 0x...` only for an
explicitly separate key study and choose another output directory.

| Configuration | Rounds | Differences, in feature-row order | Paper table |
| --- | ---: | --- | --- |
| `r4_A_1` | 4 | `000000000000000A`, `0000000000000001` | III |
| `r6_1_A` | 6 | `0000000000000001`, `000000000000000A` | IV, V |
| `r7_1_A` | 7 | `0000000000000001`, `000000000000000A` | IV |
| `r8_1_A` | 8 | `0000000000000001`, `000000000000000A` | IV |
| `r6_AAAA_F0F0` | 6 | `AAAAAAAAAAAAAAAA`, `F0F0F0F0F0F0F0F0` | V |
| `r6_0101_FFFF` | 6 | `0101010101010101`, `FFFFFFFF00000000` | V |
| `r6_weak4` | 6 | `0101010101010101`, `FFFFFFFF00000000`, `AAAAAAAAAAAAAAAA`, `F0F0F0F0F0F0F0F0` | VI |
| `r6_mixed4` | 6 | `0101010101010101`, `FFFFFFFF00000000`, `0000000000000001`, `000000000000000A` | VI |

Names such as `weak4` are historical identifiers; the corrected experiments must
re-establish whether those differences are weak. Tables IV and V share one
identical six-round experiment. The old two slightly different reported scores
for that setting are replaced by one controlled run/repetition summary.

Optional `--include-controls` adds single-difference `1`, single-difference `A`
at six rounds, and a 28-round two-difference diagnostic. These help interpret
the main results, but are not an equal-encryption-budget comparison. With `k`
rows, the first layer has more parameters when `k` increases. Do not describe
the network's parameter count as identical across matrix heights.

Default: Adam `lr=0.001`, batch 128, 15 epochs, 128/64 ReLU hidden units,
two output logits and cross-entropy, one stratified 70/30 split (seed 42),
final-epoch evaluation only. No test-based model/epoch selection occurs.
Entire matrices remain together during splitting. Bases in the same XOR-span
orbit of the selected deltas cannot cross train/test; the program detects and
rejects that rare event instead of silently leaking related examples.

## Outputs

Each run writes `metrics.json`, `history.json`, `model.pt`,
`split_indices.npz`, `predictions.npz`, and `hamming_histograms.npz`.
The top-level output contains `summary.csv`, `summary.json`,
`results_table.tex`, and `run_manifest.json`.

`metrics.json` includes accuracy, ROC-AUC, confusion matrix, key, deltas,
data/training/split seeds, implementation/data fingerprints, parameter count,
query counts, software versions, and timing. The Wilson accuracy interval is
conditional on that test protocol; it does not measure variation across keys
or independent runs. `summary.csv` reports the sample standard deviation across
repetitions when more than one seed is supplied. The LaTeX table reports means;
take standard deviations from the CSV when preparing mean +/- std tables.

Recreate the held-out-test Hamming-weight diagnostic for Figure 2 with:

```bash
python plot_hamming.py results/paper_seed0 --seed 0 --out results/paper_seed0/hamming_corrected.pdf
```

Only one dataset is generated and loaded at a time. By default the temporary
dataset is deleted after its successful run; its seed, source hashes, and
content fingerprint remain recorded. Add `--keep-datasets` to retain every NPZ.
For `N=2^20`, raw X is 128 MiB for two rows or 256 MiB for four rows. Training
keeps X as uint8 in system RAM and transfers only floating-point batches to the
selected device. Allow extra RAM for loading, splitting, and diagnostic arrays.

Use `--device cpu` to force CPU, or `--device cuda` for a working CUDA PyTorch
installation. `--threads 2` controls CPU threads and can be adjusted to the
machine. No runtime estimate is promised without benchmarking that machine.

Resume completed configurations with the **same command and `--resume`**.
Source/settings mismatches are rejected. A partially trained run is not resumed
from an optimizer checkpoint: preserve/move its directory aside, then use
`--resume` to restart that configuration. Existing outputs are never silently
overwritten. Completed runs are skipped, not retrained.

## Individual datasets and notebook

```bash
python data.py --out data/r6_1_A.npz --rounds 6 --n 1048576 --deltas 0x1,0xa --seed 0
python train.py data/r6_1_A.npz --out results/custom_r6 --seed 0
```

`single-diff-generator.py` defaults to one row (`A`);
`multi-diff-generator.py` defaults to two (`1,A`). Both use `data.py`, so the
cipher is not duplicated. The archive always uses shape `(N,k,64)`, including
`k=1`; training flattens each matrix internally. `model.ipynb` is a small local
notebook using the same modules, with no Google Drive dependency.

## Cipher and negative-class conventions

All integers follow the designers' normal displayed hexadecimal notation.
State bit 0 is the LSB; feature columns are MSB first. A reduced cipher consists
of the first r complete rounds, with constants starting at `01`, no initial
whitening, no extra final layer, and no replacement key schedule.

The default negative class literally encrypts two independent random plaintexts
per row, matching the manuscript. `--negative uniform` is an optional faster
generator with the same distribution for a fixed permutation, but performs
fewer actual encryptions; its use and both cost figures are recorded. Use the
default when reproducing the manuscript's operational cost. Zero and repeated
deltas are rejected unless `--allow-trivial` is explicitly requested for a
control. They do not belong in the substantive experiment tables.

Official reference and full round traces:
https://github.com/giftcipher/gift/tree/f43c6b3e214e39c22dc0ae5efcc39297809d9b68/implementations

Read [AUDIT.md](AUDIT.md) for the defects in the uploaded code and manuscript
changes that must follow from rerunning the corrected implementation.
