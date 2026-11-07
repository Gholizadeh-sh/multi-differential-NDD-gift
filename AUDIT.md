# Audit of the uploaded GIFT-64 experiment implementation

Date: 2026-11-02.

Inputs: `multi-differential-NDD-gift-main (1).zip` (two generators and one
notebook) and `1250___54304.pdf` (eight-page manuscript). This audit concerns
the submitted snapshot, not an unseen revision of the GitHub repository.
The archive contains no experimental datasets, saved models, run logs, or
manuscript LaTeX. Consequently, the historical table values cannot be
independently reproduced or their exact generating snapshot established.

## Conclusion

Both uploaded generators implement the same nonstandard 64-bit permutation,
not GIFT-64-128. If these generators produced the manuscript's data, their
accuracy/AUC figures are not measurements of GIFT-64. All cipher-dependent
tables, diagnostic plots, and associated claims need fresh experiments.
This invalidates the old implementation as evidence; it does not establish
in advance whether the corrected experiment will succeed or fail.

## Cipher defects (line numbers refer to the uploaded files)

| Defect | Evidence in old multi-diff-generator.py | Correct behavior |
| --- | --- | --- |
| Wrong PermBits | Line 6: `16*(i%4)+i//4` | Designers' GIFT-64 permutation; 60 of 64 destinations differ. For example P(1)=17, not 16; P(4)=48, not 1. |
| Wrong round order | Lines 72--77 add the key before substitution and permutation | SubCells, then PermBits, then AddRoundKey/constants. |
| Wrong subkey words | Lines 49--52 extract bits 112--127 and 80--95 | With conventional integer notation, U is key bits 0--15 and V is bits 16--31. |
| Wrong state positions | Lines 73--75 inject into `2*i` and `2*i+1`, reaching only the low 32 state bits | U_i into bit 4i, V_i into bit 4i+1, for i=0..15. |
| Missing state constants | No state injection of RC or the fixed high bit | RC bits 0..5 into state positions 3,7,11,15,19,23, plus XOR 1 at state bit 63. |
| Wrong key schedule | Lines 55--64 rotate words in place, apply an S-box, and inject RC into key words; no whole-key word movement | `k7||...||k0 -> ROR2(k1)||ROR12(k0)||k7||...||k2`. No S-box and no RC injection in the key schedule. |

The old S-box values and 28 listed round constants are correct. Their correct
values do not compensate for the incorrect layer structure. The single-
difference generator repeats these same defects in its own cipher copy.

The defects are observable immediately, with no ML training:

| Plaintext/key | Rounds | Uploaded code (both files) | Official/corrected |
| --- | ---: | --- | --- |
| Both zero | 1 | `000000000000ffff` | `9111111111111119` |
| Both zero | 28 | `de5e812b34970b44` | `f62bc3ef34f775ac` |

See `verification/legacy_comparison.json` for hashes of the exact uploaded
generator files and these measured outputs.

## Data generation and experiment issues

1. **Hidden sample-count dependence.** The old single generator creates
   200,000 plaintext rows when its file is absent and uses half as positives,
   for 200,000 total samples. The multi generator requests 2^20 total, but
   reuses the same `data/plaintexts.npy` and never extends a short file. If
   the single generator runs first, multi produces only 400,000 total.
   If multi runs first, single receives 524,288 plaintext rows and produces
   524,288 total samples. Thus script defaults/order do not reproduce a
   common 2^20-sample protocol. This is an observed code path, not proof of
   which order was used for the reported measurements.
2. **Unrecorded data randomness.** Dataset generation does not set or save
   a seed. The notebook seeds training after generation; it cannot recreate
   an already-generated dataset. Python `random` and NumPy are also mixed.
3. **Query cost implementation mismatch.** The old multi generator computes
   C(P) inside the row loop: 2k encryptions per positive sample, rather than
   k+1 calls through cached reuse. Including negatives, it actually performs
   2Nk calls on N balanced samples. The paper's N(3k+1)/2 expression is a
   valid optimized protocol cost, but does not describe those old call counts.
4. **Negative distributions need careful interpretation.** Single generates
   a uniform ciphertext-XOR directly; multi encrypts independent random
   plaintext pairs. Both have exactly the same row distribution for any
   fixed permutation, including the old incorrect cipher. This is not a
   statistical error by itself, but actual encryption costs differ.
5. **No provenance in NPZ.** Old archives contain only X and y: no key,
   rounds, differences, seeds, source fingerprint, or query count. Filenames
   are hardcoded and one success message names a different file. Corrected
   training rejects these unverified old archives.
6. **Scalability and local execution.** Repeated scalar Python encryption,
   repeated key expansion, nested list construction, and transferring entire
   float32 datasets to the GPU are avoidable costs. The notebook is coupled
   to Google Colab/Drive and has an unresolved dataset placeholder.
7. **No cipher validation gate.** High four-round classification accuracy
   does not validate a cipher implementation. A wrong low-diffusion SPN
   can also be easy to classify. The manuscript statement that an easy
   four-round classification result validates the implementation needs to
   be replaced with actual known-answer/reference testing.
8. **Interpretation limitations remain.** The paper intentionally uses one
   fixed key and shared-base positives versus independent-pair negatives.
   Those are protocol choices, not automatic bugs. The corrected package
   preserves them and does not claim unseen-key generalization or isolate
   which cross-row relation drives a classifier. In particular,
   X_i XOR X_j = E(P XOR delta_i) XOR E(P XOR delta_j), exposing additional
   pairwise differences delta_i XOR delta_j. More rows also increase the
   first layer's parameter count. Neither factor is independently ablated
   by the original tables.

## Corrected implementation and validation

`gift64.py` contains a readable scalar implementation, inverse, round trace,
and a faster NumPy implementation using a fused byte lookup for SubCells and
PermBits. One cached fixed-key expansion is reused throughout each dataset.
No external cryptography package or native binary is required for generation.

Reference authority: the designers' site https://giftcipher.github.io/gift/
links to https://github.com/giftcipher/gift for reference code and vectors.
The vendored source/vectors are pinned to revision
`f43c6b3e214e39c22dc0ae5efcc39297809d9b68`; their MIT license and hashes
are included. Source: `implementations/reference/GIFT64-128_cipher.cpp`;
vectors: `implementations/test vectors/GIFT64_test_vector_{1,2,3}.txt`.

Verification performed here:

- All 3 official full-round known-answer vectors.
- All 28 rounds of all 3 official traces, comparing four stages per round:
  SubCells, PermBits, AddRoundKeys, updated key (336 stage comparisons).
- Vectorized implementation checked against official prefix outputs too.
- 560 random plaintext/key/round combinations (20 per round 1..28)
  compared to a compiled harness calling the designers' unchanged C++ source.
- Scalar/vectorized agreement and decryption checks on 20 random blocks at
  each round count 0..28.
- Dataset checks covering shared bases, bit order, exact negative-pair
  construction at round zero, seeds, chunk invariance, class balance,
  trivial-difference rejection, matrix-level split/orbit overlap, fingerprints,
  legacy rejection, and overwrite protection.
- An end-to-end small execution of all eight paper configurations and the
  three additional controls, N=2048 and 2 epochs. This tests integration and
  output production, not the paper's accuracy or round coverage.

`verification/` records the checks and the explicitly labelled small-run
summary. No full-size paper results are claimed or supplied in this package.

## Rerun and manuscript update checklist

1. Preserve the old runs only as historical material. Regenerate all features
   and train fresh models; weights from the old cipher are not reusable.
2. Run the exact eight table configurations using N=2^20 **total** samples,
   original MLP and training parameters. Use one consistent six-round `1,A`
   result for both Tables IV and V. Five independent data/training-seed
   repetitions are recommended; keys remain fixed in this protocol.
3. Rebuild Tables III--VI from actual completed outputs. The prior-work
   contextual entries in Table III are not recomputed by this package and
   should still be source-checked before submission. Regenerate Figure 2
   from held-out-test histograms with explicit row labels.
4. Reassess the abstract, Section VI interpretations, Section VII limits,
   conclusion, and any numerical statements. Claims of monotonic round
   degradation, localized-difference superiority, weak-row dominance,
   100% accuracy, or an eight-round empirical boundary must follow the new
   observations rather than be retained by assumption.
5. Describe formal known-answer and reference validation, data seeds,
   fixed-key scope, negative-class generation, and repetitions. The new
   default actually reuses the base encryption, making the paper's stated
   protocol cost and generator call count agree.
6. Do not call added single-row controls an equal-query comparison: use
   explicit budget-normalized datasets if that additional claim is pursued.
   Do not tune on the held-out test set to regain old numerical results.

This package does not edit the supplied PDF or invent replacement results;
the manuscript LaTeX and full local run outputs will be needed for its final
scientific revision.
