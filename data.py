"""Balanced matrix datasets matching Algorithm 1 of the uploaded paper."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from gift64 import Gift64, unsigned, words_to_bits

DEFAULT_KEY = 0x0123456789ABCDEF0123456789ABCDEF
SCHEMA = "gift64-corrected-v1"


def random_words(rng, n):
    return rng.integers(0, 1 << 64, size=n, dtype=np.uint64)


def difference_span(deltas):
    span = {0}
    for d in deltas:
        span |= {x ^ d for x in tuple(span)}
    return sorted(span)


def orbit_ids(plaintexts, deltas):
    """Group bases related by XOR-span(S), a conservative overlap control."""
    if len(deltas) > 12:
        raise ValueError("At most 12 differences supported by the overlap check")
    result = plaintexts.copy()
    for d in difference_span(deltas):
        np.minimum(result, plaintexts ^ np.uint64(d), out=result)
    return result


def positive_words(bases, deltas, cipher):
    base_ct = cipher.encrypt(bases)  # Reuse exactly one base encryption.
    return np.stack([base_ct ^ cipher.encrypt(bases ^ np.uint64(d)) for d in deltas], axis=1)


def validate_deltas(deltas, allow_trivial=False):
    deltas = tuple(unsigned(d, 64, "difference") for d in deltas)
    if not deltas or len(deltas) > 12:
        raise ValueError("Supply 1 to 12 differences")
    if not allow_trivial and (0 in deltas or len(set(deltas)) != len(deltas)):
        raise ValueError("Zero/repeated differences are trivial controls; pass --allow-trivial explicitly")
    return deltas


def generate(n, deltas, key=DEFAULT_KEY, rounds=6, seed=0,
             negative="encrypted-pairs", chunk_size=32768, allow_trivial=False):
    if n < 4 or n % 2:
        raise ValueError("n is the TOTAL sample count and must be even, at least 4")
    if seed < 0 or chunk_size < 1:
        raise ValueError("seed must be nonnegative and chunk size positive")
    if negative not in ("encrypted-pairs", "uniform"):
        raise ValueError("Unknown negative distribution")
    deltas = validate_deltas(deltas, allow_trivial)
    cipher = Gift64(key, rounds)
    half, k = n // 2, len(deltas)
    # Separate streams per class, row, pair member: invariant to chunk size.
    streams = np.random.SeedSequence(seed).spawn(1 + 2*k)
    rng_base = np.random.default_rng(streams[0])
    rng_negative = [np.random.default_rng(s) for s in streams[1:]]
    X = np.empty((n, k, 64), dtype=np.uint8)
    y = np.concatenate((np.ones(half, np.uint8), np.zeros(half, np.uint8)))
    bases = random_words(rng_base, half)
    for start in range(0, half, chunk_size):
        end = min(start + chunk_size, half)
        X[start:end] = words_to_bits(positive_words(bases[start:end], deltas, cipher))
        for j in range(k):
            a = random_words(rng_negative[2*j], end-start)
            if negative == "encrypted-pairs":
                b = random_words(rng_negative[2*j+1], end-start)
                a = cipher.encrypt(a) ^ cipher.encrypt(b)
            X[half+start:half+end, j] = words_to_bits(a)
    groups = np.zeros(n, dtype=np.uint64)
    groups[:half] = orbit_ids(bases, deltas)
    theoretical = half * (k + 1 + 2*k)
    metadata = {
        "schema": SCHEMA, "cipher": "GIFT-64-128", "cipher_rounds": rounds,
        "implementation_sha256": hashlib.sha256(Path(__file__).with_name("gift64.py").read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "round_start": 0, "key_hex": f"{key:032x}", "n_total": n,
        "n_per_class": half, "deltas_hex": [f"{d:016x}" for d in deltas],
        "data_seed": seed, "negative": negative, "bit_order": "MSB-first",
        "shape": [n, k, 64], "allow_trivial": allow_trivial,
        "protocol_query_count": theoretical,
        "actual_encryption_calls": theoretical if negative == "encrypted-pairs" else half*(k+1),
        "positive_orbit_collisions": int(half-len(np.unique(groups[:half]))),
        "note": "uniform negatives have the same distribution as independent encrypted random pairs for a fixed permutation; query accounting differs",
    }
    return X, y, groups, metadata


def fingerprint(X, y, groups, metadata):
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for a in (X, y, groups):
        h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def save_dataset(path, X, y, groups, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; choose another path")
    metadata = dict(metadata, dataset_sha256=fingerprint(X, y, groups, metadata))
    with path.open("xb") as f:
        np.savez(f, X=X, y=y, positive_orbit_ids=groups,
                 metadata=np.array(json.dumps(metadata, sort_keys=True)))
    return metadata


def main(default_deltas="0x1,0xa", expected_rows=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=2**20, help="TOTAL samples, half per class")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--deltas", default=default_deltas)
    p.add_argument("--key", type=lambda s:int(s, 0), default=DEFAULT_KEY)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=32768)
    p.add_argument("--negative", choices=["encrypted-pairs", "uniform"], default="encrypted-pairs")
    p.add_argument("--allow-trivial", action="store_true")
    args = p.parse_args()
    deltas = [int(d, 0) for d in args.deltas.split(",")]
    if expected_rows is not None and len(deltas) != expected_rows:
        p.error(f"This entry point requires {expected_rows} difference(s); use data.py for general matrices")
    start = time.perf_counter()
    result = generate(args.n, deltas, args.key,
                      args.rounds, args.seed, args.negative, args.chunk_size, args.allow_trivial)
    metadata = save_dataset(args.out, *result)
    print(json.dumps(metadata, indent=2))
    print(f"Saved {args.out} in {time.perf_counter()-start:.2f} s", flush=True)


if __name__ == "__main__":
    main()
