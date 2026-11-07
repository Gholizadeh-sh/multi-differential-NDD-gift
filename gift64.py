"""GIFT-64-128, canonical integer notation: bit 0 is the least significant bit.

One round is SubCells -> PermBits -> AddRoundKey (including constants).
Reduced-round encryption is a prefix of the standard 28-round cipher.
This research implementation is not constant-time.
"""
from __future__ import annotations

import operator
import numpy as np

SBOX = (1, 10, 4, 12, 6, 15, 3, 9, 2, 13, 11, 7, 5, 0, 8, 14)
PBOX = (
    0,17,34,51,48,1,18,35,32,49,2,19,16,33,50,3,
    4,21,38,55,52,5,22,39,36,53,6,23,20,37,54,7,
    8,25,42,59,56,9,26,43,40,57,10,27,24,41,58,11,
    12,29,46,63,60,13,30,47,44,61,14,31,28,45,62,15,
)
ROUND_CONSTANTS = (
    0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,
    0x1E,0x3C,0x39,0x33,0x27,0x0E,0x1D,0x3A,0x35,0x2B,
    0x16,0x2C,0x18,0x30,0x21,0x02,0x05,0x0B,
)


def unsigned(value, width, name):
    value = operator.index(value)
    if not 0 <= value < (1 << width):
        raise ValueError(f"{name} must fit in {width} unsigned bits")
    return value


def check_rounds(rounds):
    rounds = operator.index(rounds)
    if not 0 <= rounds <= 28:
        raise ValueError("rounds must be between 0 and 28 inclusive")
    return rounds


def substitute(state):
    return sum(SBOX[(state >> (4*i)) & 15] << (4*i) for i in range(16))


def permute(state):
    return sum(((state >> i) & 1) << PBOX[i] for i in range(64))


def update_key(key):
    """k7||...||k0 -> ROR2(k1)||ROR12(k0)||k7||...||k2."""
    key = unsigned(key, 128, "key")
    k0, k1 = key & 0xffff, (key >> 16) & 0xffff
    r0 = ((k0 >> 12) | (k0 << 4)) & 0xffff
    r1 = ((k1 >> 2) | (k1 << 14)) & 0xffff
    return (r1 << 112) | (r0 << 96) | (key >> 32)


def round_mask(key, round_index):
    u, v = key & 0xffff, (key >> 16) & 0xffff
    mask = 1 << 63
    for i in range(16):
        mask ^= ((u >> i) & 1) << (4*i)
        mask ^= ((v >> i) & 1) << (4*i + 1)
    rc = ROUND_CONSTANTS[round_index]
    for i in range(6):
        mask ^= ((rc >> i) & 1) << (4*i + 3)
    return mask


def expand_key(key, rounds=28):
    key = unsigned(key, 128, "key")
    rounds = check_rounds(rounds)
    masks = []
    for r in range(rounds):
        masks.append(round_mask(key, r))
        key = update_key(key)
    return tuple(masks)


def encrypt(plaintext, key, rounds=28):
    """Slow, readable scalar reference implementation."""
    state = unsigned(plaintext, 64, "plaintext")
    for mask in expand_key(key, rounds):
        state = permute(substitute(state)) ^ mask
    return state


def trace(plaintext, key, rounds=28):
    """Return independent observable round stages for official-vector checks."""
    state = unsigned(plaintext, 64, "plaintext")
    key = unsigned(key, 128, "key")
    stages = []
    for r in range(check_rounds(rounds)):
        sub = substitute(state)
        perm = permute(sub)
        state = perm ^ round_mask(key, r)
        key = update_key(key)
        stages.append((sub, perm, state, key))
    return stages


def decrypt(ciphertext, key, rounds=28):
    state = unsigned(ciphertext, 64, "ciphertext")
    inv = tuple(SBOX.index(i) for i in range(16))
    for mask in reversed(expand_key(key, rounds)):
        state ^= mask
        state = sum(((state >> PBOX[i]) & 1) << i for i in range(64))
        state = sum(inv[(state >> (4*i)) & 15] << (4*i) for i in range(16))
    return state


# Fused byte SubCells/PermBits lookup. Only 8*256*8 = 16 KiB.
# Scalar encrypt() does not use these tables.
_SP = np.zeros((8, 256), dtype=np.uint64)
for _byte in range(8):
    for _value in range(256):
        _s = SBOX[_value & 15] | (SBOX[_value >> 4] << 4)
        _SP[_byte, _value] = sum(
            ((_s >> b) & 1) << PBOX[8*_byte + b] for b in range(8)
        )


class Gift64:
    """Vectorized cipher with cached key expansion for one fixed master key."""
    def __init__(self, key, rounds=28):
        self.key = unsigned(key, 128, "key")
        self.rounds = check_rounds(rounds)
        self.masks = expand_key(self.key, self.rounds)

    def encrypt(self, plaintexts):
        values = np.asarray(plaintexts)
        if values.dtype.kind not in "ui" or (values.dtype.kind == "i" and np.any(values < 0)):
            raise ValueError("plaintexts must be unsigned 64-bit integers")
        state = values.astype(np.uint64, copy=True)
        for mask in self.masks:
            result = np.zeros_like(state)
            for b in range(8):
                result ^= _SP[b, ((state >> np.uint64(8*b)) & np.uint64(255)).astype(np.uint8)]
            state = result ^ np.uint64(mask)
        return state


def gift64_encrypt(plaintext, master_key, r_start=0, r_end=28):
    """Compatibility signature. Only prefix experiments are supported."""
    if r_start != 0:
        raise ValueError("Use a prefix starting at round 0; later-round segments are a different experiment")
    return encrypt(plaintext, master_key, r_end)


def int_to_bit_array(value, n=64):
    value = unsigned(value, n, "value")
    return np.array([(value >> i) & 1 for i in range(n-1, -1, -1)], dtype=np.uint8)


def bit_array_to_int(bits):
    bits = np.asarray(bits)
    if bits.shape != (64,) or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Expected exactly 64 binary bits, MSB first")
    return int.from_bytes(np.packbits(bits.astype(np.uint8)).tobytes(), "big")


def words_to_bits(words):
    """Last axis is 64 bits, MSB first, on every host byte order."""
    words = np.asarray(words, dtype=np.uint64)
    be = words.astype(">u8")
    return np.unpackbits(be.reshape(-1).view(np.uint8).reshape(words.shape + (8,)), axis=-1)


def bits_to_words(bits):
    bits = np.asarray(bits)
    if bits.ndim == 0 or bits.shape[-1] != 64 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Expected a last axis of 64 binary bits")
    packed = np.packbits(bits.astype(np.uint8), axis=-1)
    return packed.reshape(-1, 8).copy().view(">u8").reshape(bits.shape[:-1]).astype(np.uint64)
