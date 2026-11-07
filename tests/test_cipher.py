import re
from pathlib import Path
import unittest
import numpy as np
from gift64 import (Gift64, encrypt, decrypt, trace, gift64_encrypt,
                    bit_array_to_int, int_to_bit_array, words_to_bits, bits_to_words)

ROOT = Path(__file__).resolve().parents[1]


class CipherTests(unittest.TestCase):
    def test_official_full_vectors_and_every_round_stage(self):
        for path in sorted((ROOT/"third_party/giftcipher").glob("GIFT64_test_vector_*.txt")):
            text = path.read_text().split("decryption...")[0]
            get = lambda pattern: int(re.search(pattern, text)[1].replace(" ", ""), 16)
            pt = get(r"Plaintext = ([0-9a-fA-F ]+)")
            key = get(r"masterkey = ([0-9a-fA-F ]+)")
            expected = get(r"Ciphertext = ([0-9a-fA-F ]+)")
            self.assertEqual(encrypt(pt,key), expected)
            self.assertEqual(int(Gift64(key).encrypt(np.array([pt], dtype=np.uint64))[0]), expected)
            self.assertEqual(decrypt(expected,key), pt)
            patterns = ["after SubCells", "after PermBits", "after AddRoundKeys", "updated Key"]
            columns = [[int(x.replace(" ", ""),16) for x in re.findall(
                rf"\d+: {stage}: ([0-9a-fA-F ]+)", text)] for stage in patterns]
            self.assertTrue(all(len(c) == 28 for c in columns))
            for r, actual in enumerate(trace(pt,key)):
                with self.subTest(vector=path.name, round=r):
                    self.assertEqual(actual, tuple(c[r] for c in columns))
                    self.assertEqual(encrypt(pt,key,r+1), actual[2])
                    self.assertEqual(int(Gift64(key,r+1).encrypt(np.array([pt],np.uint64))[0]), actual[2])

    def test_scalar_vectorized_and_inverse_all_rounds(self):
        rng = np.random.default_rng(314159)
        for rounds in range(29):
            key = int.from_bytes(rng.bytes(16), "big")
            pts = rng.integers(0,1<<64,size=20,dtype=np.uint64)
            expected = [encrypt(int(p),key,rounds) for p in pts]
            actual = Gift64(key,rounds).encrypt(pts)
            np.testing.assert_array_equal(actual, np.array(expected,np.uint64))
            self.assertEqual([decrypt(int(c),key,rounds) for c in actual], pts.tolist())
            np.testing.assert_array_equal(pts, Gift64(key,0).encrypt(pts))

    def test_big_endian_bit_convention(self):
        words = np.array([[0,1,1<<63,(1<<64)-1]], dtype=np.uint64)
        bits = words_to_bits(words)
        self.assertEqual(bits.shape,(1,4,64))
        self.assertEqual(bits[0,1,-1],1)
        self.assertEqual(bits[0,2,0],1)
        np.testing.assert_array_equal(bits_to_words(bits),words)
        for w in words.flat:
            self.assertEqual(bit_array_to_int(int_to_bit_array(int(w))),int(w))

    def test_validation(self):
        for rounds in (-1,29):
            with self.assertRaises(ValueError): Gift64(0,rounds)
        for pt,key in [(-1,0),(1<<64,0),(0,-1),(0,1<<128)]:
            with self.assertRaises(ValueError): encrypt(pt,key)
        with self.assertRaises(ValueError): gift64_encrypt(0,0,1,4)
        with self.assertRaises(ValueError): bit_array_to_int([0]*63)


if __name__ == "__main__": unittest.main()
