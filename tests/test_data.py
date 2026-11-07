import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from data import generate, positive_words, save_dataset, DEFAULT_KEY
from gift64 import Gift64, encrypt, bits_to_words
from train import load_dataset, split_indices


class DatasetTests(unittest.TestCase):
    def test_shared_base_against_scalar_oracle(self):
        bases = np.array([0,1,0x123456789abcdef0],np.uint64)
        deltas = (1,10,0x0101010101010101)
        words = positive_words(bases,deltas,Gift64(DEFAULT_KEY,6))
        for i,p in enumerate(bases):
            for j,d in enumerate(deltas):
                self.assertEqual(int(words[i,j]),encrypt(int(p),DEFAULT_KEY,6)^encrypt(int(p)^d,DEFAULT_KEY,6))

    def test_shape_balance_reproducible_across_chunks(self):
        a = generate(80,(1,10),seed=123,chunk_size=7)
        b = generate(80,(1,10),seed=123,chunk_size=40)
        for aa,bb in zip(a[:3],b[:3]): np.testing.assert_array_equal(aa,bb)
        self.assertEqual(a[0].shape,(80,2,64))
        self.assertEqual(a[0].dtype,np.uint8)
        self.assertEqual(a[1].sum(),40)
        self.assertEqual(a[3]["actual_encryption_calls"],280)

    def test_random_pairs_round_zero(self):
        X,y,_,_ = generate(40,(1,10),rounds=0,seed=33)
        np.testing.assert_array_equal(bits_to_words(X[y==1]),np.tile(np.array([1,10],np.uint64),(20,1)))
        streams = np.random.SeedSequence(33).spawn(5)
        for row in range(2):
            a = np.random.default_rng(streams[1+2*row]).integers(0,1<<64,20,dtype=np.uint64)
            b = np.random.default_rng(streams[2+2*row]).integers(0,1<<64,20,dtype=np.uint64)
            np.testing.assert_array_equal(bits_to_words(X[y==0,row]),a^b)

    def test_trivial_differences_require_explicit_control(self):
        for ds in [(0,1),(1,1),(),(-1,),(1<<64,)]:
            with self.assertRaises(ValueError): generate(20,ds)
        X,y,_,_ = generate(20,(0,1),allow_trivial=True)
        self.assertTrue(np.all(X[y==1,0]==0))
        with self.assertRaises(ValueError): generate(21,(1,))

    def test_save_load_integrity_and_legacy_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"new.npz"
            result = generate(40,(1,10))
            save_dataset(p,*result)
            loaded = load_dataset(p)
            for a,b in zip(result[:3],loaded[:3]): np.testing.assert_array_equal(a,b)
            train,test = split_indices(loaded[1],loaded[2])
            self.assertEqual(len(set(train)&set(test)),0)
            with self.assertRaises(FileExistsError): save_dataset(p,*result)
            legacy = Path(d)/"legacy.npz"
            np.savez(legacy,X=result[0],y=result[1])
            with self.assertRaises(ValueError): load_dataset(legacy)
            with np.load(p) as f: contents={k:f[k] for k in f.files}
            contents["X"][0,0,0] ^= 1
            np.savez(p,**contents)
            with self.assertRaises(ValueError): load_dataset(p)

    def test_related_plaintext_orbits_blocked_across_split(self):
        y=np.array([1]*20+[0]*20,dtype=np.uint8)
        groups=np.zeros(40,dtype=np.uint64)
        with self.assertRaisesRegex(ValueError,"orbits cross"):
            split_indices(y,groups)


if __name__ == "__main__": unittest.main()
