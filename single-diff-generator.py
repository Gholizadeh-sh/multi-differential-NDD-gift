"""Compatibility filename: delegate to the one corrected generator.
Example: python single-diff-generator.py --out data/single.npz --deltas 0xa
"""
from data import main
if __name__ == "__main__": main(default_deltas="0xa", expected_rows=1)
