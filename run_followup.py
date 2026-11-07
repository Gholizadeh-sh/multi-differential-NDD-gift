"""Select the three agreed six-round cases using the original experiment suite.

Copy this file next to the unmodified run_experiments.py. All cipher, dataset,
training, output, and resume operations remain those of that original package.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys


CONFIGURATIONS = ("r6_1_A", "r6_weak4", "r6_mixed4")
EXPECTED_SOURCES = {
    "gift64.py": "06db3a8dfc57950786ad2dabd0fb1a75f6ce142819086c93bc10593c4f62cc8a",
    "data.py": "80a83ae633d2b9c64983b40daf86b4ad3a2e1bfa7b5da03f10a507404011fc73",
    "train.py": "d0c7e7e16f4cf6c43b76415b70e3aeb024932aa90b86770f25a042ceee34aaef",
    "run_experiments.py": "f624e3ae6abf7324f9a11c8d5ef5dd378ba019ca46a2de745341090a2cf2f856",
}


def main():
    root = Path(__file__).resolve().parent
    for filename, expected in EXPECTED_SOURCES.items():
        path = root / filename
        if not path.is_file():
            raise SystemExit(
                f"Missing {filename}. Put run_followup.py inside the original gift64_corrected directory."
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(
                f"{filename} differs from the corrected sources used for seed zero. "
                "Stop and check the source version before comparing runs."
            )
    if "--include-controls" in sys.argv[1:]:
        raise SystemExit("This follow-up runs only the three agreed configurations; omit --include-controls.")

    import run_experiments as suite

    suite.PAPER = [case for case in suite.PAPER if case[0] in CONFIGURATIONS]
    if tuple(case[0] for case in suite.PAPER) != CONFIGURATIONS:
        raise SystemExit("Unexpected configuration definitions in run_experiments.py.")
    suite.__doc__ = "Run only r6_1_A, r6_weak4, and r6_mixed4 using the original corrected suite."
    suite.main()


if __name__ == "__main__":
    main()
