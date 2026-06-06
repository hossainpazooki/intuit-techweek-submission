"""Run Intuit's official validator against our submission/ folder.

Thin wrapper around the shipped validate_submission.py so we always check the
real gate. Exits non-zero if the submission would be disqualified.

Usage:
    python scripts/validate.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    return subprocess.call(
        [
            sys.executable,
            str(ROOT / "validate_submission.py"),
            str(ROOT / "submission"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
