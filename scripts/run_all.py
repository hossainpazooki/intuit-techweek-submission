"""Regenerate all four deliverables, then validate.

Usage:
    python scripts/run_all.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smb import pipeline  # noqa: E402


def main() -> int:
    pipeline.main()
    # Hand off to the validator so a run always ends with a PASS/FAIL verdict.
    return subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "validate.py")]
    )


if __name__ == "__main__":
    raise SystemExit(main())
