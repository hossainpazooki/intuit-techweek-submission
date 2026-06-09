"""Compute-scaling proof: run the scorecard at low/med/high and chart the curve.

Writes artifacts/compute_curve.csv (+ reports/ mirror) and artifacts/compute_curve.png
showing weighted_proxy and each sub-proxy vs the compute budget. The DoD wants this
monotone-improving, or diminishing returns stated explicitly.

Usage:
    python scripts/run_compute_curve.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from smb import compute as compute_mod  # noqa: E402
from smb import config  # noqa: E402
from scripts.run_scorecard import build_scorecard  # noqa: E402


def main() -> int:
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    rows = []
    for level in compute_mod.LEVELS:
        print(f"=== compute={level} ===")
        sc = build_scorecard(level)
        rows.append(
            {
                "compute": level,
                "bag_size": sc["budget"]["bag_size"],
                "weighted_proxy": round(sc["weighted_proxy"], 4),
                "pnl_norm": round(sc["pnl_proxy"]["norm"], 4),
                "traj_score": round(sc["traj_proxy"]["score"], 4),
                "cal_score": round(sc["cal_proxy"]["score"], 4),
                "cal_coverage": round(sc["cal_proxy"]["coverage"], 4),
                "c_norm": round(sc["c_proxy"]["norm"], 4),
                "write_frac": round(sc["writeup_proxy"]["frac"], 4),
            }
        )

    fields = list(rows[0].keys())
    for dest in (config.ARTIFACTS_DIR / "compute_curve.csv", ROOT / "reports" / "compute_curve.csv"):
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    # Plot.
    x = [r["compute"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in [
        ("weighted_proxy", "weighted (S)"),
        ("pnl_norm", "S_P&L norm"),
        ("traj_score", "S_traj"),
        ("cal_score", "S_cal"),
        ("c_norm", "S_C norm"),
    ]:
        ax.plot(x, [r[key] for r in rows], marker="o", label=label)
    ax.set_xlabel("compute budget")
    ax.set_ylabel("proxy score")
    ax.set_title("Proxy score vs compute budget")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.ARTIFACTS_DIR / "compute_curve.png", dpi=120)
    fig.savefig(ROOT / "reports" / "compute_curve.png", dpi=120)

    print("\ncompute curve:")
    for r in rows:
        print(f"  {r['compute']:>4}  weighted={r['weighted_proxy']:.4f}  "
              f"pnl={r['pnl_norm']:.3f} traj={r['traj_score']:.3f} "
              f"cal={r['cal_score']:.3f} C={r['c_norm']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
