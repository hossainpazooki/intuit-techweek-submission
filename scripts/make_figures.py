"""Render writeup figures from the scorecard artifacts (no model retraining).

Reads artifacts/scorecard.json and writes:
  artifacts/pnl_backtest.png  -- realized portfolio NPV on the OOT funded holdout
                                 under each policy (+ reports/ mirror)

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from smb import config  # noqa: E402


def main() -> int:
    sc = json.loads((config.ARTIFACTS_DIR / "scorecard.json").read_text())
    p = sc["pnl_proxy"]
    labels = ["approve-all\n(=legacy)", "flat\n(baseline)", "timing\n(WS1)", "oracle\n(hindsight)"]
    vals = [
        p["approve_all_total"],
        p["total_flat_cons"],
        p["total_timing_point"],
        p["oracle_total"],
    ]
    colors = ["#b0b0b0", "#c98a3a", "#2a7", "#48c"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, [v / 1e6 for v in vals], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("realized portfolio NPV ($M)")
    ax.set_title(f"Realized P&L on OOT funded holdout (n={p['n_holdout']}, compute={sc['compute']})")
    for b, v in zip(bars, vals):
        ax.annotate(f"${v/1e6:.2f}M", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for dest in (config.ARTIFACTS_DIR / "pnl_backtest.png", ROOT / "reports" / "pnl_backtest.png"):
        fig.savefig(dest, dpi=120)
    print(f"wrote pnl_backtest.png (timing +${p['uplift_vs_flat_baseline']:,.0f} vs flat baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
