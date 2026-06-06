"""Person-period construction for the weekly competing-risks hazard spine.

We model each labeled (funded + matured) loan as a sequence of discrete weekly
periods (loan age a = 1..WEEKS). Each period emits one person-period row carrying
the loan's static features plus a ``loan_age_weeks`` column and a 3-class target:

    y = 0  survived this week (no terminal event)
    y = 1  defaulted this week
    y = 2  paid in full this week

A loan emits rows for weeks t = 1..T_i, where T_i is its terminal (event) week:
  - defaulters:  T_i = terminal_week(days_to_default)
  - payers:      T_i = terminal_week(days_to_full_repayment)  (== 9, day 60)

y is 1 exactly on the defaulter's terminal week, 2 exactly on the payer's
terminal week, and 0 on every earlier (survival) week.

Because outcomes exist ONLY on funded+matured loans there is no right-censoring
inside the labeled set: every labeled loan reaches a terminal event within the
90-day window. ``assert_no_censoring`` makes that assumption explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import features


def terminal_week(days, weeks: int = config.WEEKS) -> int:
    """Map a day count to its discrete weekly age in [1, weeks].

    Week a covers up to day 7a; week 9 ~ day 60, weeks up to 13 cover day 90.
    terminal_week(days) = clip(ceil(days / 7), 1, weeks).
    """
    a = int(np.ceil(float(days) / 7.0))
    return int(np.clip(a, 1, weeks))


def assert_no_censoring(labeled_train: pd.DataFrame) -> None:
    """Assert the labeled set has no right-censoring (all loans reached an event).

    Outcomes are populated only on funded+matured loans, so every labeled row
    should be 'matured' (or, equivalently, have a non-null default_flag).
    """
    if "observation_status" in labeled_train.columns:
        status = labeled_train["observation_status"].astype(str).str.strip().str.lower()
        non_matured = status[status != "matured"]
        assert non_matured.empty, (
            f"{len(non_matured)} labeled rows are not 'matured' "
            f"(found statuses: {sorted(non_matured.unique().tolist())})"
        )
    else:
        unlabeled = labeled_train["default_flag"].isna().sum()
        assert unlabeled == 0, f"{unlabeled} labeled rows have null default_flag"


def build_person_periods(labeled_train: pd.DataFrame):
    """Expand labeled loans into the weekly person-period (X_pp, y_pp) design.

    Returns:
        X_pp : pd.DataFrame of static features (features.build_features columns)
               plus an appended ``loan_age_weeks`` column, dtype float32, one row
               per (loan, week) up to the loan's terminal week.
        y_pp : np.ndarray of int targets in {0, 1, 2}, aligned to X_pp rows.
    """
    assert_no_censoring(labeled_train)

    df = labeled_train.reset_index(drop=True)

    # Static numeric feature matrix (stable, sorted columns; NaN preserved).
    static = features.build_features(df).astype("float32")
    static_cols = list(static.columns)

    default_flag = pd.to_numeric(df["default_flag"], errors="coerce").to_numpy()
    days_to_default = pd.to_numeric(df["days_to_default"], errors="coerce").to_numpy()
    days_to_full = pd.to_numeric(df["days_to_full_repayment"], errors="coerce").to_numpy()

    n = len(df)
    static_values = static.to_numpy(dtype="float32", copy=False)

    block_static: list[np.ndarray] = []
    block_age: list[np.ndarray] = []
    block_y: list[np.ndarray] = []

    for i in range(n):
        is_default = default_flag[i] == 1.0

        if is_default:
            t_event = terminal_week(days_to_default[i])
            event_class = 1
        else:
            # Payer (paid_in_full): days_to_full_repayment == 60 -> week 9.
            t_event = terminal_week(days_to_full[i])
            event_class = 2

        # Weeks 1..t_event: survive (0) until the terminal week carries the event.
        ages = np.arange(1, t_event + 1, dtype="float32")
        y = np.zeros(t_event, dtype="int64")
        y[-1] = event_class

        # Repeat this loan's static row once per emitted week.
        block_static.append(np.repeat(static_values[i : i + 1, :], t_event, axis=0))
        block_age.append(ages)
        block_y.append(y)

    static_stacked = np.concatenate(block_static, axis=0)
    age_stacked = np.concatenate(block_age, axis=0).astype("float32")
    y_pp = np.concatenate(block_y, axis=0)

    X_pp = pd.DataFrame(static_stacked, columns=static_cols, dtype="float32")
    X_pp["loan_age_weeks"] = age_stacked
    X_pp = X_pp.astype("float32")

    return X_pp, y_pp
