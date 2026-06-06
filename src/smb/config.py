"""Central configuration: paths, loan economics, and the cohort calendar.

Everything here is a *fact from the challenge brief or dataset*, not a modeling
choice — keep modeling hyperparameters in the individual deliverable modules so
this file stays a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[2]  # repo root
DATASET_DIR = ROOT / "dataset"
SUBMISSION_DIR = ROOT / "submission"
ARTIFACTS_DIR = ROOT / "artifacts"  # gitignored; models + intermediate frames
EXPECTED_IDS_DIR = ROOT / "expected_ids"

TRAIN_CSV = DATASET_DIR / "train.csv"
VALIDATION_CSV = DATASET_DIR / "validation.csv"
TEST_CSV = DATASET_DIR / "test.csv"
DATA_DICTIONARY_CSV = DATASET_DIR / "data_dictionary.csv"
INTERVENTION_QUERIES_CSV = DATASET_DIR / "intervention_queries.csv"
COHORT_WEEK_DEFINITIONS_CSV = DATASET_DIR / "cohort_week_definitions.csv"
SUBMISSION_B_TEMPLATE_CSV = DATASET_DIR / "submission_B_template.csv"

# Output file names — MUST be exact (the scorer joins on them).
FILE_A = SUBMISSION_DIR / "submission_A_decisions.csv"
FILE_B = SUBMISSION_DIR / "submission_B_trajectory.csv"
FILE_C = SUBMISSION_DIR / "submission_C_counterfactuals.csv"
FILE_D = SUBMISSION_DIR / "submission_D_writeup.pdf"

# --------------------------------------------------------------------------- #
# Loan product economics (fixed terms, from dataset/README.md)
# --------------------------------------------------------------------------- #

TERM_DAYS = 60                 # daily ACH draws over a 60-day term
APR = 0.35                     # annualized
ORIGINATION_FEE_RATE = 0.03    # 3% of amount, collected up front
DEFAULT_WINDOW_DAYS = 90       # outstanding balance > 0 at day 90 => default
AMOUNT_RANGE = (5_000, 50_000)  # product range, roughly

# Default triggers (any one): 3 consecutive missed draws, 6 cumulative missed
# draws, or positive balance at day 90. Encoded in data.py label logic / B model.
DEFAULT_CONSECUTIVE_MISSES = 3
DEFAULT_CUMULATIVE_MISSES = 6

# --------------------------------------------------------------------------- #
# Deliverable B grid + cohort calendar
# --------------------------------------------------------------------------- #

N_COHORT_WEEKS = 13            # 13 x 13 grid (validator: spec.n_cohort_weeks)
MAX_LOAN_AGE_WEEKS = 13
DAYS_PER_WEEK = 7              # cumulative default by day 7*age

# --------------------------------------------------------------------------- #
# Submission sizing (assertions to fail loudly if the data changes)
# --------------------------------------------------------------------------- #

N_DECISION_APPLICANTS = 13_306  # validation + test combined (Deliverable A)
N_INTERVENTION_QUERIES = 900    # Deliverable C

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

RANDOM_SEED = 42

# 90% prediction intervals everywhere.
INTERVAL_ALPHA = 0.10
INTERVAL_LOWER_Q = 0.05
INTERVAL_UPPER_Q = 0.95

# --------------------------------------------------------------------------- #
# Hazard spine + recovery + ensemble bagging
# --------------------------------------------------------------------------- #

WEEKS = 13              # discrete weekly periods in the competing-risks hazard
RECOVERY_PRIOR = 0.09   # default recovery rate prior for NPV economics
N_BAG_SEEDS = 5         # number of bagged hazard-model seeds in the ensemble

# Deliverable A decision rule (see policy.portfolio_decisions):
#   "flat"   -- break-even baseline (expected_npv at upper PD bound)   [BASELINE]
#   "timing" -- WS1 timing-integrated E[NPV] over default-week timing  [VALIDATED]
# WS1: timing-integrated E[NPV] is the headline rule (Step 0 baseline was "flat").
POLICY_RULE = "timing"
# Decision basis: False -> approve iff point E[NPV] > 0 (P&L-optimal primary rule);
# True -> conservative variant, decide at the lower E[NPV] credible bound.
POLICY_CONSERVATIVE = False
