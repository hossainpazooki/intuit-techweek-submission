"""SMB Underwriting Challenge — solution package.

The modeling spine is a weekly competing-risks discrete-time hazard (default vs
payoff) fit unweighted on funded labeled data; Deliverables A/B/C all read off that
one fit, with an out-of-time isotonic calibration on top.

    config        paths, loan economics, cohort calendar, OOT-calibration flag
    data          load/clean train/val/test, build labels, encode missingness
    features      numeric feature matrix (excludes prior_* selection cols) + C metadata
    survival_data weekly person-period builder (competing-risks layout)
    survival      competing-risks hazard ensemble; CIF; lifetime PD; B trajectory
    recovery      empirical recovery-rate estimate for the NPV economics
    propensity    funding-rule + positivity DIAGNOSTICS only (no IPW; not in the fit)
    model_pd      standalone binary HistGB+isotonic PD baseline (unweighted)
    policy        upper-bound NPV approve/decline rule (Deliverable A)
    survival      cohort x loan-age cumulative-default forecast (Deliverable B)
    causal        do(feature=value) counterfactual estimation (Deliverable C)
    calibration   isotonic (incl. OOT) + 90% prediction-interval machinery (A/B/C)
    dag           intervention DAG + structural propagation (Deliverable C)
    pipeline      orchestrates everything -> writes submission/*.csv + validates
"""

__all__ = [
    "config",
    "data",
    "features",
    "survival_data",
    "survival",
    "recovery",
    "propensity",
    "model_pd",
    "policy",
    "causal",
    "calibration",
    "dag",
    "pipeline",
]
