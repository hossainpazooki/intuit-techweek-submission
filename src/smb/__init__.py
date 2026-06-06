"""SMB Underwriting Challenge — solution package.

Submodules map 1:1 onto the four deliverables:

    config      paths, loan economics, cohort calendar, RNG seed
    data        load/clean train/val/test, build labels, encode missingness
    features    feature engineering + intervenable-feature metadata
    model_pd    probability-of-default model (Deliverable A)
    policy      expected-profit approve/decline rule (Deliverable A)
    survival    cohort x loan-age cumulative-default forecast (Deliverable B)
    causal      do(feature=value) counterfactual estimation (Deliverable C)
    calibration shared 90% prediction-interval machinery (A/B/C)
    pipeline    orchestrates everything -> writes submission/*.csv
"""

__all__ = [
    "config",
    "data",
    "features",
    "model_pd",
    "policy",
    "survival",
    "causal",
    "calibration",
    "pipeline",
]
