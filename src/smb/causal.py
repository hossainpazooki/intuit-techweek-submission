"""Deliverable C — counterfactual PD under do(feature = value).

This is the most heavily weighted section of the writeup. The task asks for an
*interventional* prediction P(default | do(feature = v)), NOT the observational
P(default | feature = v). The difference, and how you defend it, is the whole
game.

intervention_queries.csv: (query_id, applicant_id, feature_name, intervention_value).
900 queries over 300 applicants and ~30 intervenable features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_queries() -> pd.DataFrame:
    """Read intervention_queries.csv."""
    raise NotImplementedError


def estimate_counterfactual(model, applicant_rows: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Estimate post-intervention PD for each query.

    Spectrum of rigor (pick one and DEFEND it in Deliverable D):

      (a) Naive model perturbation: set the feature, re-predict with the PD
          model. Fast, but conflates observation with intervention and ignores
          downstream effects on correlated features. State what you give up.
      (b) Causal-graph aware: posit a DAG over the intervenable features, and
          when you set X, propagate to its causal descendants before predicting
          (e.g. via structural equations / a learned SCM).
      (c) Refute: for each driver, articulate how you'd defend it to a regulator
          (sign, magnitude, mechanism), and flag features that are proxies.

    Returns columns exactly:
        query_id, predicted_pd_cf, pd_cf_lower_90, pd_cf_upper_90
    """
    raise NotImplementedError
