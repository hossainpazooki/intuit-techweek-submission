"""Feature engineering and the intervenable-feature metadata used by C.

The data dictionary marks each field with an ``intervenable`` flag. Deliverable C
only ever intervenes on intervenable features, so we load that mapping here and
expose the feature groups (business_identity, self_reported, bank_feed,
bureau_credit, platform_engagement, application_context, prior_underwriter).
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_data_dictionary() -> pd.DataFrame:
    """Read data_dictionary.csv (field, dtype, group, intervenable, notes)."""
    raise NotImplementedError


def intervenable_features() -> list[str]:
    """Feature names with intervenable=True — the only valid targets for C."""
    raise NotImplementedError


def feature_groups() -> dict[str, list[str]]:
    """group name -> list of field names, from the data dictionary."""
    raise NotImplementedError


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw split into the model matrix.

    TODO:
      - categorical encoding (sector, geography_region, bands are integer-coded)
      - missingness indicators (see data.missingness_indicators)
      - engineered ratios (requested_amount_to_observed_revenue is given; add
        debt-to-revenue, utilization buckets, etc.)
      - DO NOT include outcome columns; be deliberate about prior_* leakage.
    """
    raise NotImplementedError
