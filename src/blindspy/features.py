"""Preprocessing steps used inside model pipelines (nothing here is applied to the data files).

The signed log sign(x) * log1p(|x|) compresses the heavy-tailed timing values (microseconds up
to about 1.2e8) for distance-based models. Unlike log1p it is defined for the -1 "not
measurable" codes and the negative min_flowiat values (E00), keeps the order of values and
keeps -1 apart from 0. It is a fixed formula, so nothing is learned from the data.
"""

import numpy as np
from sklearn.preprocessing import FunctionTransformer


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def signed_log_step() -> FunctionTransformer:
    return FunctionTransformer(signed_log, feature_names_out="one-to-one")


# E09 feature groups, found by name in the data's columns (A2 has total_fiat / total_biat where
# A1 and B have std_fiat / std_biat, E00): the gap statistics by suffix, the rest by name.
GROUP_SUFFIX = {"fiat": "_fiat", "biat": "_biat", "flowiat": "_flowiat", "active": "_active",
                "idle": "_idle"}  # fmt: skip
RATES_DURATION = ["duration", "flowPktsPerSecond", "flowBytesPerSecond"]
GROUPS = [*GROUP_SUFFIX, "rates_duration"]
FEATURE_SETS = ["all", *[f"only_{g}" for g in GROUPS], *[f"no_{g}" for g in GROUPS], "timing_only"]


def feature_groups(columns) -> dict[str, list[str]]:
    groups = {g: [c for c in columns if c.endswith(sfx)] for g, sfx in GROUP_SUFFIX.items()}
    groups["rates_duration"] = [c for c in columns if c in RATES_DURATION]
    return groups


def select_features(columns, feature_set: str) -> list[str]:
    """Columns kept by an E09 feature set: "all", "only_<group>", "no_<group>" or "timing_only"
    (drops flowBytesPerSecond, the only feature that uses packet sizes)."""
    columns = list(columns)
    groups = feature_groups(columns)
    if feature_set == "all":
        return columns
    if feature_set == "timing_only":
        return [c for c in columns if c != "flowBytesPerSecond"]
    kind, _, group = feature_set.partition("_")
    if kind in ("only", "no") and group in groups:
        keep = set(groups[group]) if kind == "only" else set(columns) - set(groups[group])
        return [c for c in columns if c in keep]
    raise ValueError(f"unknown feature set {feature_set!r}; one of {FEATURE_SETS}")
