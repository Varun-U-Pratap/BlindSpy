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
