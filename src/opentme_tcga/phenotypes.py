"""Immune phenotypes from OpenTME lymphocyte densities.

* Hot / cold: molecular leukocyte fraction above / below a cohort quantile.
* Spatial phenotype within hot tumours, from the infiltration index
  log10((core + eps) / (outer margin + eps)): inflamed (>= threshold) or excluded.
* Median rules (inflamed / excluded / desert), as in TME Studio, used only to compare
  the whole-slide and tumour-anchored definitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hot_flag(leukocyte_fraction: pd.Series, quantile: float = 0.5) -> tuple[pd.Series, float]:
    """1.0 hot (>= cohort quantile), 0.0 cold, NaN if unknown; returns the flag and the threshold."""
    threshold = leukocyte_fraction.quantile(quantile)
    flag = (leukocyte_fraction >= threshold).astype(float).where(leukocyte_fraction.notna())
    return flag, threshold


def infiltration_index(core_mm2: pd.Series, margin_mm2: pd.Series, eps: float = 1e-2) -> pd.Series:
    """log10((core + eps) / (margin + eps)), densities in cells/mm²; > 0 = more lymphocytes in the core."""
    return np.log10((core_mm2 + eps) / (margin_mm2 + eps))


def spatial_phenotype(is_hot: pd.Series, index: pd.Series, threshold: float = 0.0) -> pd.Series:
    """'Inflamed' / 'Excluded' for hot tumours with an index, 'Cold' for cold tumours, None otherwise."""
    hot = is_hot == 1
    has = index.notna()
    return pd.Series(np.select([hot & has & (index >= threshold), hot & has & (index < threshold), is_hot == 0],
                               ["Inflamed", "Excluded", "Cold"], default=None),
                     index=index.index, dtype=object)


def median_rule(inside: pd.Series, outside: pd.Series) -> pd.Series:
    """TME Studio-style rule: inflamed if inside > median, else excluded if outside > median, else desert."""
    out = pd.Series(np.select([inside > inside.median(), outside > outside.median()],
                              ["inflamed", "excluded"], default="desert"), index=inside.index, dtype=object)
    return out.where(inside.notna() & outside.notna())


def add_densities_mm2(df: pd.DataFrame, image: dict, factors: dict[str, float], suffix: str = "_MM2") -> pd.DataFrame:
    """Add a cells/mm² copy (`<feature>_MM2`) of every density feature in the `image` config section.

    `image` maps a role to {feature, table}; `factors` are the per-table unit factors (density_per_mm2).
    """
    from opentme_tcga.opentme import density_per_mm2

    out = df.copy()
    for spec in image.values():
        if "DENSITY" in spec["feature"]:
            out[spec["feature"] + suffix] = density_per_mm2(out[spec["feature"]], spec["table"], factors)
    return out


TLS_CLASSES = ["No TLS", "Immature only", "Mature"]


def tls_class(tls_present: pd.Series, mature_present: pd.Series) -> pd.Series:
    """'No TLS', 'Immature only' (TLS but none mature) or 'Mature' (at least one mature TLS); None if unknown."""
    return pd.Series(np.select([tls_present == 0, (tls_present == 1) & (mature_present == 0), mature_present == 1],
                               TLS_CLASSES, default=None), index=tls_present.index, dtype=object)
