"""Statistical helpers shared by the analysis notebooks (04 immune phenotypes, 05 TLS).

All models compare two groups coded 1 / 0 in an `exposure` column (e.g. inflamed = 1,
excluded = 0), so a positive effect means "higher in group 1".
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def zscore(series: pd.Series) -> pd.Series:
    """Standardise to mean 0 and SD 1 (NaN kept)."""
    return (series - series.mean()) / series.std()


def spearman_ci(x, y, n_boot: int = 1000, seed: int = 42, min_n: int = 5) -> tuple[float, float, float, int]:
    """Spearman rho with a percentile bootstrap 95 % CI; returns (rho, low, high, n)."""
    sub = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(sub) < min_n:
        return np.nan, np.nan, np.nan, len(sub)
    rho = spearmanr(sub["x"], sub["y"])[0]
    idx = np.random.default_rng(seed).integers(0, len(sub), size=(n_boot, len(sub)))
    boots = [spearmanr(sub["x"].values[b], sub["y"].values[b])[0] for b in idx]
    low, high = np.nanpercentile(boots, [2.5, 97.5])
    return rho, low, high, len(sub)


def fit_effect(data: pd.DataFrame, outcome: str, exposure: str, categorical=(), continuous=(),
               min_per_group: int = 5) -> dict:
    """OLS of `outcome` on `exposure` (+ covariates) with HC3 robust SEs.

    Returns the exposure coefficient with 95 % CI and p-value; NaN when either group has
    fewer than `min_per_group` observations or the model cannot be fitted.
    """
    d = data[[outcome, exposure, *categorical, *continuous]].dropna()
    n1, n0 = int((d[exposure] == 1).sum()), int((d[exposure] == 0).sum())
    out = {"effect": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan,
           "n": len(d), "n_group1": n1, "n_group0": n0}
    if min(n1, n0) < min_per_group:
        return out
    terms = [exposure] + [f'C(Q("{c}"))' for c in categorical] + [f'Q("{c}")' for c in continuous]
    try:
        fit = smf.ols(f'Q("{outcome}") ~ ' + " + ".join(terms), data=d).fit(cov_type="HC3")
    except Exception:
        return out
    low, high = fit.conf_int().loc[exposure]
    return {**out, "effect": fit.params[exposure], "ci_lower": low, "ci_upper": high,
            "p_value": fit.pvalues[exposure]}


def fit_logit(data: pd.DataFrame, outcome: str, exposure: str, categorical=(), continuous=(),
              min_level: int = 30, min_events: int = 10, min_events_per_group: int = 3) -> dict:
    """Logistic regression of a binary `outcome`; effect = log2 odds ratio of the exposure (with 95 % CI).

    Levels of the (first) categorical covariate with fewer than `min_level` observations are
    dropped. NaN when there are too few events or the model does not converge.
    """
    d = data[[outcome, exposure, *categorical, *continuous]].dropna()
    for c in categorical:
        d = d[d[c].map(d[c].value_counts()) >= min_level]
    out = {"effect": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan, "n": len(d)}
    ev1 = int(d.loc[d[exposure] == 1, outcome].sum())
    ev0 = int(d.loc[d[exposure] == 0, outcome].sum())
    if ev1 + ev0 < min_events or min(ev1, ev0) < min_events_per_group:
        return out
    terms = [exposure] + [f'C(Q("{c}"))' for c in categorical] + [f'Q("{c}")' for c in continuous]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.logit(f'Q("{outcome}") ~ ' + " + ".join(terms), data=d).fit(disp=0, maxiter=200)
    except Exception:
        return out
    if not fit.mle_retvals.get("converged", True):
        return out
    low, high = fit.conf_int().loc[exposure] / np.log(2)
    return {**out, "effect": fit.params[exposure] / np.log(2), "ci_lower": low, "ci_upper": high,
            "p_value": fit.pvalues[exposure]}


def add_fdr(frame: pd.DataFrame, by: str | None = None, p_col: str = "p_value") -> pd.DataFrame:
    """Add Benjamini-Hochberg q-values (`fdr`), computed within each level of `by` or over all rows."""
    frame = frame.copy()
    frame["fdr"] = np.nan
    groups = frame.groupby(by).groups.values() if by else [frame.index]
    for idx in groups:
        p = frame.loc[idx, p_col].dropna()
        if len(p):
            frame.loc[p.index, "fdr"] = multipletests(p, method="fdr_bh")[1]
    return frame


def stars(q: float, alpha: float = 0.05) -> str:
    """'***' q < 0.001, '**' q < 0.01, '*' q < alpha."""
    if q is None or np.isnan(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < alpha else ""


# Feature families used to colour the molecular screen (keyword match on the lower-case name).
FAMILY_KEYWORDS = [
    ("Stroma & matrix", ["col10a1", "matrix", "fibroblast", "tgfb", "stroma", "caf", "endothel", "angiogen"]),
    ("Cytotoxic & lymphoid", ["cd8", "effector", "ifng", "lymphocyte", "th1", "cytotoxic", "plasma", "b_cell",
                              "gamma_delta", "t_cells", "nk_"]),
    ("Brakes & myeloid", ["checkpoint", "treg", "macrophage", "myeloid", "monocyte", "neutrophil", "suppress",
                          "mast", "dendritic"]),
    ("Metabolism, hypoxia & proliferation", ["glycolysis", "hypoxia", "buffa", "proliferation"]),
    ("Genomics & antigens", ["tmb", "hrd", "aneuploidy", "heterogeneity", "neoantigen", "mhci"]),
]


def assign_family(feature: str) -> str:
    """Biological family of a molecular variable, from its prefix or name."""
    f = feature.lower()
    if f.startswith("mut_"):
        return "Driver mutations"
    if f.startswith("path_"):
        return "Oncogenic pathways"
    if f.startswith("aneu_"):
        return "Arm-level copy number"
    for family, words in FAMILY_KEYWORDS:
        if any(w in f for w in words):
            return family
    return "Other signatures"
