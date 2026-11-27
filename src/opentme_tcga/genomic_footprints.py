"""Genomic footprints in tissue architecture.

Tools to test whether a molecular variable (a mutation, a copy-number event,
a pathway alteration, a continuous score) leaves a recognisable mark on the
tumour and its microenvironment, as captured by interpretable H&E features,
beyond what subtype and purity already explain.

Two complementary analyses are provided:

* An unbiased screen (`footprint_screen`, `omnibus_permutation_test`) that
  crosses every genomic variable with every image feature and asks, for each
  genomic variable, whether the image as a whole differs more than expected by
  chance, using permutations within subtype strata that preserve the
  correlation structure among image features.
* A hypothesis-driven analysis (`tissue_footprint`, `footprint_map`) on a
  pre-specified short list of features with expected directions, with exact
  OLS confidence intervals.

All functions are agnostic to the cancer type and to the molecular variable;
only the configuration changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Molecular targets
# ---------------------------------------------------------------------------
@dataclass
class MolecularTarget:
    """A molecular variable whose tissue footprint is tested."""

    name: str
    variable: str
    kind: str = "binary"                      # "binary" or "continuous"
    role: str = "exploratory"                 # "control", "main" or "exploratory"
    cohort: str | None = None                 # pandas query defining the analysable cohort
    extra_covariates: list[str] = field(default_factory=list)
    expected: dict[str, str] = field(default_factory=dict)  # feature family -> "up" / "down"
    use_case: str | None = None               # translational question the target informs

    @classmethod
    def from_config(cls, name: str, spec: dict) -> "MolecularTarget":
        return cls(
            name=name,
            variable=spec["variable"],
            kind=spec.get("type", "binary"),
            role=spec.get("role", "exploratory"),
            cohort=spec.get("cohort"),
            extra_covariates=spec.get("extra_covariates", []),
            expected=spec.get("expected", {}),
            use_case=spec.get("use_case"),
        )


def discover_targets(data: pd.DataFrame, binary_prefixes: list[str], continuous: list[str],
                     exclude: list[str] | None = None, min_altered: int = 20) -> list[MolecularTarget]:
    """Build the list of genomic targets for the screen from column prefixes.

    Binary targets with fewer than `min_altered` altered (or non-altered) cases
    are dropped, since their footprint cannot be estimated reliably.
    """
    exclude = set(exclude or [])
    targets = []
    for col in data.columns:
        if col in exclude or not any(col.startswith(p) for p in binary_prefixes):
            continue
        x = data[col].dropna()
        n_alt = int(x.sum())
        if min(n_alt, len(x) - n_alt) >= min_altered:
            targets.append(MolecularTarget(name=col, variable=col, kind="binary"))
    for col in continuous:
        if col in data.columns and col not in exclude:
            targets.append(MolecularTarget(name=col, variable=col, kind="continuous"))
    return targets



def two_hit(mutation: pd.Series, arm_loss: pd.Series) -> pd.Series:
    """Both alleles hit: mutation of a gene and loss of the chromosome arm carrying the other copy.

    Defined (0 / 1) only where both inputs are known; missing otherwise.
    """
    both = ((mutation == 1) & (arm_loss == 1)).astype(float)
    return both.where(mutation.notna() & arm_loss.notna())


def resolve_terms(terms: list[str], aliases: dict[str, str]) -> list[str]:
    """Covariate terms of the configuration as model terms: aliases (e.g. 'subtype' ->
    'C(clin_pam50)', 'purity' -> 'abs_purity') are replaced, other terms are kept as written."""
    return [aliases.get(t, t) for t in terms]

# ---------------------------------------------------------------------------
# Image feature preparation
# ---------------------------------------------------------------------------
def select_image_features(data: pd.DataFrame, candidates: list[str], drop_prefixes: list[str] = (),
                          drop_substrings: list[str] = (), drop_columns: list[str] = (),
                          max_missing: float = 0.20, max_dominant_fraction: float = 0.95) -> list[str]:
    """Image features suitable for the screen, from a list of candidate columns.

    Candidates are typically the columns of the OpenTME master table. The
    selection keeps numeric features only, and removes absolute counts and
    areas (which mainly reflect the amount of tissue), technical quality-control
    readouts, features with too many missing values and near-constant features,
    defined as those in which a single value covers more than
    `max_dominant_fraction` of the non-missing observations. This criterion
    keeps informative binary features (e.g. presence of tertiary lymphoid
    structures) while removing features that barely vary. None of these filters
    uses the genomic variables, so the selection is independent of the results.
    """
    keep = []
    for c in candidates:
        if c not in data.columns or c in drop_columns or not pd.api.types.is_numeric_dtype(data[c]):
            continue
        if any(c.startswith(p) for p in drop_prefixes) or any(s in c for s in drop_substrings):
            continue
        values = data[c].dropna()
        if data[c].isna().mean() > max_missing or values.nunique() < 2:
            continue
        if values.value_counts(normalize=True).iloc[0] > max_dominant_fraction:
            continue
        keep.append(c)
    return keep


_CELL_TYPES = ["CARCINOMA_CELL", "LYMPHOCYTE", "MACROPHAGE", "PLASMA_CELL", "GRANULOCYTE",
               "FIBROBLAST", "ENDOTHELIAL_CELL", "EPITHELIAL_CELL"]
_TISSUES = ["NECROSIS", "STROMA", "CARCINOMA", "VESSEL", "TLS"]


def feature_family(name: str) -> str:
    """Coarse family of an OpenTME feature: measurement type and main subject.

    E.g. 'RATIO_OF_MACROPHAGES_AROUND_CARCINOMA_CELL_IN_STROMA_20' ->
    'neighbourhood: macrophage'; 'NUCLEUS_AVG_AREA_CARCINOMA_CELL' ->
    'nucleus: carcinoma cell'.
    """
    n = name.upper()
    if n.startswith("NUCLEUS"):
        kind = "nucleus"
    elif "_AROUND_" in n:
        kind = "neighbourhood"
    elif n.startswith(("DENSITY", "CELL_")):
        kind = "cell density / composition"
    elif n.startswith("RELATIVE_AREA"):
        kind = "tissue composition"
    elif n.startswith("AVG_"):
        kind = "region shape"
    elif n.startswith("DERIVED"):
        kind = "derived index"
    else:
        kind = "other"
    subject_part = re.split(r"_AROUND_", n)[0] if "_AROUND_" in n else n
    for token in _CELL_TYPES:
        stem = token.rstrip("S")
        if stem in subject_part:
            return f"{kind}: {token.lower().replace('_', ' ')}"
    for token in _TISSUES:
        if token in n:
            return f"{kind}: {token.lower()}"
    return kind


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def rank_inverse_normal(s: pd.Series) -> pd.Series:
    """Rank-based inverse normal transformation (Blom offset).

    Maps the ranks of the non-missing values onto standard normal quantiles,
    so that skewed features and outliers cannot dominate a linear model.
    Ties receive the same value; missing values stay missing.
    """
    out = pd.Series(np.nan, index=s.index, dtype=float)
    values = s.dropna()
    if values.empty:
        return out
    ranks = values.rank(method="average")
    out[values.index] = stats.norm.ppf((ranks - 0.375) / (len(values) + 0.25))
    return out


def _standardise(s: pd.Series, transform: str | None) -> pd.Series:
    """Standardise one variable, after an optional rank-based transformation."""
    if transform == "rank_normal":
        s = rank_inverse_normal(s)
    elif transform not in (None, "none"):
        raise ValueError(f"Unknown transform: {transform}")
    return _zscore(s)


def _bare(term: str) -> str:
    """Column name behind a formula term, e.g. 'C(pam50)' -> 'pam50'."""
    return term.removeprefix("C(").removesuffix(")")


# ---------------------------------------------------------------------------
# Vectorised association engine (screen)
# ---------------------------------------------------------------------------
def _design(data: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    """Covariate design matrix with intercept; 'C(x)' terms are one-hot encoded."""
    parts = [np.ones((len(data), 1))]
    for term in covariates:
        col = _bare(term)
        if term.startswith("C("):
            dummies = pd.get_dummies(data[col].astype("category"), drop_first=True, dtype=float)
            parts.append(dummies.to_numpy())
        else:
            parts.append(data[[col]].to_numpy(dtype=float))
    return np.hstack(parts)


def _residualise(M: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Residuals of each column of M after least-squares projection on Z."""
    coef, *_ = np.linalg.lstsq(Z, M, rcond=None)
    return M - Z @ coef


def _ols_t(Rx: np.ndarray, RY: np.ndarray, df_resid: int):
    """Exposure coefficients and t-statistics from residualised exposure(s) and outcomes
    (Frisch-Waugh-Lovell). Rx: (n, B), RY: (n, p). Returns arrays of shape (B, p)."""
    sxx = (Rx ** 2).sum(axis=0)[:, None]                    # (B, 1)
    beta = (Rx.T @ RY) / sxx                                 # (B, p)
    sse = (RY ** 2).sum(axis=0)[None, :] - beta ** 2 * sxx   # (B, p)
    se = np.sqrt(np.clip(sse, 1e-12, None) / df_resid / sxx)
    return beta, beta / se


def _prepare_target(data: pd.DataFrame, target: MolecularTarget, features: list[str],
                    covariates: list[str], transform: str | None = None,
                    exposure_transform: str | None = None):
    """Analysable cohort for one target: complete exposure and covariates, image
    features standardised within the cohort and missing values set to the mean
    (zero after standardisation). Missingness is capped upstream by
    `select_image_features`, so the imputation only shrinks effects slightly."""
    cohort = data.query(target.cohort) if target.cohort else data
    covs = list(covariates) + [f"C({c})" for c in target.extra_covariates]
    needed = [target.variable] + [_bare(c) for c in covs]
    cohort = cohort.dropna(subset=needed)
    Y = cohort[features].apply(_standardise, transform=transform)
    Y = Y.fillna(0.0)
    x = cohort[target.variable].astype(float)
    if target.kind == "continuous":
        x = _standardise(x, exposure_transform)
    Z = _design(cohort, covs)
    strata = cohort[_bare(covariates[0])].to_numpy() if covariates else np.zeros(len(cohort))
    return x.to_numpy(), Y.to_numpy(), Z, strata


def footprint_screen(data: pd.DataFrame, targets: list[MolecularTarget], features: list[str],
                     covariates: list[str], transform: str | None = None,
                     exposure_transform: str | None = None) -> pd.DataFrame:
    """Adjusted association of every target with every image feature.

    Effects are in SD of the feature (per SD of the exposure for continuous
    targets). FDR is controlled within each target and across the full screen.
    With transform="rank_normal", image features are rank-transformed to a
    normal distribution before standardisation; `exposure_transform` does the
    same for continuous genomic variables.
    """
    rows = []
    for t in targets:
        x, Y, Z, _ = _prepare_target(data, t, features, covariates, transform, exposure_transform)
        df_resid = len(x) - Z.shape[1] - 1
        Rx = _residualise(x[:, None], Z)
        RY = _residualise(Y, Z)
        beta, tstat = _ols_t(Rx, RY, df_resid)
        p = 2 * stats.t.sf(np.abs(tstat[0]), df_resid)
        n_alt = int(x.sum()) if t.kind == "binary" else np.nan
        rows.append(pd.DataFrame({
            "target": t.name, "feature": features, "effect": beta[0], "t": tstat[0], "p": p,
            "n": len(x), "n_altered": n_alt,
        }))
    out = pd.concat(rows, ignore_index=True)
    out["q_within"] = out.groupby("target")["p"].transform(lambda s: multipletests(s, method="fdr_bh")[1])
    out["q_global"] = multipletests(out["p"], method="fdr_bh")[1]
    return out


def footprints_by_adjustment(data: pd.DataFrame, targets: list[MolecularTarget], features: list[str],
                             adjustment_sets: dict[str, list[str]], alpha: float = 0.05,
                             transform: str | None = None, exposure_transform: str | None = None):
    """Run the pairwise screen under several covariate sets and compare them.

    Typical use contrasts a purity-only model (what the image reveals about a
    genomic variable, including through subtype) with a subtype-adjusted model
    (what the variable adds beyond subtype). Returns the long table of all
    screens and a per-target summary with the number of significant features
    (global FDR within each screen) under each adjustment and the share of the
    least-adjusted footprint retained by each more adjusted one.
    """
    screens = []
    for label, covs in adjustment_sets.items():
        sc = footprint_screen(data, targets, features, covs, transform, exposure_transform)
        sc.insert(0, "adjustment", label)
        screens.append(sc)
    long = pd.concat(screens, ignore_index=True)
    labels = list(adjustment_sets)
    summary = (long.assign(sig=long["q_global"] < alpha)
               .pivot_table(index="target", columns="adjustment", values="sig", aggfunc="sum")
               [labels].astype(int))
    reference = summary[labels[0]].replace(0, np.nan)
    for label in labels[1:]:
        summary[f"retained_{label}"] = (summary[label] / reference).round(2)
    summary = summary.sort_values(labels[0], ascending=False)
    return long, summary


def _permute_within(x: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    xp = x.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        xp[idx] = x[rng.permutation(idx)]
    return xp


def omnibus_permutation_test(data: pd.DataFrame, targets: list[MolecularTarget], features: list[str],
                             covariates: list[str], n_perm: int = 1000, seed: int = 0,
                             transform: str | None = None, exposure_transform: str | None = None) -> pd.DataFrame:
    """Does each target change the image as a whole, beyond subtype and purity?

    Two image-wide statistics are computed from the per-feature t-statistics:
    the mean squared t, sensitive to diffuse footprints spread over many
    correlated features (e.g. a global change in immune infiltration), and the
    maximum squared t, sensitive to sparse footprints concentrated in a few
    features (e.g. a loss of cohesion). Null distributions are obtained by
    permuting the target within strata of the first covariate (subtype), which
    keeps the subtype composition and the correlation among image features
    intact. The two permutation p-values are combined with a Bonferroni
    correction, and FDR is controlled across targets on the combined p-value.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t in targets:
        x, Y, Z, strata = _prepare_target(data, t, features, covariates, transform, exposure_transform)
        df_resid = len(x) - Z.shape[1] - 1
        RY = _residualise(Y, Z)
        _, t_obs = _ols_t(_residualise(x[:, None], Z), RY, df_resid)
        X_perm = np.column_stack([_permute_within(x, strata, rng) for _ in range(n_perm)])
        _, t_perm = _ols_t(_residualise(X_perm, Z), RY, df_resid)
        row = {"target": t.name, "kind": t.kind, "n": len(x),
               "n_altered": int(x.sum()) if t.kind == "binary" else np.nan}
        for label, agg in (("mean_t2", np.mean), ("max_t2", np.max)):
            obs = float(agg(t_obs ** 2))
            null = agg(t_perm ** 2, axis=1)
            row[label] = obs
            row[f"{label}_enrichment"] = obs / float(np.median(null))
            row[f"p_{label}"] = (1 + np.sum(null >= obs)) / (1 + n_perm)
        row["top_feature"] = features[int(np.argmax(t_obs[0] ** 2))]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_perm"] = np.minimum(1.0, 2 * out[["p_mean_t2", "p_max_t2"]].min(axis=1))
    out["q_perm"] = multipletests(out["p_perm"], method="fdr_bh")[1]
    return out.sort_values("p_perm", ignore_index=True)


# ---------------------------------------------------------------------------
# Hypothesis-driven footprints (exact OLS, complete cases per feature)
# ---------------------------------------------------------------------------
def fit_footprint(data: pd.DataFrame, feature: str, target: MolecularTarget,
                  covariates: list[str], transform: str | None = None,
                  exposure_transform: str | None = None) -> dict:
    """OLS of one standardised image feature on one target plus covariates."""
    cols = list(dict.fromkeys([feature, target.variable] + [_bare(c) for c in covariates]))
    sub = data[cols].dropna().copy()
    sub["_y"] = _standardise(sub[feature], transform)
    sub["_x"] = (_standardise(sub[target.variable], exposure_transform) if target.kind == "continuous"
                 else sub[target.variable])
    fit = smf.ols("_y ~ _x" + "".join(f" + {c}" for c in covariates), data=sub).fit()
    lo, hi = fit.conf_int().loc["_x"]
    return {"effect": fit.params["_x"], "ci_low": lo, "ci_high": hi, "p": fit.pvalues["_x"],
            "n": int(fit.nobs),
            "n_altered": int(sub["_x"].sum()) if target.kind == "binary" else np.nan}


def tissue_footprint(data: pd.DataFrame, target: MolecularTarget, features: list[str],
                     feature_family: dict[str, str], covariates: list[str],
                     transform: str | None = None, exposure_transform: str | None = None) -> pd.DataFrame:
    """Footprint of one target on a pre-specified list of features.

    Reports adjusted and unadjusted effects, whether the adjusted direction
    matches the pre-specified hypothesis, and the attenuation, i.e. the share
    of the crude effect absorbed by subtype, purity and target-specific covariates.
    """
    cohort = data.query(target.cohort) if target.cohort else data
    adj_covs = list(covariates) + [f"C({c})" for c in target.extra_covariates]
    rows = []
    for feat in features:
        adj = fit_footprint(cohort, feat, target, adj_covs, transform, exposure_transform)
        crude = fit_footprint(cohort, feat, target, [], transform, exposure_transform)
        family = feature_family.get(feat)
        expected = target.expected.get(family)
        rows.append({
            "target": target.name, "role": target.role, "feature": feat, "family": family, **adj,
            "effect_unadjusted": crude["effect"], "p_unadjusted": crude["p"], "expected": expected,
            "matches_expected": (np.sign(adj["effect"]) == (1 if expected == "up" else -1))
                                if expected else np.nan,
        })
    out = pd.DataFrame(rows)
    crude_eff = out["effect_unadjusted"]
    out["attenuation"] = np.where(crude_eff.abs() >= 0.1, 1 - out["effect"] / crude_eff, np.nan)
    out["q_within"] = multipletests(out["p"], method="fdr_bh")[1]
    return out


def footprint_map(data: pd.DataFrame, targets: list[MolecularTarget], features: list[str],
                  feature_family: dict[str, str], covariates: list[str],
                  transform: str | None = None, exposure_transform: str | None = None) -> pd.DataFrame:
    """Hypothesis-driven footprints of several targets, with FDR within each target and globally."""
    fp = pd.concat([tissue_footprint(data, t, features, feature_family, covariates, transform, exposure_transform)
                    for t in targets],
                   ignore_index=True)
    fp["q_global"] = multipletests(fp["p"], method="fdr_bh")[1]
    return fp


def summarise_footprints(fp: pd.DataFrame, targets: list[MolecularTarget],
                         q_col: str = "q_within", alpha: float = 0.05) -> pd.DataFrame:
    """One row per target: strength, specificity and hypothesis support of its footprint."""
    rows = []
    for t in targets:
        sub = fp[fp["target"] == t.name]
        top = sub.loc[sub["effect"].abs().idxmax()]
        hyp = sub.dropna(subset=["expected"])
        rows.append({
            "target": t.name, "role": t.role, "n": int(sub["n"].max()), "n_altered": sub["n_altered"].max(),
            "n_features_significant": int((sub[q_col] < alpha).sum()),
            "strongest_feature": top["feature"], "strongest_family": top["family"],
            "strongest_effect_sd": round(top["effect"], 2),
            "hypotheses_tested": len(hyp),
            "hypotheses_supported": int(((hyp[q_col] < alpha) & (hyp["matches_expected"] == True)).sum()),
            "median_attenuation": round(sub["attenuation"].median(), 2),
            "use_case": t.use_case,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One genomic variable across adjustment levels
# ---------------------------------------------------------------------------
def effects_across_adjustments(screens_long: pd.DataFrame, target: str, adjustments: list[str],
                               select_in: str | None = None, alpha: float = 0.05,
                               top_n: int | None = 15) -> pd.DataFrame:
    """Effects of selected image features on one genomic variable under every adjustment.

    Features are those significant under `select_in` (default: the strictest
    adjustment), ranked by the absolute effect there and limited to `top_n`.
    Comparing the columns shows whether a feature's effect is carried by
    subtype or histology (it shrinks) or belongs to the alteration (it stays).
    """
    select_in = select_in or adjustments[-1]
    d = screens_long[screens_long["target"] == target]
    chosen = d[(d["adjustment"] == select_in) & (d["q_global"] < alpha)]
    chosen = chosen.reindex(chosen["effect"].abs().sort_values(ascending=False).index)
    if top_n is not None:
        chosen = chosen.head(top_n)
    effects = (d[d["feature"].isin(chosen["feature"])]
               .pivot(index="feature", columns="adjustment", values="effect")
               .reindex(index=chosen["feature"], columns=adjustments))
    effects[f"q ({select_in})"] = chosen.set_index("feature")["q_global"]
    return effects


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_omnibus(omnibus: pd.DataFrame, q_col: str = "q_perm", alpha: float = 0.05,
                 title: str = "Genomic footprints in tissue architecture", label=None):
    """Image-wide association over the permutation null for each genomic target,
    for the diffuse (mean t²) and sparse (max t²) statistics. `label` maps a target
    name to the text shown on the axis (default: the name itself)."""
    d = omnibus.sort_values("mean_t2_enrichment")
    sig = (d[q_col] < alpha).to_numpy()
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9, 0.28 * len(d) + 1.8))
    for ax, col, panel in zip(axes, ["mean_t2_enrichment", "max_t2_enrichment"],
                              ["Diffuse footprint (mean t²)", "Sparse footprint (max t²)"]):
        ax.barh(y, d[col], color=np.where(sig, "#1f4e79", "#b0b0b0"))
        ax.axvline(1, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Observed / permutation median")
        ax.set_title(panel, fontsize=9)
    axes[0].set_yticks(y, [label(t) if label else t for t in d["target"]], fontsize=8)
    fig.suptitle(f"{title} (dark: q < {alpha})", fontsize=10)
    fig.tight_layout()
    return fig


def plot_footprints_by_adjustment(summary: pd.DataFrame, labels: list[str],
                                  title: str = "Image footprint under increasing adjustment"):
    """Number of significant image features per genomic target under each adjustment."""
    d = summary.sort_values(labels[0])
    y = np.arange(len(d))
    height = 0.8 / len(labels)
    colors = ["#b0b0b0", "#1f4e79", "#6a9fcb", "#c97b2b"]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.5))
    for i, label in enumerate(labels):
        ax.barh(y + (i - (len(labels) - 1) / 2) * height, d[label], height=height,
                color=colors[i % len(colors)], label=label)
    ax.set_yticks(y, d.index, fontsize=8)
    ax.set_xlabel("Image features with q < 0.05")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig



def plot_adjustment_bars(summary: pd.DataFrame, targets: list[str], labels: list[str], alpha: float = 0.05,
                         title: str = "", label=None):
    """Significant image features of selected targets under each adjustment, as grouped bars
    (first target at the top). `label` maps a target name to the text shown on the axis."""
    d = summary.loc[targets, labels].iloc[::-1]
    colors = ["#b0b0b0", "#1f4e79", "#6a9fcb", "#c97b2b"]
    height = 0.8 / len(labels)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(d) + 1.8), dpi=150)
    for i, adj in enumerate(labels):
        ax.barh(y + ((len(labels) - 1) / 2 - i) * height, d[adj], height=height, color=colors[i % len(colors)],
                label=adj)
    ax.set_yticks(y, [label(t) if label else t for t in d.index], fontsize=10)
    ax.set_xlabel(f"Image features with q < {alpha}", fontsize=10.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=32)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(labels), frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def plot_effects_across_adjustments(effects: pd.DataFrame, adjustments: list[str], title: str = ""):
    """Effect of each feature under each adjustment, one row per feature: points
    joined by a line show how much of the effect survives each adjustment."""
    d = effects.iloc[::-1]
    colors = ["#b0b0b0", "#1f4e79", "#6a9fcb", "#c97b2b"]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(d) + 1.6))
    y = np.arange(len(d))
    for i in y:
        ax.plot(d.iloc[i][adjustments].to_numpy(dtype=float), [i] * len(adjustments),
                color="#d0d0d0", lw=1, zorder=1)
    for k, adj in enumerate(adjustments):
        ax.scatter(d[adj], y, color=colors[k % len(colors)], s=28, label=adj, zorder=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, d.index, fontsize=6.5)
    ax.set_xlabel("Effect (SD of the image feature)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.6 / max(len(d), 1)),
              ncol=len(adjustments))
    fig.tight_layout()
    return fig


def plot_feature_by_group(data: pd.DataFrame, feature: str, group: str, order: list[str],
                          ylabel: str | None = None, title: str | None = None):
    """Distribution of one image feature across patient groups (box plot with points)."""
    groups = [data.loc[data[group] == g, feature].dropna().to_numpy() for g in order]
    fig, ax = plt.subplots(figsize=(1.6 * len(order) + 1.5, 4))
    ax.boxplot(groups, showfliers=False, widths=0.5, medianprops={"color": "#1f4e79", "lw": 2})
    rng = np.random.default_rng(0)
    for i, values in enumerate(groups, start=1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values, s=6, alpha=0.35, color="#555555")
    ax.set_xticks(range(1, len(order) + 1), [f"{g}\n(n = {len(v)})" for g, v in zip(order, groups)], fontsize=8)
    ax.set_ylabel(ylabel or feature, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_footprint_heatmap(fp: pd.DataFrame, target_order: list[str], feature_family: dict[str, str],
                           q_col: str = "q_within", alpha: float = 0.05,
                           title: str = "Tissue footprints (adjusted)"):
    """Heatmap of adjusted effects: targets x image features grouped by family."""
    feats = sorted(fp["feature"].unique(), key=lambda f: (feature_family.get(f, ""), f))
    eff = fp.pivot(index="target", columns="feature", values="effect").loc[target_order, feats]
    sig = fp.pivot(index="target", columns="feature", values=q_col).loc[target_order, feats] < alpha
    vmax = np.nanmax(np.abs(eff.values))
    fig, ax = plt.subplots(figsize=(0.5 * len(feats) + 3.5, 0.5 * len(target_order) + 2.5))
    im = ax.imshow(eff.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    yy, xx = np.where(sig.values)
    ax.scatter(xx, yy, s=14, c="black", label=f"q < {alpha}")
    ax.set_xticks(range(len(feats)), [f"{feature_family.get(f, '')} | {f}" for f in feats],
                  rotation=90, fontsize=7)
    ax.set_yticks(range(len(target_order)), target_order)
    fig.colorbar(im, ax=ax, label="Adjusted effect (SD)")
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.12), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
