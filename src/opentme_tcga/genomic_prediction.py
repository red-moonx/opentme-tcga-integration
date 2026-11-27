"""Prediction of molecular subtype and genomic alterations from interpretable H&E features.

Building blocks of notebook 07:

* evaluation design: held-out tissue source sites (`site_split`), removal of
  near-duplicate image features (`drop_redundant`) and a preprocessing step
  learned inside each training fold (`make_preprocessor`);
* model choice: hyperparameter search spaces read from the configuration
  (`parse_space`), the position of each best value within its range
  (`edge_check`) and the rule that picks the simplest of the models whose
  performance is equivalent to the best one (`select_equivalent`);
* evaluation: Spearman scorer for continuous targets, bootstrap confidence
  intervals (`bootstrap_ci`, `paired_bootstrap`) and performance within groups
  such as subtypes (`score_within_groups`);
* figures of feature importance (`plot_stacked_importance`, `plot_signed_importance`,
  `plot_pair_importance`) and of fold-wise performance (`plot_folds`).

Nothing here is specific to a cancer type or a target.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


# ---------------------------------------------------------------------------
# Evaluation design
# ---------------------------------------------------------------------------
def site_split(data: pd.DataFrame, group_column: str, test_size: float, seed: int) -> set:
    """Tissue source sites held out for the final test: all patients of a site go either to
    training or to test. Returns the set of test sites."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_idx = next(splitter.split(data, groups=data[group_column]))
    return set(data[group_column].iloc[test_idx])


def drop_redundant(data: pd.DataFrame, features: list[str], threshold: float) -> list[str]:
    """Remove near-duplicate features without looking at any target.

    Features are visited from fewest to most missing values (ties keep the given order); a
    feature is kept only if its absolute Spearman correlation with every feature already kept
    is at most `threshold`, so from each group of redundant features the most complete one stays.
    """
    corr = data[features].corr(method="spearman").abs()
    ordered = data[features].isna().mean().sort_values(kind="stable").index
    kept: list[str] = []
    for col in ordered:
        if not (corr.loc[col, kept] > threshold).any():
            kept.append(col)
    return kept


def make_preprocessor(n_quantiles: int = 500) -> ColumnTransformer:
    """Preprocessing learned on the training data of each fold.

    Numeric columns: median imputation and a rank-based mapping to a normal distribution.
    Other columns: missing values become their own category, then one-hot encoding.
    Columns are assigned by data type, so the same preprocessor serves every feature set.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rank_normal", QuantileTransformer(output_distribution="normal", n_quantiles=n_quantiles)),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include="number")),
        ("categorical", categorical, make_column_selector(dtype_exclude="number")),
    ])


def feature_names_out(pipeline: Pipeline, step: str = "preprocess") -> list[str]:
    """Column names after the preprocessing step, without the 'numeric__' / 'categorical__' prefix."""
    return [name.split("__", 1)[1] for name in pipeline.named_steps[step].get_feature_names_out()]


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------
_DISTRIBUTIONS = {"loguniform", "uniform", "randint"}


def parse_space(spec: dict, prefix: str = "") -> dict:
    """Search space from the configuration.

    Each hyperparameter is either a list of values, or a one-key mapping naming a scipy
    distribution with its range: ``{loguniform: [low, high]}``, ``{uniform: [low, high]}`` or
    ``{randint: [low, high]}`` (high excluded, as in scipy). `prefix` is added to every name
    (e.g. 'regressor__' when the pipeline is wrapped in a TransformedTargetRegressor).
    """
    space = {}
    for name, value in spec.items():
        if isinstance(value, dict):
            (kind, (low, high)), = value.items()
            if kind not in _DISTRIBUTIONS:
                raise ValueError(f"Unknown distribution for {name}: {kind}")
            if kind == "loguniform":
                dist = stats.loguniform(float(low), float(high))
            elif kind == "uniform":
                dist = stats.uniform(loc=float(low), scale=float(high) - float(low))
            else:
                dist = stats.randint(int(low), int(high))
        else:
            dist = list(value)
        space[f"{prefix}{name}"] = dist
    return space


def edge_check(space: dict, best_params: dict, margin: float = 0.05) -> pd.DataFrame:
    """Position of each best continuous hyperparameter within its range (0 = lower end, 1 = upper end,
    log scale for log-uniform ranges). A value at the edge suggests the range should be widened."""
    rows = []
    for param, dist in space.items():
        if isinstance(dist, list):
            continue
        low, high = dist.support()
        value = best_params[param]
        if dist.dist.name == "loguniform":
            position = (np.log(value) - np.log(low)) / (np.log(high) - np.log(low))
        else:
            position = (value - low) / (high - low) if high > low else 0.5
        rows.append({"hyperparameter": param, "best_value": value, "range": f"[{low:g}, {high:g}]",
                     "position": position, "at_edge": position < margin or position > 1 - margin})
    return pd.DataFrame(rows, columns=["hyperparameter", "best_value", "range", "position", "at_edge"])


def fold_scores(search, metric: str | None = None) -> list[float]:
    """Validation score of the best configuration of a fitted search, fold by fold."""
    key = f"_test_{metric}" if metric else "_test_score"
    n = search.n_splits_
    return [search.cv_results_[f"split{k}{key}"][search.best_index_] for k in range(n)]


def select_equivalent(folds: pd.DataFrame, order: list[str]) -> tuple[str, float, list[str]]:
    """Simplest model among those equivalent to the best one.

    `folds` has one column per model and one row per cross-validation fold (same folds for all
    models). Models whose mean score is at least the best mean minus the best model's standard
    deviation across folds are equivalent; the first of them in `order` (simplest first) is selected.
    Returns the selected model, the equivalence threshold and the equivalent models.
    """
    mean = folds.mean()
    best = mean.idxmax()
    threshold = mean[best] - folds[best].std(ddof=0)
    equivalent = [m for m in order if m in mean and mean[m] >= threshold]
    return equivalent[0], float(threshold), equivalent


# ---------------------------------------------------------------------------
# Metrics and uncertainty
# ---------------------------------------------------------------------------
def spearman_score(y_true, y_pred) -> float:
    """Spearman correlation; 0 if predictions are constant (no ranking information)."""
    if np.ptp(np.asarray(y_pred)) == 0:
        return 0.0
    return float(spearmanr(y_true, y_pred)[0])


spearman_scorer = make_scorer(spearman_score)


def bootstrap_ci(y, pred, metric, n_boot: int = 1000, seed: int = 0, n_classes: int = 2,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of `metric(y, pred)` over patients.

    Resamples that do not contain all `n_classes` classes are skipped (set n_classes=1 for
    continuous targets).
    """
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if n_classes > 1 and len(np.unique(y[idx])) < n_classes:
            continue
        draws.append(metric(y[idx], pred[idx]))
    alpha = (1 - level) / 2 * 100
    low, high = np.percentile(draws, [alpha, 100 - alpha])
    return float(low), float(high)


def paired_bootstrap(y, pred_a, pred_b, metric, n_boot: int = 1000, seed: int = 0, n_classes: int = 2,
                     level: float = 0.95) -> dict:
    """Paired bootstrap of two models on the same patients: confidence intervals of each model and of
    the difference a - b, and the share of resamples in which a does not beat b (one-sided)."""
    y, pred_a, pred_b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if n_classes > 1 and len(np.unique(y[idx])) < n_classes:
            continue
        a, b = metric(y[idx], pred_a[idx]), metric(y[idx], pred_b[idx])
        draws.append((a, b, a - b))
    draws = np.array(draws)
    alpha = (1 - level) / 2 * 100
    low, high = np.percentile(draws, [alpha, 100 - alpha], axis=0)
    return {"a_low": low[0], "a_high": high[0], "b_low": low[1], "b_high": high[1],
            "diff_low": low[2], "diff_high": high[2], "p_a_not_better": float(np.mean(draws[:, 2] <= 0))}


def score_within_groups(y, pred, groups, metric, levels: list, min_n: int = 20,
                        min_per_class: int | None = None) -> dict:
    """`metric` computed separately within each level of `groups` (e.g. subtype).

    Levels with fewer than `min_n` patients, or (if `min_per_class` is given, for binary targets)
    fewer than `min_per_class` patients in either class, are reported as missing.
    """
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    out = {}
    for level in levels:
        mask = groups == level
        y_sub = y[mask]
        enough = mask.sum() >= min_n
        if min_per_class is not None:
            enough = enough and min(y_sub.sum(), len(y_sub) - y_sub.sum()) >= min_per_class
        out[level] = metric(y_sub, pred[mask]) if enough else np.nan
    return out



def signed_importance(shap_values, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| of each feature, signed by the direction of its effect: positive if higher values of the
    feature raise the prediction (Spearman correlation between the feature and its SHAP values), negative
    if they lower it. Works for any model, including trees, which have no coefficients."""
    values = np.asarray(shap_values)
    out = {}
    for k, col in enumerate(X.columns):
        rho = spearmanr(X[col], values[:, k])[0] if np.ptp(X[col]) > 0 and np.ptp(values[:, k]) > 0 else 0.0
        out[col] = np.abs(values[:, k]).mean() * (np.sign(rho) if np.isfinite(rho) else 0.0)
    return pd.Series(out)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_folds(folds: pd.DataFrame, ylabel: str, title: str):
    """Fold-wise score of each model: one grey line per fold and the mean in black."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for _, row in folds.iterrows():
        ax.plot(folds.columns, row.values, marker="o", color="grey", alpha=0.6)
    ax.plot(folds.columns, folds.mean().values, marker="o", color="black", linewidth=2, label="mean")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_stacked_importance(importance: pd.DataFrame, colors: dict, top_n: int = 15, title: str = ""):
    """Most important features (rows of `importance`), with bars stacked over classes (columns)."""
    top = importance.loc[importance.sum(axis=1).sort_values().index[-top_n:]]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(top) + 1.2))
    left = np.zeros(len(top))
    for cls in importance.columns:
        ax.barh(top.index, top[cls], left=left, color=colors.get(cls), label=cls)
        left += top[cls].values
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("mean |SHAP| (stacked over classes)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_signed_importance(signed: pd.DataFrame, top_n: int = 8, title: str = "", xlabel: str | None = None):
    """One panel per class (column of `signed`): the `top_n` features by mean |SHAP|, on the right if a
    higher value makes the class more likely and on the left if it makes it less likely."""
    classes = list(signed.columns)
    ncols = 2 if len(classes) > 1 else 1
    nrows = int(np.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 4.5 * nrows), squeeze=False)
    for ax, cls in zip(axes.ravel(), classes):
        values = signed[cls]
        top = values[values != 0].abs().sort_values().index[-top_n:]
        ax.barh(top, values[top], color=["#c0392b" if v > 0 else "#2471a3" for v in values[top]])
        ax.axvline(0, color="black", linewidth=0.8)
        limit = values[top].abs().max() * 1.1
        ax.set_xlim(-limit, limit)
        ax.set_title(cls)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel(xlabel or "mean |SHAP|  ·  left: higher value → less likely  |  right: higher value → more likely")
    for ax in axes.ravel()[len(classes):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_pair_importance(signed: pd.DataFrame, pair: list[str], colors: dict, top_n: int = 6, title: str = ""):
    """Two classes side by side: union of the `top_n` most important features of each, with the
    direction of their effect for both classes."""
    top = set()
    for cls in pair:
        top |= set(signed[cls].abs().sort_values().index[-top_n:])
    table = signed.loc[list(top), pair]
    table = table.loc[table.abs().sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(table) + 1.5))
    y = np.arange(len(table))
    for offset, cls in zip([0.2, -0.2], pair):
        ax.barh(y + offset, table[cls], height=0.4, color=colors.get(cls), label=cls)
    ax.set_yticks(y, table.index)
    ax.axvline(0, color="black", linewidth=0.8)
    limit = table.abs().values.max() * 1.1
    ax.set_xlim(-limit, limit)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("mean |SHAP|  ·  left: higher value → less likely  |  right: higher value → more likely")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
