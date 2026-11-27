"""Tests for opentme_tcga.genomic_prediction on small simulated data with a known answer."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from opentme_tcga.genomic_prediction import (
    bootstrap_ci,
    drop_redundant,
    edge_check,
    make_preprocessor,
    paired_bootstrap,
    parse_space,
    plot_pair_importance,
    plot_signed_importance,
    plot_stacked_importance,
    score_within_groups,
    select_equivalent,
    signed_importance,
    site_split,
    spearman_score,
)


def test_site_split_keeps_sites_whole():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"site": rng.choice([f"S{i}" for i in range(20)], 400)})
    test_sites = site_split(d, "site", test_size=0.2, seed=1)
    assert 0 < len(test_sites) < 20
    assert test_sites == site_split(d, "site", test_size=0.2, seed=1)      # reproducible
    train_sites = set(d["site"]) - test_sites
    assert not train_sites & test_sites


def test_drop_redundant_keeps_most_complete_of_each_group():
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    d = pd.DataFrame({"A_copy_missing": a, "A": a + rng.normal(scale=0.01, size=200), "B": rng.normal(size=200)})
    d.loc[:20, "A_copy_missing"] = np.nan
    kept = drop_redundant(d, ["A_copy_missing", "A", "B"], threshold=0.9)
    assert kept == ["A", "B"]


def test_preprocessor_handles_numeric_and_missing_categories():
    d = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "subtype": ["LumA", None, "Basal", "LumA"]})
    out = make_preprocessor(n_quantiles=4).fit_transform(d)
    assert out.shape == (4, 1 + 3)            # x + LumA, Basal, missing
    assert not np.isnan(np.asarray(out, dtype=float)).any()


def test_parse_space_and_edge_check():
    space = parse_space({"C": {"loguniform": [1e-4, 10]}, "depth": {"randint": [1, 7]},
                         "frac": {"uniform": [0.3, 1.0]}, "kernel": ["linear", "rbf"]}, prefix="model__")
    assert set(space) == {"model__C", "model__depth", "model__frac", "model__kernel"}
    assert space["model__kernel"] == ["linear", "rbf"]
    assert space["model__frac"].support() == (0.3, 1.0)
    edges = edge_check(space, {"model__C": 9.9, "model__depth": 3, "model__frac": 0.65, "model__kernel": "rbf"})
    assert edges.set_index("hyperparameter").loc["model__C", "at_edge"]
    assert not edges.set_index("hyperparameter").loc["model__frac", "at_edge"]
    with pytest.raises(ValueError):
        parse_space({"C": {"normal": [0, 1]}})


def test_select_equivalent_prefers_simplest_within_one_sd():
    folds = pd.DataFrame({"logistic": [0.80, 0.84, 0.82], "xgboost": [0.81, 0.86, 0.84],
                          "svm": [0.70, 0.72, 0.71]})
    selected, threshold, equivalent = select_equivalent(folds, ["logistic", "svm", "xgboost"])
    assert selected == "logistic" and "svm" not in equivalent and threshold < 0.8367


def test_spearman_score_constant_predictions():
    assert spearman_score([1, 2, 3], [5, 5, 5]) == 0.0
    assert spearman_score([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_bootstrap_intervals_contain_the_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    good = y + rng.normal(scale=0.8, size=300)
    noise = rng.normal(size=300)
    low, high = bootstrap_ci(y, good, roc_auc_score, n_boot=300, seed=0)
    assert low < roc_auc_score(y, good) < high
    res = paired_bootstrap(y, good, noise, roc_auc_score, n_boot=300, seed=0)
    assert res["diff_low"] > 0 and res["p_a_not_better"] < 0.05


def test_score_within_groups_skips_small_or_one_class_groups():
    y = np.array([0, 1] * 30 + [1] * 30)
    pred = y + 0.1
    groups = np.array(["A"] * 60 + ["B"] * 30)
    out = score_within_groups(y, pred, groups, roc_auc_score, ["A", "B", "C"], min_n=20, min_per_class=10)
    assert out["A"] == pytest.approx(1.0)
    assert np.isnan(out["B"]) and np.isnan(out["C"])


def test_importance_figures():
    imp = pd.DataFrame({"LumA": [0.3, 0.1, 0.2], "Basal": [0.1, 0.4, 0.05]}, index=["f1", "f2", "f3"])
    signed = imp * np.array([[1, -1], [-1, 1], [1, 1]])
    colors = {"LumA": "#1f77b4", "Basal": "#c0392b"}
    fig = plot_stacked_importance(imp, colors, top_n=2)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["f1", "f2"]
    assert len(plot_signed_importance(signed, top_n=2).axes) == 2
    assert len(plot_pair_importance(signed, ["LumA", "Basal"], colors, top_n=1).axes) == 1


def test_signed_importance_direction_without_coefficients():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"up": rng.normal(size=100), "down": rng.normal(size=100), "flat": np.zeros(100)})
    shap_values = np.column_stack([0.5 * X["up"], -0.2 * X["down"], np.zeros(100)])
    out = signed_importance(shap_values, X)
    assert out["up"] > 0 > out["down"] and out["flat"] == 0
    assert abs(out["up"]) > abs(out["down"])
