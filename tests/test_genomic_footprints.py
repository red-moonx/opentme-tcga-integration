"""Tests for opentme_tcga.genomic_footprints on simulated data with a known answer."""
import numpy as np
import pandas as pd
import pytest

from opentme_tcga.genomic_footprints import (
    MolecularTarget,
    fit_footprint,
    footprint_screen,
    omnibus_permutation_test,
    plot_adjustment_bars,
    resolve_terms,
    select_image_features,
    two_hit,
)

COVARIATES = ["C(pam50)", "abs_purity"]


@pytest.fixture
def cohort():
    """Simulated cohort: a subtype-linked mutation with one planted footprint and a null mutation."""
    rng = np.random.default_rng(1)
    n, p = 600, 40
    pam50 = rng.choice(["LumA", "LumB", "Basal", "Her2"], n)
    d = pd.DataFrame({"pam50": pam50, "abs_purity": rng.uniform(0.2, 0.9, n)})
    basal = (pam50 == "Basal").astype(float)
    d["mut_A"] = (rng.uniform(size=n) < 0.1 + 0.6 * basal).astype(float)  # enriched in basal tumours
    d["mut_null"] = (rng.uniform(size=n) < 0.2).astype(float)
    for j in range(p):
        d[f"FEAT_{j}"] = rng.normal(size=n) + 0.8 * basal                 # subtype affects all features
    d["FEAT_0"] += -0.7 * d["mut_A"]                                     # planted footprint beyond subtype
    d["COUNT_CELLS"] = rng.integers(100, 10_000, n)                      # absolute count, must be dropped
    d.loc[rng.choice(n, 200, replace=False), "FEAT_39"] = np.nan          # too many missing values
    d["FEAT_BINARY"] = (rng.uniform(size=n) < 0.4).astype(float)          # informative binary, must be kept
    d["FEAT_RARE"] = (rng.uniform(size=n) < 0.01).astype(float)           # near-constant, must be dropped
    return d


@pytest.fixture
def features(cohort):
    candidates = [c for c in cohort.columns if c.startswith(("FEAT_", "COUNT_"))]
    return select_image_features(cohort, candidates, drop_prefixes=["COUNT_"])


def test_feature_selection(features):
    assert "COUNT_CELLS" not in features and "FEAT_39" not in features and "FEAT_RARE" not in features
    assert "FEAT_BINARY" in features


def test_screen_recovers_planted_footprint_and_removes_subtype_confounding(cohort, features):
    target = MolecularTarget("mut_A", "mut_A")
    adjusted = footprint_screen(cohort, [target], features, COVARIATES)
    crude = footprint_screen(cohort, [target], features, [])
    # Without subtype adjustment the mutation looks associated with almost every feature;
    # with it, only the planted feature remains, with the right sign
    assert (crude["q_within"] < 0.05).sum() > 30
    assert set(adjusted.query("q_within < 0.05")["feature"]) == {"FEAT_0"}
    assert adjusted.set_index("feature").loc["FEAT_0", "effect"] < 0
    # The vectorised screen gives the same estimate as a standard OLS fit
    exact = fit_footprint(cohort, "FEAT_0", target, COVARIATES)
    assert adjusted.set_index("feature").loc["FEAT_0", "effect"] == pytest.approx(exact["effect"], rel=1e-6)


def test_omnibus_separates_signal_from_null(cohort, features):
    targets = [MolecularTarget("mut_A", "mut_A"), MolecularTarget("mut_null", "mut_null")]
    res = omnibus_permutation_test(cohort, targets, features, COVARIATES, n_perm=200, seed=0).set_index("target")
    assert res.loc["mut_A", "p_perm"] < 0.01
    assert res.loc["mut_null", "p_perm"] > 0.05


def test_rank_transformations_remove_outlier_driven_associations():
    """A few extreme slides (image side) or hypermutated tumours (genomic side) create
    spurious associations that the rank-based transformations remove; a real shift is kept."""
    rng = np.random.default_rng(7)
    n = 800
    d = pd.DataFrame({"pam50": rng.choice(["LumA", "Basal"], n), "abs_purity": rng.uniform(0.2, 0.9, n)})
    d["mut"] = (rng.uniform(size=n) < 0.2).astype(float)
    d["SPIKY"] = rng.exponential(size=n)
    d.loc[np.flatnonzero(d["mut"] == 1)[:4], "SPIKY"] = 200.0       # four extreme slides
    d["SHIFTED"] = rng.normal(size=n) + 0.5 * d["mut"]               # genuine shift
    d["tmb"] = rng.normal(size=n)
    d["FEAT"] = rng.normal(size=n)
    hyper = rng.choice(n, 5, replace=False)
    d.loc[hyper, "tmb"] = 40.0                                       # hypermutated tumours
    d.loc[hyper, "FEAT"] += 3.0

    binary = [MolecularTarget("mut", "mut")]
    raw = footprint_screen(d, binary, ["SPIKY", "SHIFTED"], COVARIATES).set_index("feature")
    rin = footprint_screen(d, binary, ["SPIKY", "SHIFTED"], COVARIATES, transform="rank_normal").set_index("feature")
    assert raw.loc["SPIKY", "p"] < 0.01 and rin.loc["SPIKY", "p"] > 0.05
    assert rin.loc["SHIFTED", "p"] < 0.001

    continuous = [MolecularTarget("tmb", "tmb", kind="continuous")]
    raw = footprint_screen(d, continuous, ["FEAT"], COVARIATES).iloc[0]
    rin = footprint_screen(d, continuous, ["FEAT"], COVARIATES, exposure_transform="rank_normal").iloc[0]
    assert raw["p"] < 0.001 and rin["p"] > 0.05


def test_two_hit_needs_both_events_and_both_known():
    mut = pd.Series([1, 1, 0, np.nan, 1.0])
    arm = pd.Series([1, 0, 1, 1, np.nan])
    out = two_hit(mut, arm)
    assert out.iloc[:3].tolist() == [1.0, 0.0, 0.0]
    assert out.iloc[3:].isna().all()


def test_resolve_terms_replaces_aliases_only():
    aliases = {"subtype": "C(clin_pam50)", "purity": "abs_purity"}
    assert resolve_terms(["subtype", "purity", "C(clin_histology)"], aliases) == \
        ["C(clin_pam50)", "abs_purity", "C(clin_histology)"]


def test_plot_adjustment_bars_keeps_target_order():
    import matplotlib
    matplotlib.use("Agg")
    summary = pd.DataFrame({"a": [5, 2, 0], "b": [3, 1, 0]}, index=["mut_X", "mut_Y", "mut_Z"])
    fig = plot_adjustment_bars(summary, ["mut_Y", "mut_X"], ["a", "b"], label=lambda t: t.upper())
    ticks = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    assert ticks == ["MUT_X", "MUT_Y"]        # first target drawn at the top


def test_plot_omnibus_uses_label_function():
    import matplotlib
    matplotlib.use("Agg")
    from opentme_tcga.genomic_footprints import plot_omnibus
    om = pd.DataFrame({"target": ["mut_A", "mut_B"], "mean_t2_enrichment": [2.0, 1.0],
                       "max_t2_enrichment": [3.0, 1.0], "q_perm": [0.01, 0.5]})
    fig = plot_omnibus(om, label=lambda t: t.upper())
    assert {tick.get_text() for tick in fig.axes[0].get_yticklabels()} == {"MUT_A", "MUT_B"}
