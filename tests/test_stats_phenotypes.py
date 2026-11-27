"""Tests for opentme_tcga.stats and opentme_tcga.phenotypes (simulated data with a known answer)."""
import numpy as np
import pandas as pd
import pytest

from opentme_tcga.phenotypes import add_densities_mm2, hot_flag, tls_class, infiltration_index, median_rule, spatial_phenotype
from opentme_tcga.stats import add_fdr, assign_family, fit_effect, fit_logit, spearman_ci, stars, zscore


@pytest.fixture
def groups():
    rng = np.random.default_rng(0)
    n = 400
    g = np.repeat([1.0, 0.0], n // 2)
    sub = rng.choice(["A", "B"], n)
    return pd.DataFrame({
        "group": g,
        "subtype": sub,
        "purity": rng.uniform(0.3, 0.9, n),
        "y": 0.8 * g + rng.normal(size=n),                      # true effect 0.8
        "null": rng.normal(size=n),
        "event": (rng.uniform(size=n) < np.where(g == 1, 0.6, 0.2)).astype(float),
    })


def test_fit_effect_recovers_effect(groups):
    r = fit_effect(groups, "y", "group", categorical=["subtype"], continuous=["purity"])
    assert r["ci_lower"] < 0.8 < r["ci_upper"] and r["p_value"] < 1e-6
    assert r["n_group1"] == r["n_group0"] == 200


def test_fit_effect_too_small_group(groups):
    small = pd.concat([groups[groups["group"] == 0], groups[groups["group"] == 1].head(3)])
    assert np.isnan(fit_effect(small, "y", "group")["effect"])


def test_fit_logit_positive_log2_or(groups):
    r = fit_logit(groups, "event", "group", categorical=["subtype"], continuous=["purity"])
    true_log2_or = np.log2((0.6 / 0.4) / (0.2 / 0.8))
    assert abs(r["effect"] - true_log2_or) < 0.8 and r["p_value"] < 1e-6
    assert r["ci_lower"] < r["effect"] < r["ci_upper"]


def test_fit_logit_drops_small_levels(groups):
    d = groups.copy()
    d.loc[d.index[:10], "subtype"] = "rare"
    assert fit_logit(d, "event", "group", categorical=["subtype"], min_level=30)["n"] == len(d) - 10


def test_add_fdr_by_group():
    df = pd.DataFrame({"p_value": [0.01, 0.04, 0.01, np.nan], "spec": ["a", "a", "b", "b"]})
    out = add_fdr(df, by="spec")
    assert out.loc[0, "fdr"] == pytest.approx(0.02) and out.loc[2, "fdr"] == pytest.approx(0.01)
    assert np.isnan(out.loc[3, "fdr"]) and "fdr" not in df


def test_small_helpers():
    assert zscore(pd.Series([1.0, 2, 3])).tolist() == [-1, 0, 1]
    assert stars(0.0005) == "***" and stars(0.02) == "*" and stars(0.2) == "" and stars(np.nan) == ""
    assert assign_family("mut_TP53") == "Driver mutations"
    assert assign_family("expr_COL10A1") == "Stroma & matrix"
    assert assign_family("sig_checkpoint_molecules") == "Brakes & myeloid"
    rho, low, high, n = spearman_ci(np.arange(50), np.arange(50) ** 2, n_boot=50)
    assert rho == pytest.approx(1) and n == 50 and low <= rho <= high


def test_hot_and_spatial_phenotype():
    lf = pd.Series([0.1, 0.2, 0.3, 0.4, np.nan])
    hot, thr = hot_flag(lf, 0.5)
    assert thr == pytest.approx(0.25) and hot.tolist()[:4] == [0, 0, 1, 1] and np.isnan(hot.iloc[4])

    core = pd.Series([100.0, 10, 100, 1, 5])
    margin = pd.Series([10.0, 100, 1, 100, 5])
    idx = infiltration_index(core, margin, eps=0)
    assert idx.tolist()[:2] == pytest.approx([1, -1])
    pheno = spatial_phenotype(hot, idx)
    assert pheno.tolist() == ["Cold", "Cold", "Inflamed", "Excluded", None]


def test_infiltration_index_pseudocount():
    idx = infiltration_index(pd.Series([0.0]), pd.Series([0.0]), eps=0.01)
    assert idx.iloc[0] == 0


def test_median_rule():
    inside = pd.Series([10.0, 1, 1, 10, np.nan])
    outside = pd.Series([1.0, 10, 1, 10, 5])
    # medians: inside 5.5, outside 7.5
    assert median_rule(inside, outside).tolist()[:4] == ["inflamed", "excluded", "desert", "inflamed"]
    assert pd.isna(median_rule(inside, outside).iloc[4])


def test_add_densities_mm2():
    df = pd.DataFrame({"CELL_DENSITY_A": [1e-4], "CELL_DENSITY_B": [100.0], "CELL_PERCENTAGE_C": [5.0]})
    image = {"a": {"feature": "CELL_DENSITY_A", "table": "tme"}, "b": {"feature": "CELL_DENSITY_B", "table": "wtr_cell"},
             "c": {"feature": "CELL_PERCENTAGE_C", "table": "tme"}}
    out = add_densities_mm2(df, image, {"tme": 1e6, "wtr_cell": 1.0})
    assert out["CELL_DENSITY_A_MM2"].iloc[0] == pytest.approx(100) and out["CELL_DENSITY_B_MM2"].iloc[0] == 100
    assert "CELL_PERCENTAGE_C_MM2" not in out


def test_tls_class():
    present = pd.Series([0.0, 1, 1, np.nan])
    mature = pd.Series([0.0, 0, 1, np.nan])
    assert tls_class(present, mature).tolist() == ["No TLS", "Immature only", "Mature", None]
