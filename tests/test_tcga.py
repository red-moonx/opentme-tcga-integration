"""Tests for opentme_tcga.tcga on small synthetic tables."""
import numpy as np
import pandas as pd
import pytest

from opentme_tcga.tcga import (
    MolecularTable,
    arm_calls,
    arm_events,
    clinical_block,
    cooccurrence,
    correlated_pairs,
    coverage,
    infer_type,
    figure_label,
    make_label,
    map_aliases,
    median_mad_scale,
    mutation_frequencies,
    mutation_matrix,
    passes_filter,
    primary_one_per_patient,
    recode_columns,
    redacted_patients,
    select_redundant,
    simplify_stage,
    ssgsea_running_sum,
    thorsson_subtype,
    to_patient,
)


# ---------------------------------------------------------------------------
# Barcodes and cohort
# ---------------------------------------------------------------------------
def test_to_patient():
    assert to_patient(pd.Series([" tcga-a1-0001-01A-11D ", "TCGA-A1-0002"])).tolist() == [
        "TCGA-A1-0001", "TCGA-A1-0002"]


def test_primary_one_per_patient_prefers_vial_a():
    df = pd.DataFrame({"bc": ["TCGA-A1-0001-01B-01", "TCGA-A1-0001-01A-02", "TCGA-A1-0002-11A-01",
                              "TCGA-A1-0003-06A-01", "TCGA-A1-0003-01A-01"], "x": [1, 2, 3, 4, 5]})
    out, log = primary_one_per_patient(df, "bc")
    assert out.set_index("patient")["x"].to_dict() == {"TCGA-A1-0001": 2, "TCGA-A1-0003": 5}
    assert log["n_patients"] == 3 and log["n_primary_patients"] == 2 and log["n_extra_samples"] == 1
    assert log["sample_types"] == {"01": 3, "11": 1, "06": 1}


def test_coverage():
    ref = pd.Index(["P1", "P2", "P3", "P4"])
    cov = coverage(["P1", "P3", "P9", None], ref)
    assert cov["n"] == 2 and cov["pct"] == 50.0 and cov["missing"] == ["P2", "P4"]


def test_redacted_patients():
    cdr = pd.DataFrame({"bcr_patient_barcode": ["TCGA-A1-0001", "TCGA-A1-0002"], "Redaction": [np.nan, "Redacted"]})
    assert list(redacted_patients(cdr)) == ["TCGA-A1-0002"]


# ---------------------------------------------------------------------------
# Molecular table
# ---------------------------------------------------------------------------
def test_infer_type():
    assert infer_type(pd.Series([0, 1, np.nan])) == "binary"
    assert infer_type(pd.Series([True, False])) == "binary"
    assert infer_type(pd.Series([0.1, 2.5])) == "continuous"
    assert infer_type(pd.Series(["a", "b"])) == "categorical"


def test_molecular_table_left_join_and_rerun():
    table = MolecularTable(pd.Index(["P1", "P2", "P3"]))
    block = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]}, index=["P1", "P9"])   # P9 not in the cohort
    table.add(block, source="S", level="main", descriptions={"x": "an x"})
    assert table.data.shape == (3, 2)
    assert table.data["x"].tolist()[:1] == [1.0] and table.data["x"].isna().sum() == 2

    table.add(block.assign(x=[5.0, 6.0]), source="S2", level="main")             # re-run replaces
    assert table.data["x"].iloc[0] == 5.0 and table.data.shape == (3, 2)
    d = table.dictionary().set_index("column")
    assert d.loc["x", "source"] == "S2" and d.loc["x", "n_with_data"] == 1 and d.loc["y", "type"] == "categorical"


def test_molecular_table_rejects_duplicates():
    table = MolecularTable(pd.Index(["P1"]))
    with pytest.raises(ValueError):
        table.add(pd.DataFrame({"x": [1, 2]}, index=["P1", "P1"]), "S", "main")


# ---------------------------------------------------------------------------
# Clinical
# ---------------------------------------------------------------------------
def test_simplify_stage():
    s = pd.Series(["Stage IIA", "Stage I", "Stage X", "Stage IV", np.nan, "Stage IIIC"])
    out = simplify_stage(s)
    assert out.iloc[0] == "II" and pd.isna(out.iloc[2]) and pd.isna(out.iloc[4]) and out.iloc[5] == "III"


def test_clinical_block():
    cdr = pd.DataFrame({
        "bcr_patient_barcode": ["TCGA-A1-0001", "TCGA-B2-0002", "TCGA-B2-0003"],
        "histological_type": ["Infiltrating Ductal Carcinoma", "Metaplastic", np.nan],
        "ajcc_pathologic_tumor_stage": ["Stage IIB", "Stage X", "Stage I"],
        "OS": ["1", "0", "x"],
    })
    out = clinical_block(cdr, {"histological_type": "clin_histological_type",
                               "ajcc_pathologic_tumor_stage": "clin_stage", "OS": "clin_os"},
                         numeric=["clin_os"], histology_map={"Infiltrating Ductal Carcinoma": "ductal"})
    assert out.index.tolist() == ["TCGA-A1-0001", "TCGA-B2-0002", "TCGA-B2-0003"]
    assert out["clin_histology"].tolist()[:2] == ["ductal", "other"] and pd.isna(out["clin_histology"].iloc[2])
    assert out["clin_stage_simple"].iloc[0] == "II" and "clin_stage" not in out
    assert out["clin_tss"].tolist() == ["A1", "B2", "B2"]
    assert np.isnan(out["clin_os"].iloc[2])


def test_recode_columns():
    df = pd.DataFrame({"sub": ["LumA", "CLOW", "Basal"], "tn": ["Yes", "No", "Unknown"]})
    out = recode_columns(df, {
        "pam50": {"source": "sub", "recode": {"CLOW": "Basal"}},
        "tnbc": {"source": "tn", "recode": {"Yes": True, "No": False}, "keep_unmapped": False, "dtype": "boolean"},
    })
    assert out["pam50"].tolist() == ["LumA", "Basal", "Basal"]
    assert out["tnbc"].tolist()[:2] == [True, False] and pd.isna(out["tnbc"].iloc[2])


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------
@pytest.fixture
def maf():
    return pd.DataFrame({
        "Tumor_Sample_Barcode": ["TCGA-A1-0001-01A", "TCGA-A1-0001-01A", "TCGA-A1-0002-01A", "TCGA-A1-0002-01A"],
        "Hugo_Symbol": ["TP53", "TP53", "PIK3CA", "TP53"],
        "FILTER": ["PASS", "wga", "PASS,oxog", "native_wga_mix,wga"],
    })


def test_passes_filter(maf):
    assert passes_filter(maf["FILTER"], {"PASS", "wga", "native_wga_mix"}).tolist() == [True, True, False, True]


def test_mutation_matrix_keeps_profiled_wild_type(maf):
    profiled = pd.Index(["TCGA-A1-0001", "TCGA-A1-0002", "TCGA-A1-0003"])     # 0003 profiled, no mutation
    m = mutation_matrix(maf, profiled)
    assert m.loc["TCGA-A1-0001", "TP53"] == 1 and m.loc["TCGA-A1-0001", "PIK3CA"] == 0
    assert (m.loc["TCGA-A1-0003"] == 0).all()
    freq = mutation_frequencies(m, ["TP53", "PIK3CA", "CDH1"], min_freq=0.5)
    assert freq.loc["TP53", "n_mutated"] == 2 and freq.loc["TP53", "kept"]
    assert not freq.loc["PIK3CA", "kept"] and freq.loc["CDH1", "n_mutated"] == 0


def test_cooccurrence_detects_exclusivity():
    a = np.array([1] * 50 + [0] * 50)
    df = pd.DataFrame({"A": a, "B": 1 - a, "C": np.tile([0, 1], 50)})
    pairs = cooccurrence(df).set_index(["gene_a", "gene_b"])
    assert pairs.loc[("A", "B"), "pattern"] == "exclusive"
    assert pairs.loc[("A", "C"), "pattern"] == "-"


# ---------------------------------------------------------------------------
# Arm-level events
# ---------------------------------------------------------------------------
def test_arm_calls_and_events():
    raw = pd.DataFrame({"NAME": ["1q", "16q"],
                        "TCGA-A1-0001-01": ["Gain", "Loss"], "TCGA-A1-0002-01": ["Gain", "Unchanged"],
                        "TCGA-A1-0003-01": ["Unchanged", np.nan], "TCGA-A1-0004-01": ["Gain", "Unchanged"]})
    calls = arm_calls(raw, "NAME", {"Gain": 1, "Loss": -1, "Unchanged": 0})
    assert calls.set_index("sample").loc["TCGA-A1-0001-01", "16q"] == -1
    arms = calls.set_index("sample")
    events, freq = arm_events(arms, min_freq=0.5, required=["16q_loss"])
    assert list(events.columns) == ["aneu_1q_gain", "aneu_16q_loss"]                # 16q loss forced in
    assert events["aneu_1q_gain"].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert np.isnan(events.loc["TCGA-A1-0003-01", "aneu_16q_loss"])                  # not called stays NaN


# ---------------------------------------------------------------------------
# Signatures and redundancy
# ---------------------------------------------------------------------------
def test_map_aliases_and_median_mad():
    assert map_aliases({"s": ["CXCL8", "CD3E"]}, {"CXCL8": "IL8"}) == {"s": ["IL8", "CD3E"]}
    scaled = median_mad_scale(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0]}))
    assert scaled["a"].median() == 0 and (scaled["a"] - scaled["a"].median()).abs().median() == pytest.approx(1)


def test_correlated_pairs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"a": x, "b": x * 2 + 0.01 * rng.normal(size=200), "c": rng.normal(size=200), "d": -x})
    pairs = correlated_pairs(df, ["a", "b", "c", "d"], 0.9)
    assert {frozenset(p) for p in zip(pairs["var1"], pairs["var2"])} == {
        frozenset("ab"), frozenset("ad"), frozenset("bd")}


@pytest.fixture
def redundant():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=300), rng.normal(size=300)
    df = pd.DataFrame({
        "few": x + 0.05 * rng.normal(size=300),      # same signal as "many", fewer patients
        "many": x,
        "hub": y,                                    # correlated with both "leaf" and "other"
        "leaf": y + 0.4 * rng.normal(size=300),
        "other": y + 0.4 * rng.normal(size=300),
        "free": rng.normal(size=300),
    })
    df.loc[:19, "few"] = np.nan
    return df


def test_select_redundant_prefers_more_patients(redundant):
    drop, log = select_redundant(redundant, list(redundant), 0.9)
    assert drop["few"] == "many"
    assert "fewer patients (280 vs 300)" in log.set_index("decision").loc["drop few", "reason"]


def test_select_redundant_tie_drops_more_redundant(redundant):
    drop, log = select_redundant(redundant, list(redundant), 0.9)
    assert "hub" in drop                       # highest mean |rho| of the y-cluster
    assert "leaf" not in drop and "other" not in drop
    assert (log["decision"] == "resolved").any()
    assert "free" not in drop and "many" not in drop


def test_select_redundant_respects_keep_pairs(redundant):
    drop, log = select_redundant(redundant, list(redundant), 0.9, keep_pairs=[["many", "few"]])
    assert "few" not in drop and "many" not in drop
    assert (log["decision"] == "kept").sum() == 1


def test_select_redundant_never_drops_protected(redundant):
    drop, log = select_redundant(redundant, list(redundant), 0.9, protected=["few", "hub"])
    assert drop["many"] == "few"               # protected wins over more patients
    assert "hub" not in drop
    drop, log = select_redundant(redundant, list(redundant), 0.9, protected=["few", "many"])
    assert "few" not in drop and "many" not in drop
    assert "both pre-specified (protected)" in log["reason"].tolist()


def test_figure_label():
    labels = {"cib_t_cells_cd8": "CD8 T cells (CIBERSORT)"}
    assert figure_label("cib_t_cells_cd8", labels) == "CD8 T cells"
    assert figure_label("mut_TP53", labels) == "TP53 mutation"
    assert figure_label("unknown_col", labels) == "unknown_col"


def test_ssgsea_running_sum_is_higher_when_signature_is_on_top():
    genes = [f"G{i}" for i in range(100)]
    top = pd.Series(np.arange(100, 0, -1.0), index=genes)          # G0 most expressed
    bottom = top[::-1].set_axis(genes)                              # G0 least expressed
    _, hits, running, es_top = ssgsea_running_sum(top, ["G0", "G1", "G2"])
    _, _, _, es_bottom = ssgsea_running_sum(bottom, ["G0", "G1", "G2"])
    assert hits.sum() == 3 and es_top > 0 > es_bottom
    assert running[-1] == pytest.approx(0)                          # the walk ends at 0
