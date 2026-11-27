"""Tests for opentme_tcga.opentme on small synthetic tables."""
import numpy as np
import pandas as pd
import pytest

from opentme_tcga.opentme import (
    match_thumbnail,
    ID_COLS,
    TABLE_GROUPS,
    add_derived_features,
    apply_exclusions,
    build_tme_exclusions,
    check_same_slides,
    density_per_mm2,
    list_tables,
    load_indication_tables,
    match_pattern,
    merge_slide_tables,
    non_constant,
    one_slide_per_patient,
    parse_table_path,
    patient_id,
    presence_flag,
    safe_ratio,
    sample_type,
    select_tls_features,
    select_wtr_features,
    slide_qc,
    spearman_with,
    zero_or_missing,
)

BARCODES = [
    "TCGA-A1-0001-01Z-00-DX1.svs",
    "TCGA-A1-0001-01Z-00-DX2.svs",
    "TCGA-A1-0002-06Z-00-DX1.svs",
    "TCGA-A1-0003-01Z-00-DX1.svs",
]


def _ids(barcodes=BARCODES):
    return pd.DataFrame({
        "TCGA_FILE_NAME": barcodes,
        "TCGA_SLIDE_UUID": [f"uuid{i}" for i in range(len(barcodes))],
        "TCGA_CASE_ID": [b[:12] for b in barcodes],
        "TCGA_PROJECT_ID": "TCGA-XX",
        "INDICATION": "test",
    })


@pytest.fixture
def tables():
    """Five slide-level tables sharing the same slides, with one name collision."""
    tme = _ids().assign(ABSOLUTE_AREA=[10.0, 20, 30, 40], CELL_DENSITY_LYMPHOCYTE=[1.0, 2, 3, 4])
    wtr_tissue = _ids().assign(ABSOLUTE_AREA=[1.0, 2, 3, 4])
    wtr_cell = _ids().assign(CELL_DENSITY_LYMPHOCYTE_IN_TUMOR_CORE=[5.0, 6, 7, 8])
    wtr_neighborhood = _ids().assign(NEIGHBOR_X=[0.0, 1, 0, 1])
    tls = _ids().assign(DENSITY_TLS_MATURE=[np.nan, 1, 0, np.nan])
    return {"tme": tme, "wtr_cell": wtr_cell, "wtr_tissue": wtr_tissue,
            "wtr_neighborhood": wtr_neighborhood, "tls": tls}


# ---------------------------------------------------------------------------
# File names and loading
# ---------------------------------------------------------------------------
def test_parse_table_path():
    assert parse_table_path("tme_features_breast_cancer_RUO.csv") == {"group": "tme", "cancer": "breast_cancer"}
    assert parse_table_path("whole_tumor_region_cell_features_lung_cancer_RUO.csv") == {
        "group": "wtr_cell", "cancer": "lung_cancer"}
    with pytest.raises(ValueError):
        parse_table_path("thumbnails_breast_cancer.csv")


def test_load_indication_tables(tmp_path, tables):
    folder = tmp_path / "data" / "breast_cancer"
    folder.mkdir(parents=True)
    names = {"tme": "tme_features", "wtr_cell": "whole_tumor_region_cell_features",
             "wtr_tissue": "whole_tumor_region_tissue_features",
             "wtr_neighborhood": "whole_tumor_region_neighborhood_features",
             "tls": "tertiary_lymphoid_structures_tls_features"}
    for group, df in tables.items():
        df.to_csv(folder / f"{names[group]}_breast_cancer_RUO.csv", index=False)

    inventory = list_tables(tmp_path)
    assert len(inventory) == 5
    loaded = load_indication_tables(inventory, "breast_cancer")
    assert set(loaded) == set(names)
    with pytest.raises(FileNotFoundError):
        load_indication_tables(inventory, "lung_cancer")


def test_load_rejects_duplicated_slides(tmp_path, tables):
    folder = tmp_path / "data" / "x"
    folder.mkdir(parents=True)
    tables["tme"] = pd.concat([tables["tme"], tables["tme"].iloc[[0]]])
    for group, df in tables.items():
        prefix = {v: k for k, v in TABLE_GROUPS.items()}[group]
        df.to_csv(folder / f"{prefix}_x_RUO.csv", index=False)
    with pytest.raises(ValueError, match="duplicated"):
        load_indication_tables(list_tables(tmp_path), "x")


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------
def test_check_same_slides(tables):
    tables["tls"] = tables["tls"].iloc[:3]
    report = check_same_slides(tables)
    assert report.loc["tls", "missing_here"] == 1
    assert report.loc["tme", "missing_here"] == 0


def test_merge_keeps_one_row_per_slide_and_renames(tables):
    renames = {"wtr_tissue": {"ABSOLUTE_AREA": "ABSOLUTE_AREA_WTR"}}
    slides, source = merge_slide_tables(tables, renames)
    assert len(slides) == 4
    assert source["ABSOLUTE_AREA"] == "tme"
    assert source["ABSOLUTE_AREA_WTR"] == "wtr_tissue"
    assert slides["ABSOLUTE_AREA_WTR"].tolist() == [1.0, 2, 3, 4]
    assert list(slides.columns[:len(ID_COLS)]) == ID_COLS


def test_merge_without_rename_keeps_first_table(tables):
    slides, source = merge_slide_tables(tables)
    assert slides["ABSOLUTE_AREA"].tolist() == [10.0, 20, 30, 40]
    assert "ABSOLUTE_AREA_WTR" not in slides


def test_merge_rejects_duplicated_keys(tables):
    tables["wtr_cell"] = pd.concat([tables["wtr_cell"], tables["wtr_cell"].iloc[[0]]])
    with pytest.raises(pd.errors.MergeError):
        merge_slide_tables(tables)


# ---------------------------------------------------------------------------
# Barcodes
# ---------------------------------------------------------------------------
def test_barcode_parsing():
    s = pd.Series(BARCODES)
    assert patient_id(s).tolist() == ["TCGA-A1-0001", "TCGA-A1-0001", "TCGA-A1-0002", "TCGA-A1-0003"]
    assert sample_type(s).tolist() == ["01", "01", "06", "01"]


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
FEATURES = [
    "CELL_DENSITY_LYMPHOCYTE_IN_CARCINOMA",
    "CELL_DENSITY_LYMPHOCYTE_IN_BLOOD",
    "RELATIVE_AREA_OUT_OF_FOCUS",
    "DENSITY_OF_LYMPHOCYTES_AROUND_CARCINOMA_CELL_IN_STROMA_40",
    "DENSITY_OF_LYMPHOCYTES_AROUND_CARCINOMA_CELL_IN_STROMA_20",
    "DENSITY_OF_MACROPHAGES_AROUND_CARCINOMA_CELL_IN_VESSEL_20",
    "LARGEST_REGION_AREA_CARCINOMA",
    "RELATIVE_AREA_VESSEL",
    "CELL_DENSITY_LYMPHOCYTE_IN_VESSEL",
    "CELL_DENSITY_ENDOTHELIAL_CELL",
]


def test_build_and_apply_exclusions():
    rules = build_tme_exclusions(
        {"classes": ["Blood"], "model_variable_keys": ["qc_cls"],
         "substrings": {"40 µm": ["_40"], "shape": ["LARGEST"]}},
        model_variables={"qc_cls": ["Out of focus"]},
    )
    assert rules["Tissues and cell types not studied"] == ["BLOOD"]
    kept, log = apply_exclusions(FEATURES, rules)
    assert "CELL_DENSITY_LYMPHOCYTE_IN_BLOOD" not in kept
    assert "RELATIVE_AREA_OUT_OF_FOCUS" not in kept
    assert "DENSITY_OF_LYMPHOCYTES_AROUND_CARCINOMA_CELL_IN_STROMA_40" not in kept
    assert "LARGEST_REGION_AREA_CARCINOMA" not in kept
    assert len(kept) == 6
    assert log["n_left"].iloc[-1] == 6 and log["n_removed"].sum() == 4


def test_build_exclusions_requires_model_variables():
    with pytest.raises(KeyError):
        build_tme_exclusions({"model_variable_keys": ["qc_cls"]}, model_variables={})


def test_vessel_rule_removes_only_features_measured_inside_vessels():
    vessel = match_pattern(FEATURES, substrings=["_IN_VESSEL_"], suffixes=["_IN_VESSEL"])
    assert vessel == ["DENSITY_OF_MACROPHAGES_AROUND_CARCINOMA_CELL_IN_VESSEL_20",
                      "CELL_DENSITY_LYMPHOCYTE_IN_VESSEL"]
    assert "RELATIVE_AREA_VESSEL" not in vessel            # vessel tissue itself is kept
    assert "CELL_DENSITY_ENDOTHELIAL_CELL" not in vessel   # endothelial cells are kept


def test_select_tls_and_wtr_features():
    source = pd.Series({
        "COUNT_TLS_MATURE": "tls", "DENSITY_TLS_MATURE": "tls", "ECCENTRICITY_TLS_1": "tls",
        "CELL_DENSITY_LYMPHOCYTE_IN_TUMOR_CORE": "wtr_cell",
        "CELL_DENSITY_LYMPHOCYTE_IN_TUMOR_CORE_NECROSIS": "wtr_cell",
        "CELL_DENSITY_MACROPHAGE_IN_TUMOR_CORE": "wtr_cell",
        "ABSOLUTE_AREA_TUMOR_CORE": "wtr_tissue",
    })
    assert select_tls_features(source, ["COUNT_", "DENSITY_"]) == ["COUNT_TLS_MATURE", "DENSITY_TLS_MATURE"]
    wtr = select_wtr_features(source, "wtr_cell", ["CELL_DENSITY", "LYMPHOCYTE"], ["NECROSIS"],
                              ["ABSOLUTE_AREA_TUMOR_CORE"])
    assert wtr == ["CELL_DENSITY_LYMPHOCYTE_IN_TUMOR_CORE", "ABSOLUTE_AREA_TUMOR_CORE"]
    with pytest.raises(KeyError):
        select_wtr_features(source, "wtr_cell", ["LYMPHOCYTE"], extra_features=["NOT_A_FEATURE"])


# ---------------------------------------------------------------------------
# QC and one slide per patient
# ---------------------------------------------------------------------------
@pytest.fixture
def slides():
    df = _ids()
    df["SAMPLE_TYPE"] = sample_type(df["TCGA_FILE_NAME"])
    df["ABSOLUTE_AREA_CARCINOMA"] = [3e6, 5e6, 9e6, 1e6]           # µm²; last slide too small
    df["RELATIVE_AREA_OUT_OF_FOCUS"] = [10.0, np.nan, 0, 0]
    df["RELATIVE_AREA_MARKER"] = [50.0, 0, 0, 0]                   # first slide 60% unusable
    return df


def test_slide_qc(slides):
    qc = slide_qc(slides, "ABSOLUTE_AREA_CARCINOMA", 2.0,
                  ["RELATIVE_AREA_OUT_OF_FOCUS", "RELATIVE_AREA_MARKER"], 0.5)
    assert qc["primary_tumor"].tolist() == [True, True, False, True]
    assert qc["enough_carcinoma"].tolist() == [True, True, True, False]
    assert qc["good_quality"].tolist() == [False, True, True, True]
    assert qc["unusable_fraction"].tolist() == pytest.approx([0.6, 0.0, 0.0, 0.0])
    assert qc["qc_pass"].tolist() == [False, True, False, False]


def test_one_slide_per_patient_keeps_largest_passing_slide(slides):
    qc_pass = pd.Series([True, True, False, True], index=slides.index)
    selected = one_slide_per_patient(slides, qc_pass, by="ABSOLUTE_AREA_CARCINOMA")
    assert selected["TCGA_CASE_ID"].is_unique
    assert set(selected["TCGA_FILE_NAME"]) == {BARCODES[1], BARCODES[3]}
    assert selected.set_index("TCGA_CASE_ID")["N_SLIDES_TOTAL"].to_dict() == {
        "TCGA-A1-0001": 2, "TCGA-A1-0003": 1}


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------
def test_safe_ratio_and_presence():
    r = safe_ratio(pd.Series([1.0, 1, 0, np.nan]), pd.Series([2.0, 0, 0, 1]))
    assert r.iloc[0] == 0.5 and r.iloc[1:].isna().all()

    df = pd.DataFrame({"a": [0.0, 1, 0, np.nan], "b": [0.0, 0, 2, 0]})
    assert presence_flag(df, ["a", "b"]).tolist()[:3] == [0.0, 1.0, 1.0]
    assert np.isnan(presence_flag(df, ["a", "b"]).iloc[3])


def test_add_derived_features():
    df = pd.DataFrame({"x": [2.0, 4], "y": [1.0, 0], "t": [0.0, 3]})
    out, names = add_derived_features(df, {"RATIO": ["x", "y"]}, {"PRESENT": ["t"]})
    assert names == ["RATIO", "PRESENT"]
    assert out["RATIO"].iloc[0] == 2.0 and np.isnan(out["RATIO"].iloc[1])
    assert out["PRESENT"].tolist() == [0.0, 1.0]
    assert "RATIO" not in df   # input not modified


def test_non_constant():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [np.nan] * 3, "c": [1, 2, np.nan]})
    assert non_constant(df, ["a", "b", "c"]) == ["c"]


# ---------------------------------------------------------------------------
# Units and summaries
# ---------------------------------------------------------------------------
def test_density_per_mm2():
    factors = {"tme": 1e6, "wtr_cell": 1.0}
    assert density_per_mm2(pd.Series([1e-4]), "tme", factors).iloc[0] == pytest.approx(100)
    assert density_per_mm2(pd.Series([100.0]), "wtr_cell", factors).iloc[0] == 100
    with pytest.raises(KeyError):
        density_per_mm2(pd.Series([1.0]), "tls", factors)


def test_zero_or_missing_and_spearman():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"ref": rng.normal(size=50)})
    df["up"] = df["ref"] * 2 + 1
    df["sparse"] = np.where(np.arange(50) < 40, 0.0, 1.0)
    df.loc[:4, "sparse"] = np.nan
    assert zero_or_missing(df, ["sparse"])["sparse"] == pytest.approx(0.8)
    rho = spearman_with(df, ["up", "ref"], "ref")    # the reference itself may be in the list
    assert rho["up"] == pytest.approx(1.0) and rho["ref"] == 1.0


def test_list_tables_fails_clearly_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="No OpenTME tables"):
        list_tables(tmp_path)


def test_match_thumbnail_prefers_first_key_and_ignores_other_files():
    files = ["data/breast_cancer/tables/TCGA-AA-0001.csv",
             "data/breast_cancer/per_slide_results/TCGA-AA-0001-01Z-00-DX1.UUID-A/cell_classification.png",
             "data/breast_cancer/per_slide_results/TCGA-AA-0001-01Z-00-DX2.UUID-B/cell_classification.png"]
    assert match_thumbnail(files, [None, "UUID-B", "TCGA-AA-0001"]).split("/")[-2].endswith("UUID-B")
    assert match_thumbnail(files, [float("nan"), "TCGA-AA-0001"]).split("/")[-2].endswith("UUID-A")
    assert match_thumbnail(files, ["TCGA-ZZ-9999"]) is None


def test_match_thumbnail_filters_by_layer():
    files = ["data/breast_cancer/per_slide_results/TCGA-AA-0001-01Z-00-DX1.U/tissue_segmentation.png",
             "data/breast_cancer/per_slide_results/TCGA-AA-0001-01Z-00-DX1.U/cell_classification.png"]
    assert match_thumbnail(files, ["TCGA-AA-0001"], kind="cell_classification").endswith("cell_classification.png")
    assert match_thumbnail(files, ["TCGA-AA-0001"], kind="other") is None
