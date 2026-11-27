"""Tests for opentme_tcga.integration."""
import numpy as np
import pandas as pd
import pytest

from opentme_tcga.integration import check_patients, integrate, load_patient_table, provenance


@pytest.fixture
def tables():
    opentme = pd.DataFrame({"TCGA_CASE_ID": ["TCGA-A1-0001", "TCGA-A1-0002", "TCGA-A1-0003"],
                            "CELL_DENSITY_X": [1.0, 2.0, 3.0]})
    molecular = pd.DataFrame({"thor_tmb": [0.5, np.nan], "clin_age": [50, 60]},
                             index=pd.Index(["TCGA-A1-0001", "TCGA-A1-0002"], name="patient"))
    return opentme, molecular


def test_load_patient_table_column_and_index(tmp_path, tables):
    opentme, molecular = tables
    opentme.to_parquet(tmp_path / "o.parquet", index=False)
    molecular.to_parquet(tmp_path / "m.parquet")
    o = load_patient_table(tmp_path / "o.parquet", "TCGA_CASE_ID")
    m = load_patient_table(tmp_path / "m.parquet")
    assert "patient" in o and "patient" in m
    assert m["patient"].tolist() == ["TCGA-A1-0001", "TCGA-A1-0002"]


def test_check_patients():
    ok = pd.DataFrame({"patient": ["TCGA-A1-0001", "TCGA-A1-0002"]})
    assert check_patients(ok, "x") == {"table": "x", "rows": 2, "patients": 2}
    with pytest.raises(ValueError, match="duplicated"):
        check_patients(pd.DataFrame({"patient": ["TCGA-A1-0001"] * 2}), "x")
    with pytest.raises(ValueError, match="malformed"):
        check_patients(pd.DataFrame({"patient": ["TCGA-A1-001"]}), "x")


def test_integrate_excludes_patients_without_molecular_data(tables):
    opentme, molecular = tables
    o = opentme.rename(columns={"TCGA_CASE_ID": "patient"})
    m = molecular.reset_index()
    integrated, excluded = integrate(o, m)
    assert excluded == ["TCGA-A1-0003"]
    assert integrated["patient"].tolist() == ["TCGA-A1-0001", "TCGA-A1-0002"]
    assert np.isnan(integrated.loc[1, "thor_tmb"])        # missing stays missing


def test_integrate_rejects_shared_columns(tables):
    opentme, molecular = tables
    o = opentme.rename(columns={"TCGA_CASE_ID": "patient"}).assign(clin_age=1)
    with pytest.raises(ValueError, match="clin_age"):
        integrate(o, molecular.reset_index())


def test_provenance(tables):
    opentme, molecular = tables
    o = opentme.rename(columns={"TCGA_CASE_ID": "patient"})
    integrated, _ = integrate(o, molecular.reset_index())
    dictionary = pd.DataFrame({"column": ["thor_tmb", "clin_age"], "source": ["Thorsson", "CDR"],
                               "level": ["main", "main"], "label": ["TMB", "Age"]})
    prov = provenance(integrated, o.columns, pd.Series({"CELL_DENSITY_X": "tme"}), dictionary).set_index("column")
    assert prov.loc["CELL_DENSITY_X", "modality"] == "opentme"
    assert prov.loc["CELL_DENSITY_X", "source"] == "OpenTME tme"
    assert prov.loc["thor_tmb", "source"] == "Thorsson" and prov.loc["thor_tmb", "label"] == "TMB"
    assert prov.loc["thor_tmb", "n_with_data"] == 1 and prov.loc["clin_age", "pct_with_data"] == 100.0


def test_provenance_without_sources(tables):
    opentme, molecular = tables
    o = opentme.rename(columns={"TCGA_CASE_ID": "patient"})
    integrated, _ = integrate(o, molecular.reset_index())
    prov = provenance(integrated, o.columns).set_index("column")
    assert prov.loc["CELL_DENSITY_X", "source"] == "OpenTME" and pd.isna(prov.loc["thor_tmb", "source"])
