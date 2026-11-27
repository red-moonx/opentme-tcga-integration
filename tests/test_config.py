"""Tests for opentme_tcga.config."""
from pathlib import Path

import pytest
import yaml

from opentme_tcga.config import DATA_DIR_ENV, deep_merge, find_repo_root, load_config, load_dotenv, source_file


def test_deep_merge_overrides_nested_values():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    merged = deep_merge(base, {"b": {"y": 3, "z": 4}, "c": 5})
    assert merged == {"a": 1, "b": {"x": 1, "y": 3, "z": 4}, "c": 5}
    assert base["b"]["y"] == 2   # input not modified


def _fake_repo(root: Path, indication: dict) -> Path:
    (root / "configs" / "indications").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    project = {
        "seed": 1,
        "paths": {"data": "data", "raw_opentme": "{data}/raw/opentme", "processed": "{data}/processed",
                  "results": "results/{indication}", "figures": "figures/{indication}"},
        "files": {"opentme_patients": "{processed}/opentme_{opentme}_patients.parquet"},
    }
    analysis = {"opentme_processing": {"slide_qc": {"min_carcinoma_area_mm2": 2.0, "max_unusable_fraction": 0.5}}}
    (root / "configs" / "project.yaml").write_text(yaml.safe_dump(project))
    (root / "configs" / "analysis.yaml").write_text(yaml.safe_dump(analysis))
    (root / "configs" / "indications" / f"{indication['indication']['name']}.yaml").write_text(
        yaml.safe_dump(indication))
    return root


def test_load_config_merges_and_resolves_paths(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    root = _fake_repo(tmp_path, {
        "indication": {"name": "lung", "opentme": "lung_cancer"},
        "opentme_processing": {"slide_qc": {"min_carcinoma_area_mm2": 1.0}},   # override
    })
    cfg = load_config("lung", root=root)
    assert cfg["opentme_processing"]["slide_qc"] == {"min_carcinoma_area_mm2": 1.0, "max_unusable_fraction": 0.5}
    assert cfg["paths"]["raw_opentme"] == root / "data" / "raw" / "opentme"
    assert cfg["paths"]["results"] == root / "results" / "lung"
    assert cfg["paths"]["figures"] == root / "figures" / "lung"
    assert cfg["paths"]["opentme_patients"] == root / "data" / "processed" / "opentme_lung_cancer_patients.parquet"


def test_data_dir_can_be_overridden(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path / "repo", {"indication": {"name": "lung", "opentme": "lung_cancer"}})
    outside = tmp_path / "elsewhere"
    monkeypatch.setenv(DATA_DIR_ENV, str(outside))
    cfg = load_config("lung", root=root)
    assert cfg["paths"]["processed"] == outside / "processed"


def test_unknown_indication(tmp_path):
    root = _fake_repo(tmp_path, {"indication": {"name": "lung", "opentme": "lung_cancer"}})
    with pytest.raises(FileNotFoundError, match="Available: \\['lung'\\]"):
        load_config("breast", root=root)


def test_find_repo_root_from_subfolder(tmp_path):
    root = _fake_repo(tmp_path, {"indication": {"name": "lung", "opentme": "lung_cancer"}})
    (root / "notebooks").mkdir()
    assert find_repo_root(root / "notebooks") == root


def test_repository_configs_load():
    """The real configuration files of the repository load for every indication."""
    cfg = load_config("breast")
    proc = cfg["opentme_processing"]
    assert cfg["indication"]["opentme"] == "breast_cancer"
    assert proc["slide_qc"]["min_carcinoma_area_mm2"] == 2.0
    assert set(cfg["density_per_mm2"]) >= {"tme", "tls", "wtr_cell"}


def test_dotenv_sets_data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    root = _fake_repo(tmp_path / "repo", {"indication": {"name": "lung", "opentme": "lung_cancer"}})
    outside = tmp_path / "my data"
    (root / ".env").write_text(f'# local settings\n{DATA_DIR_ENV}="{outside}"\n')
    cfg = load_config("lung", root=root)
    assert cfg["paths"]["processed"] == outside / "processed"


def test_dotenv_does_not_override_existing_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_VAR", "shell")
    (tmp_path / ".env").write_text("SOME_VAR=file\nexport OTHER=1\n")
    assert load_dotenv(tmp_path / ".env") == {"SOME_VAR": "file", "OTHER": "1"}
    import os
    assert os.environ["SOME_VAR"] == "shell"
    monkeypatch.delenv("OTHER")


def test_source_file_formats_templates(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    root = _fake_repo(tmp_path, {"indication": {"name": "lung", "opentme": "lung_cancer", "tcga_code": "LUAD",
                                                "cbioportal_study": "luad_study"}})
    cfg = load_config("lung", root=root)
    cfg["paths"]["raw_tcga"] = root / "raw"
    cfg["tcga_molecular"] = {"sources": {"arm": {"file": "x_{tcga_code_lower}.txt", "url": "https://h/{cbioportal_study}/a"}},
                             "subtypes": {"file": "subtypes.xlsx"}}
    arm = source_file(cfg, "arm")
    assert arm["path"] == root / "raw" / "x_luad.txt" and arm["url"] == "https://h/luad_study/a"
    assert source_file(cfg, "subtypes")["path"] == root / "raw" / "subtypes.xlsx"


def test_repository_tcga_sources_resolve():
    cfg = load_config("breast")
    assert source_file(cfg, "arm_level")["url"].endswith("brca_tcga_pan_can_atlas_2018/data_armlevel_cna.txt")
    assert source_file(cfg, "rna")["path"].name == "TCGA.BRCA.HiSeqV2.gz"
    assert source_file(cfg, "subtypes")["path"].name == "Thennavan_2021_DataS1.xlsx"
