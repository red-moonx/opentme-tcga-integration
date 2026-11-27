"""OpenTME processing: from the five slide-level tables to a patient-level master table.

OpenTME (Galama et al., arXiv:2604.12075, 2026) provides, for each cancer type, five
slide-level tables computed with Atlas H&E-TME on TCGA diagnostic slides:

==================  ================================================================
``tme``             whole-slide tissue, cell and neighbourhood features (~4,500)
``wtr_cell``        cell features in the zones of the whole tumour region
``wtr_tissue``      area and shape of those zones
``wtr_neighborhood`` cell neighbourhoods in those zones
``tls``             tertiary lymphoid structures: number, density, area, maturity
==================  ================================================================

The functions below download and load these tables, merge them into one row per
slide, select features with explicit rules, apply slide quality control, keep one
slide per patient and add a few derived features. All parameters come from the
configuration (``configs/analysis.yaml``); nothing is specific to a cancer type.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# OpenTME schema
# ---------------------------------------------------------------------------
ID_COLS = ["TCGA_FILE_NAME", "TCGA_SLIDE_UUID", "TCGA_CASE_ID", "TCGA_PROJECT_ID", "INDICATION"]
SLIDE_KEY = "TCGA_FILE_NAME"
PATIENT_KEY = "TCGA_CASE_ID"

# File prefix -> short name of each slide-level table
TABLE_GROUPS = {
    "tme_features": "tme",
    "whole_tumor_region_cell_features": "wtr_cell",
    "whole_tumor_region_tissue_features": "wtr_tissue",
    "whole_tumor_region_neighborhood_features": "wtr_neighborhood",
    "tertiary_lymphoid_structures_tls_features": "tls",
}


# ---------------------------------------------------------------------------
# Download and file inventory
# ---------------------------------------------------------------------------
def download_opentme(raw_dir: Path, repo_id: str, allow_patterns: list[str], revision: str | None = None) -> Path:
    """Download the tabular part of OpenTME at version `revision` (requires `hf auth login`; files already
    present are skipped)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=raw_dir,
                                  allow_patterns=allow_patterns, revision=revision))


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def match_thumbnail(files, keys, kind: str | None = None, folder: str = "per_slide_results") -> str | None:
    """First slide image path that contains one of `keys`, tried in order (e.g. slide file name, slide UUID,
    case ID).

    OpenTME stores one folder of images per slide, `<cancer>/<folder>/<slide file name>/<layer>.png`, with one
    image per analysis layer (cell_classification, tissue_segmentation, tumor_zones, wsi, ...). Only images
    inside `folder` are considered and, if `kind` is given, only the layer of that name."""
    images = [str(f) for f in files if Path(str(f)).suffix.lower() in IMAGE_SUFFIXES and f"/{folder}/" in str(f)]
    if kind:
        images = [f for f in images if Path(f).stem == kind]
    for key in keys:
        if not isinstance(key, str) or not key:
            continue
        for f in images:
            if key in f:
                return f
    return None


def get_thumbnail(raw_dir: Path, repo_id: str, keys, kind: str | None = None,
                  revision: str | None = None) -> Path | None:
    """Local path of an OpenTME slide image: found in `raw_dir` or downloaded there once (requires
    `hf auth login`), so only the images that are needed are downloaded."""
    raw_dir = Path(raw_dir)
    local = match_thumbnail([f for f in raw_dir.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES], keys, kind)
    if local:
        return Path(local)
    from huggingface_hub import HfApi, hf_hub_download

    remote = match_thumbnail(HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision), keys, kind)
    if remote is None:
        return None
    return Path(hf_hub_download(repo_id=repo_id, filename=remote, repo_type="dataset", local_dir=raw_dir,
                                revision=revision))


def parse_table_path(path: Path | str) -> dict:
    """Split '<table>_<cancer>_RUO.csv' into its table group and cancer name."""
    stem = Path(path).stem.removesuffix("_RUO")
    for prefix, group in TABLE_GROUPS.items():
        if stem.startswith(prefix + "_"):
            return {"group": group, "cancer": stem.removeprefix(prefix + "_")}
    raise ValueError(f"Unknown OpenTME table: {Path(path).name}")


def list_tables(raw_dir: Path) -> pd.DataFrame:
    """Inventory of the downloaded tables: one row per file with group, cancer and path."""
    files = sorted((Path(raw_dir) / "data").glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No OpenTME tables in {Path(raw_dir) / 'data'}. Check the data folder "
            "(paths.data in configs/project.yaml or OPENTME_TCGA_DATA_DIR in .env), "
            "or download the tables first."
        )
    return pd.DataFrame([{**parse_table_path(f), "path": f} for f in files],
                        columns=["group", "cancer", "path"])


def load_model_variables(raw_dir: Path) -> dict:
    """Class vocabularies of the Atlas H&E-TME models (settings/model_variables.yaml)."""
    with open(Path(raw_dir) / "settings" / "model_variables.yaml") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Loading and merging
# ---------------------------------------------------------------------------
def validate_table(df: pd.DataFrame, group: str) -> None:
    """Check identifier columns and that each slide appears once."""
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{group}: missing identifier columns {missing}")
    if df[SLIDE_KEY].duplicated().any():
        raise ValueError(f"{group}: duplicated slides")


def load_indication_tables(inventory: pd.DataFrame, cancer: str) -> dict[str, pd.DataFrame]:
    """Load and validate the five slide-level tables of one cancer type."""
    paths = inventory.loc[inventory["cancer"] == cancer].set_index("group")["path"]
    missing = set(TABLE_GROUPS.values()) - set(paths.index)
    if missing:
        raise FileNotFoundError(f"{cancer}: missing tables {sorted(missing)}")

    tables = {}
    for group in TABLE_GROUPS.values():
        df = pd.read_csv(paths[group], low_memory=False)
        validate_table(df, group)
        tables[group] = df
    return tables


def check_same_slides(tables: dict[str, pd.DataFrame], reference: str = "tme") -> pd.DataFrame:
    """Slides present only in each table, or missing from it, relative to the reference table."""
    ref = set(tables[reference][SLIDE_KEY])
    rows = []
    for group, df in tables.items():
        slides = set(df[SLIDE_KEY])
        rows.append({"table": group, "n_slides": len(slides),
                     "only_here": len(slides - ref), "missing_here": len(ref - slides)})
    return pd.DataFrame(rows).set_index("table")


def merge_slide_tables(tables: dict[str, pd.DataFrame], renames: dict[str, dict[str, str]] | None = None,
                       reference: str = "tme") -> tuple[pd.DataFrame, pd.Series]:
    """Merge all tables one-to-one on the slide key.

    Returns the wide slide-level table and `feature_source`, a Series mapping each
    feature to the table it comes from. A feature name already taken by an earlier
    table is skipped (use `renames` to keep both).
    """
    tables = {g: df.rename(columns=(renames or {}).get(g, {})) for g, df in tables.items()}
    slides = tables[reference][ID_COLS].copy()
    source: dict[str, str] = {}
    for group, df in tables.items():
        features = [c for c in df.columns if c not in ID_COLS and c not in source]
        source.update({c: group for c in features})
        slides = slides.merge(df[[SLIDE_KEY] + features], on=SLIDE_KEY, how="left", validate="one_to_one")
    if len(slides) != len(tables[reference]):
        raise RuntimeError("Row count changed while merging slide tables.")
    return slides.copy(), pd.Series(source, name="table").rename_axis("feature")


# ---------------------------------------------------------------------------
# TCGA barcodes
# ---------------------------------------------------------------------------
def patient_id(barcodes: pd.Series) -> pd.Series:
    """Participant barcode: first 12 characters (e.g. 'TCGA-A2-A04P')."""
    return barcodes.str[:12]


def sample_type(barcodes: pd.Series) -> pd.Series:
    """Sample type code: characters 14-15 ('01' primary solid tumour, '06' metastasis, '11' normal)."""
    return barcodes.str[13:15]


def slides_per_patient(slides: pd.DataFrame, patient_col: str = PATIENT_KEY) -> pd.Series:
    """Number of slides of each patient."""
    return slides.groupby(patient_col).size().rename("n_slides")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
def split_numeric(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Split `features` into numeric and non-numeric columns of `df`."""
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    return numeric, [c for c in features if c not in numeric]


def downcast_features(df: pd.DataFrame, features: list[str], dtype: str = "float32") -> pd.DataFrame:
    """Copy of `df` with the given numeric features cast to `dtype`."""
    out = df.copy()
    out[features] = out[features].astype(dtype)
    return out


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def to_caps(name: str) -> str:
    """Class name as it appears in feature names ('Out of focus' -> 'OUT_OF_FOCUS')."""
    return name.upper().replace(" ", "_")


def search_features(feature_source: pd.Series, *keywords: str, table: str | None = None) -> list[str]:
    """Features containing all keywords (case-insensitive), optionally within one table."""
    names = feature_source.index if table is None else feature_source.index[feature_source == table]
    return [n for n in names if all(k.upper() in n for k in keywords)]


def build_tme_exclusions(rules: dict, model_variables: dict | None = None) -> dict[str, list[str]]:
    """Ordered exclusion rules {reason: substrings} from the `tme_exclusions` config section."""
    out = {"Tissues and cell types not studied": [to_caps(c) for c in rules.get("classes", [])]}
    for key in rules.get("model_variable_keys", []):
        if model_variables is None or key not in model_variables:
            raise KeyError(f"model_variables has no key '{key}'")
        out[f"Model classes '{key}'"] = [to_caps(c) for c in model_variables[key]]
    out.update(rules.get("substrings", {}))
    return out


def apply_exclusions(features: list[str], rules: dict[str, list[str]]) -> tuple[list[str], pd.DataFrame]:
    """Drop features containing any substring of each rule, in order.

    Returns the kept features and a log with the number of features left after each rule.
    """
    kept = list(features)
    log = [{"step": "Start", "n_removed": 0, "n_left": len(kept)}]
    for reason, words in rules.items():
        before = len(kept)
        kept = [f for f in kept if not any(w in f for w in words)]
        log.append({"step": f"Exclude {reason}", "n_removed": before - len(kept), "n_left": len(kept)})
    return kept, pd.DataFrame(log)


def match_pattern(features: list[str], substrings: list[str] = (), suffixes: list[str] = ()) -> list[str]:
    """Features containing any of `substrings` or ending with any of `suffixes`."""
    return [f for f in features
            if any(s in f for s in substrings) or any(f.endswith(s) for s in suffixes)]


def select_tls_features(feature_source: pd.Series, prefixes: list[str], table: str = "tls") -> list[str]:
    """Summary TLS features (e.g. COUNT_, DENSITY_, AREA_ by maturity)."""
    return [f for f in search_features(feature_source, table=table) if f.startswith(tuple(prefixes))]


def select_wtr_features(feature_source: pd.Series, table: str, include_keywords: list[str],
                        exclude_keywords: list[str] = (), extra_features: list[str] = ()) -> list[str]:
    """Whole-tumour-region features containing all `include_keywords` and none of `exclude_keywords`."""
    found = search_features(feature_source, *include_keywords, table=table)
    selected = [f for f in found if not any(w in f for w in exclude_keywords)] + list(extra_features)
    missing = [f for f in selected if f not in feature_source.index]
    if missing:
        raise KeyError(f"Features not found: {missing}")
    return selected


# ---------------------------------------------------------------------------
# Slide QC and patient level
# ---------------------------------------------------------------------------
def slide_qc(slides: pd.DataFrame, area_feature: str, min_carcinoma_area_mm2: float,
             unusable_features: list[str], max_unusable_fraction: float,
             sample_types: list[str] = ("01",), sample_type_col: str = "SAMPLE_TYPE") -> pd.DataFrame:
    """QC metrics and pass/fail flags per slide.

    * primary_tumor: sample type in `sample_types`;
    * enough_carcinoma: carcinoma area (µm² -> mm²) >= `min_carcinoma_area_mm2`;
    * good_quality: unusable tissue (sum of % features, missing = 0) <= `max_unusable_fraction`.
    """
    qc = slides[[SLIDE_KEY, PATIENT_KEY, sample_type_col]].copy()
    qc["carcinoma_area_mm2"] = slides[area_feature] / 1e6
    qc["unusable_fraction"] = slides[list(unusable_features)].fillna(0).sum(axis=1) / 100
    qc["primary_tumor"] = qc[sample_type_col].isin(list(sample_types))
    qc["enough_carcinoma"] = qc["carcinoma_area_mm2"] >= min_carcinoma_area_mm2
    qc["good_quality"] = qc["unusable_fraction"] <= max_unusable_fraction
    qc["qc_pass"] = qc["primary_tumor"] & qc["enough_carcinoma"] & qc["good_quality"]
    return qc


def qc_sensitivity(values: pd.Series, thresholds: list[float], fails_below: bool) -> pd.DataFrame:
    """Number of slides excluded at each alternative threshold."""
    excluded = [(values < t).sum() if fails_below else (values > t).sum() for t in thresholds]
    return pd.DataFrame({"threshold": thresholds, "slides_excluded": excluded})


def one_slide_per_patient(slides: pd.DataFrame, qc_pass: pd.Series, by: str,
                          patient_col: str = PATIENT_KEY) -> pd.DataFrame:
    """Keep, for each patient, the QC-passing slide with the largest `by`; add N_SLIDES_TOTAL."""
    n_total = slides_per_patient(slides, patient_col)
    selected = (slides[qc_pass.to_numpy()]
                .sort_values([by, SLIDE_KEY], ascending=[False, True])
                .drop_duplicates(patient_col)
                .copy())
    selected["N_SLIDES_TOTAL"] = selected[patient_col].map(n_total)
    return selected


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------
def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, with NaN instead of ±inf when the denominator is 0."""
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def presence_flag(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """1.0 if the sum of `features` is > 0, 0.0 if it is 0, NaN if any value is missing."""
    total = df[list(features)].sum(axis=1, min_count=len(features))
    return (total > 0).astype(float).where(total.notna())


def add_derived_features(df: pd.DataFrame, ratios: dict[str, list[str]],
                         presence: dict[str, list[str]]) -> tuple[pd.DataFrame, list[str]]:
    """Add ratio features and presence flags; return the new table and the derived names."""
    out = df.copy()
    for name, (num, den) in ratios.items():
        out[name] = safe_ratio(out[num], out[den])
    for name, features in presence.items():
        out[name] = presence_flag(out, features)
    return out, list(ratios) + list(presence)


def non_constant(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Features with more than one distinct non-missing value."""
    n_values = df[features].nunique()
    return n_values[n_values > 1].index.tolist()


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
def density_per_mm2(values: pd.Series, table: str, factors: dict[str, float]) -> pd.Series:
    """Convert a density feature of `table` to per-mm² using the configured factor."""
    if table not in factors:
        raise KeyError(f"No density unit factor configured for table '{table}'")
    return values * factors[table]


# ---------------------------------------------------------------------------
# Exploratory summaries
# ---------------------------------------------------------------------------
def completeness(df: pd.DataFrame, features: list[str], feature_source: pd.Series) -> pd.DataFrame:
    """Per-feature source table, fraction missing and fraction zero."""
    return pd.DataFrame({
        "table": pd.Series(features, index=features).map(feature_source).fillna("derived"),
        "frac_missing": df[features].isna().mean(),
        "frac_zero": (df[features] == 0).mean(),
    })


def completeness_by_table(comp: pd.DataFrame, missing_flag: float, zero_flag: float) -> pd.DataFrame:
    """Summary of `completeness` by source table."""
    return comp.groupby("table").agg(
        n_features=("frac_missing", "size"),
        median_missing=("frac_missing", "median"),
        n_missing_flagged=("frac_missing", lambda x: int((x > missing_flag).sum())),
        n_zero_flagged=("frac_zero", lambda x: int((x > zero_flag).sum())),
    )


def max_abs_spearman(df: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series]:
    """For each feature, the largest |Spearman rho| with any other feature, and all unique pair values."""
    rho = df[features].corr(method="spearman").abs()
    off_diag = rho.where(~np.eye(len(rho), dtype=bool))
    pairs = rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1)).stack()
    return off_diag.max(), pairs


def zero_or_missing(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """Fraction of rows with a missing or zero value, per feature."""
    block = df[features]
    return (block.isna() | (block == 0)).mean()


def spearman_with(df: pd.DataFrame, features: list[str], reference: str, min_n: int = 10) -> pd.Series:
    """Spearman correlation of each feature with a reference column (NaN when fewer than `min_n` pairs)."""
    out = {}
    for f in features:
        if f == reference:
            out[f] = 1.0
            continue
        pair = df[[f, reference]].dropna()
        out[f] = pair[f].corr(pair[reference], method="spearman") if len(pair) >= min_n else np.nan
    return pd.Series(out, name=f"rho_{reference}")
