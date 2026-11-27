"""Integration of the OpenTME patient table (notebook 01) with the TCGA molecular table (notebook 02).

One row per patient, keyed by the 12-character TCGA barcode (`patient`). The OpenTME
patients define the cohort; patients absent from the molecular table (redacted in
TCGA, notebook 02) are excluded.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PATIENT = "patient"
BARCODE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")


def load_patient_table(path: Path, key: str | None = None) -> pd.DataFrame:
    """Read a patient table (parquet or csv) and name its barcode column `patient`.

    `key` is the barcode column; when None (or absent) the index holds the barcode.
    Barcodes are truncated to 12 characters.
    """
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if key is not None and key in df.columns:
        df = df.rename(columns={key: PATIENT})
    else:
        df = df.rename_axis(PATIENT).reset_index()
    df[PATIENT] = df[PATIENT].astype(str).str.strip().str.upper().str[:12]
    return df


def check_patients(df: pd.DataFrame, name: str) -> dict:
    """Fail if any barcode is malformed or duplicated; return a short summary."""
    malformed = ~df[PATIENT].str.match(BARCODE)
    duplicated = df[PATIENT].duplicated()
    if malformed.any():
        raise ValueError(f"{name}: malformed barcodes {df.loc[malformed, PATIENT].head().tolist()}")
    if duplicated.any():
        raise ValueError(f"{name}: duplicated patients {df.loc[duplicated, PATIENT].head().tolist()}")
    return {"table": name, "rows": len(df), "patients": df[PATIENT].nunique()}


def integrate(opentme: pd.DataFrame, molecular: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Join the two tables on `patient`, keeping OpenTME patients present in the molecular table.

    Returns the integrated table and the OpenTME patients excluded because they are not
    in the molecular table. Raises if a column name (other than `patient`) is in both tables.
    """
    shared = sorted((set(opentme.columns) & set(molecular.columns)) - {PATIENT})
    if shared:
        raise ValueError(f"Column names in both tables: {shared}")
    excluded = sorted(set(opentme[PATIENT]) - set(molecular[PATIENT]))
    kept = opentme[~opentme[PATIENT].isin(excluded)]
    integrated = kept.merge(molecular, on=PATIENT, how="left", validate="one_to_one")
    return integrated, excluded


def provenance(integrated: pd.DataFrame, opentme_columns, opentme_source: pd.Series | None = None,
               dictionary: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per column: modality (opentme / tcga), detailed source, label and coverage.

    `opentme_source` maps OpenTME features to their table (notebook 01); `dictionary` is the
    molecular data dictionary (notebook 02), with `column`, `source`, `level` and `label`.
    """
    cols = [c for c in integrated.columns if c != PATIENT]
    opentme_columns = set(opentme_columns)
    out = pd.DataFrame({"column": cols})
    is_opentme = out["column"].isin(opentme_columns)
    out["modality"] = is_opentme.map({True: "opentme", False: "tcga"})

    table = out["column"].map(opentme_source) if opentme_source is not None else pd.Series(None, index=out.index, dtype=object)
    opentme_src = table.map(lambda t: f"OpenTME {t}" if isinstance(t, str) else "OpenTME")
    d = dictionary.set_index("column") if dictionary is not None else pd.DataFrame()
    tcga_src = out["column"].map(d["source"]) if "source" in d else pd.Series(None, index=out.index, dtype=object)
    out["source"] = opentme_src.where(is_opentme, tcga_src)
    for field in ("level", "label"):
        out[field] = out["column"].map(d[field]) if field in d else None

    out["n_with_data"] = out["column"].map(integrated[cols].notna().sum())
    out["pct_with_data"] = (100 * out["n_with_data"] / len(integrated)).round(1)
    return out
