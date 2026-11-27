"""TCGA molecular data: from public PanCanAtlas resources to one row per patient.

Every source (clinical data, subtypes, purity, immune landscape, mutations, pathways,
arm-level copy number, hypoxia, RNA signatures) is reduced to a block indexed by the
12-character patient barcode, restricted to primary tumours (one sample per patient),
and added to a `MolecularTable` that keeps a data dictionary of every column.

Missing values are preserved: NaN means "not profiled", never "not altered".
Source files, column names and thresholds come from the configuration.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

PATIENT = "patient"


# ---------------------------------------------------------------------------
# Barcodes
# ---------------------------------------------------------------------------
def to_patient(barcodes: pd.Series | pd.Index) -> pd.Series:
    """12-character patient barcode (e.g. 'TCGA-A2-A04P') from any TCGA barcode."""
    return pd.Series(barcodes).astype(str).str.strip().str.upper().str[:12]


def primary_one_per_patient(df: pd.DataFrame, barcode_col: str,
                            sample_types: tuple[str, ...] = ("01",)) -> tuple[pd.DataFrame, dict]:
    """Keep primary-tumour samples and one sample per patient (vial A first, then lowest barcode).

    Returns the filtered table with a `patient` column, and a log with the sample types
    found, the number of patients before and after, and the extra samples dropped.
    """
    bc = df[barcode_col].astype(str).str.strip().str.upper()
    out = df.assign(**{PATIENT: bc.str[:12].to_numpy()}, _type=bc.str[13:15].to_numpy(),
                    _vial=bc.str[15:16].to_numpy(), _bc=bc.to_numpy())
    log = {"sample_types": out["_type"].value_counts().to_dict(), "n_patients": out[PATIENT].nunique()}
    out = out[out["_type"].isin(list(sample_types))]
    log["n_extra_samples"] = int(out[PATIENT].duplicated().sum())
    out = (out.sort_values([PATIENT, "_vial", "_bc"])
              .drop_duplicates(PATIENT, keep="first")
              .drop(columns=["_type", "_vial", "_bc"]))
    log["n_primary_patients"] = len(out)
    return out, log


def coverage(source_patients, reference: pd.Index) -> dict:
    """How many reference patients have data in a source."""
    found = set(pd.Series(source_patients).dropna().unique())
    present = reference.isin(list(found))
    return {"n": int(present.sum()), "total": len(reference),
            "pct": round(100 * present.mean(), 1), "missing": list(reference[~present])}


def format_coverage(name: str, cov: dict, show_missing: int = 5) -> str:
    text = f"[{name}] cohort patients with data: {cov['n']} / {cov['total']} ({cov['pct']:.1f} %)"
    if cov["missing"] and show_missing:
        text += f"; first missing: {cov['missing'][:show_missing]}"
    return text


# ---------------------------------------------------------------------------
# Molecular table with data dictionary
# ---------------------------------------------------------------------------
def infer_type(s: pd.Series) -> str:
    """'binary', 'continuous' or 'categorical'."""
    values = set(s.dropna().unique())
    if pd.api.types.is_bool_dtype(s) or (pd.api.types.is_numeric_dtype(s) and values <= {0, 1}):
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        return "continuous"
    return "categorical"


class MolecularTable:
    """Patient-level table built block by block, with a dictionary entry per column.

    Adding a block whose columns already exist replaces them, so notebook cells can be re-run.
    """

    def __init__(self, patients: pd.Index):
        self.data = pd.DataFrame(index=pd.Index(patients, name=PATIENT))
        self._entries: dict[str, dict] = {}

    def add(self, block: pd.DataFrame, source: str, level: str,
            descriptions: dict[str, str] | None = None) -> None:
        if not block.index.is_unique:
            raise ValueError("block must have exactly one row per patient")
        block = block.reindex(self.data.index)
        self.data = self.data.drop(columns=[c for c in block.columns if c in self.data.columns])
        self.data = self.data.join(block, how="left", validate="one_to_one")
        for col in block.columns:
            self._entries[col] = {"column": col, "source": source, "level": level,
                                  "type": infer_type(self.data[col]),
                                  "description": (descriptions or {}).get(col, "")}

    def drop_patients(self, patients) -> None:
        self.data = self.data.drop(index=[p for p in patients if p in self.data.index])

    def dictionary(self) -> pd.DataFrame:
        d = pd.DataFrame(list(self._entries.values()))
        d["n_with_data"] = d["column"].map(self.data.notna().sum())
        d["pct_with_data"] = (100 * d["n_with_data"] / len(self.data)).round(1)
        return d

    def __repr__(self) -> str:
        return f"MolecularTable({self.data.shape[0]} patients x {self.data.shape[1]} columns)"


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def download(url: str, path: Path, timeout: int = 300, chunk_size: int = 1 << 20) -> Path:
    """Download `url` to `path` unless the file already exists (streamed, written atomically)."""
    import requests

    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size):
                fh.write(block)
    tmp.rename(path)
    return path


def require_file(path: Path, source: str = "") -> Path:
    """Fail with a clear message when a manually downloaded file is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {Path(path).name}. Download it manually{f' from {source}' if source else ''} "
                                f"and save it as {path}.")
    return Path(path)


# ---------------------------------------------------------------------------
# Clinical data (TCGA-CDR)
# ---------------------------------------------------------------------------
def redacted_patients(cdr: pd.DataFrame, id_col: str = "bcr_patient_barcode",
                      flag_col: str = "Redaction") -> pd.Index:
    """Patients flagged as redacted by TCGA (to be excluded)."""
    return pd.Index(sorted(to_patient(cdr.loc[cdr[flag_col].notna(), id_col]).unique()), name=PATIENT)


def simplify_stage(stage: pd.Series) -> pd.Series:
    """'Stage IIA' -> 'II'; 'Stage X' and anything unexpected -> NaN."""
    s = stage.astype(str).str.extract(r"^Stage\s+([IV]+)[A-C]?$", expand=False)
    return s.where(s.isin(["I", "II", "III", "IV"]))


def tissue_source_site(patients: pd.Index | pd.Series) -> pd.Series:
    """Tissue source site (hospital) code: barcode characters 6-7."""
    return pd.Series(patients, index=patients).str[5:7]


def clinical_block(cdr: pd.DataFrame, columns: dict[str, str], numeric: list[str],
                   histology_map: dict[str, str] | None = None, id_col: str = "bcr_patient_barcode",
                   other_histology: str = "other") -> pd.DataFrame:
    """Clinical variables, one row per patient.

    `columns` maps CDR names to new names and must include 'clin_stage' and
    'clin_histological_type' if stage and histology are to be simplified.
    """
    out = (cdr[[id_col] + list(columns)].rename(columns=columns)
           .assign(**{PATIENT: lambda d: to_patient(d[id_col]).to_numpy()})
           .drop(columns=id_col).set_index(PATIENT))
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "clin_menopause_status" in out:
        out["clin_menopause_status"] = out["clin_menopause_status"].str.extract(r"^(Post|Pre|Peri)", expand=False)
    if "clin_histological_type" in out:
        if histology_map:
            out["clin_histology"] = (out["clin_histological_type"].map(histology_map).fillna(other_histology)
                                     .where(out["clin_histological_type"].notna()))
        else:                                  # no grouping configured: keep the original types
            out["clin_histology"] = out["clin_histological_type"]
    if "clin_stage" in out:
        out["clin_stage_simple"] = simplify_stage(out.pop("clin_stage"))
    out["clin_tss"] = tissue_source_site(out.index)
    return out


def thorsson_subtype(thorsson: pd.DataFrame, id_col: str, column: str, tcga_code: str) -> pd.Series:
    """TCGA molecular subtype from Thorsson Table S1, without the cancer prefix ('BRCA.LumA' -> 'LumA')."""
    s = thorsson.set_index(to_patient(thorsson[id_col]).to_numpy())[column]
    return s.astype("string").str.removeprefix(f"{tcga_code}.").astype(object).where(s.notna())


def recode_columns(df: pd.DataFrame, spec: dict[str, dict]) -> pd.DataFrame:
    """Build new columns from `spec`: {new_name: {source, recode (optional), dtype (optional)}}."""
    out = pd.DataFrame(index=df.index)
    for name, s in spec.items():
        col = df[s["source"]]
        if "recode" in s:
            col = col.replace(s["recode"]) if s.get("keep_unmapped", True) else col.map(s["recode"])
        if "dtype" in s:
            col = col.astype(s["dtype"])
        out[name] = col
    return out


# ---------------------------------------------------------------------------
# Mutations (MC3)
# ---------------------------------------------------------------------------
def read_maf(path: Path, patients: pd.Index, columns: list[str], chunksize: int = 500_000) -> pd.DataFrame:
    """Read the pan-cancer MAF in chunks, keeping only mutations of the given patients."""
    keep = set(patients)
    parts = [chunk[to_patient(chunk["Tumor_Sample_Barcode"]).isin(keep).to_numpy()]
             for chunk in pd.read_csv(path, sep="\t", usecols=columns, chunksize=chunksize, low_memory=False)]
    return pd.concat(parts, ignore_index=True)


def passes_filter(filter_labels: pd.Series, allowed: set[str]) -> pd.Series:
    """True when every comma-separated FILTER label is allowed."""
    return filter_labels.fillna("").str.split(",").apply(lambda labels: set(labels) <= set(allowed))


def mutation_matrix(maf: pd.DataFrame, profiled: pd.Index) -> pd.DataFrame:
    """Patients x genes, 1 if the gene carries at least one mutation; profiled patients without any = 0."""
    m = (maf.assign(**{PATIENT: to_patient(maf["Tumor_Sample_Barcode"]).to_numpy()}, value=1)
            .pivot_table(index=PATIENT, columns="Hugo_Symbol", values="value", aggfunc="max"))
    return m.reindex(pd.Index(profiled, name=PATIENT)).fillna(0).astype(int)


def mutation_frequencies(matrix: pd.DataFrame, genes: list[str], min_freq: float) -> pd.DataFrame:
    """Number and fraction of profiled patients mutated in each gene, and whether it passes `min_freq`."""
    m = matrix.reindex(columns=genes).fillna(0)
    freq = pd.DataFrame({"n_mutated": m.sum().astype(int), "freq": m.mean()})
    freq["kept"] = freq["freq"] >= min_freq
    return freq.sort_values("n_mutated", ascending=False)


def cooccurrence(binary: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Fisher test for every pair of binary columns, with BH-adjusted q-values."""
    rows = []
    for a, b in combinations(binary.columns, 2):
        table = pd.crosstab(binary[a].astype(bool), binary[b].astype(bool)).reindex(
            index=[False, True], columns=[False, True], fill_value=0)
        odds, p = fisher_exact(table)
        rows.append({"gene_a": a, "gene_b": b, "both": int(table.loc[True, True]), "odds_ratio": odds, "p": p})
    pairs = pd.DataFrame(rows)
    pairs["q"] = multipletests(pairs["p"], method="fdr_bh")[1]
    pairs["pattern"] = np.where(pairs["q"] >= alpha, "-", np.where(pairs["odds_ratio"] > 1, "co-occur", "exclusive"))
    return pairs.sort_values("q")


# ---------------------------------------------------------------------------
# Pathways and arm-level copy number
# ---------------------------------------------------------------------------
def snake(name: str) -> str:
    """'RTK RAS' -> 'rtk_ras', 'TGF-Beta' -> 'tgf_beta'."""
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def arm_calls(arms_raw: pd.DataFrame, arm_col: str, calls: dict[str, int]) -> pd.DataFrame:
    """Samples x arms with calls converted to numbers (gain 1, loss -1, unchanged 0)."""
    samples = [c for c in arms_raw.columns if c.startswith("TCGA-")]
    out = (arms_raw.set_index(arm_col)[samples].T
           .replace(calls).apply(pd.to_numeric, errors="coerce"))
    return out.rename_axis("sample").reset_index()


def arm_frequencies(arms: pd.DataFrame) -> pd.DataFrame:
    """Fraction of patients with a gain and with a loss of each arm."""
    return pd.DataFrame({"gain": (arms == 1).mean(), "loss": (arms == -1).mean()})


def arm_events(arms: pd.DataFrame, min_freq: float, required: list[str] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Binary arm-level events (`aneu_<arm>_<gain|loss>`) present in at least `min_freq` of patients.

    Events listed in `required` (e.g. '16q_loss') are kept whatever their frequency.
    Returns the event table (NaN where the arm was not called) and the event frequencies.
    """
    freq = arm_frequencies(arms)
    events = pd.concat([
        freq["gain"].rename("freq").to_frame().assign(direction="gain", value=1),
        freq["loss"].rename("freq").to_frame().assign(direction="loss", value=-1),
    ]).rename_axis("arm").reset_index()
    events["event"] = events["arm"] + "_" + events["direction"]
    events["kept"] = (events["freq"] >= min_freq) | events["event"].isin(list(required))
    kept = events[events["kept"]].sort_values("freq", ascending=False)
    table = pd.DataFrame({f"aneu_{r.event}": (arms[r.arm] == r.value).astype(float).where(arms[r.arm].notna())
                          for r in kept.itertuples()}, index=arms.index)
    return table, events


# ---------------------------------------------------------------------------
# Gene signatures
# ---------------------------------------------------------------------------
def map_aliases(signatures: dict[str, list[str]], aliases: dict[str, str]) -> dict[str, list[str]]:
    """Rename genes to the symbols used in the expression matrix."""
    return {name: [aliases.get(g, g) for g in genes] for name, genes in signatures.items()}


def ssgsea_running_sum(expr: pd.Series, genes: list[str], alpha: float = 0.25):
    """ssGSEA running sum of one sample (Barbie et al. 2009, as in gseapy with rank normalisation).

    Genes are ranked within the sample (most expressed first); the walk steps up at signature
    genes (weighted by rank**alpha) and down at the others. Returns the ranked gene names, the
    hit mask, the running sum and the enrichment score (the area under the running sum).
    """
    ranked = expr.dropna().sort_values(ascending=False)
    weight = ranked.rank().values ** alpha
    hits = ranked.index.isin(genes)
    step_up = np.cumsum(np.where(hits, weight, 0)) / weight[hits].sum()
    step_down = np.cumsum(~hits) / (~hits).sum()
    running = step_up - step_down
    return ranked.index, hits, running, running.sum()


def median_mad_scale(scores: pd.DataFrame) -> pd.DataFrame:
    """Centre each column on its median and divide by its median absolute deviation."""
    centred = scores - scores.median()
    return centred / centred.abs().median()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def make_label(column: str, labels: dict[str, str]) -> str | None:
    """Readable name of a molecular variable; mut_ and aneu_ columns are named automatically."""
    if column in labels:
        return labels[column]
    if column.startswith("mut_"):
        return f"{column[4:]} mutation"
    if column.startswith("aneu_"):
        return column[5:].replace("_", " ")
    return None


def load_labels(path: Path) -> dict[str, str]:
    """Readable variable names from configs/variable_labels.yaml."""
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)["labels"]


def figure_label(column: str, labels: dict[str, str]) -> str:
    """Short label for figures: readable name without the source in parentheses ('X (CIBERSORT)' -> 'X')."""
    import re

    return re.sub(r"\s*\([^)]*\)", "", make_label(column, labels) or column).strip()


# ---------------------------------------------------------------------------
# Redundancy
# ---------------------------------------------------------------------------
def continuous_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns with more than two distinct values."""
    num = df.select_dtypes("number")
    return num.columns[num.nunique() > 2].tolist()


def correlated_pairs(df: pd.DataFrame, columns: list[str], threshold: float) -> pd.DataFrame:
    """Pairs of columns with |Spearman rho| >= threshold, strongest first."""
    corr = df[columns].corr(method="spearman")
    lower = corr.where(np.tril(np.ones(corr.shape, dtype=bool), k=-1)).stack()
    pairs = lower.rename("rho").reset_index()
    pairs.columns = ["var1", "var2", "rho"]
    pairs = pairs[pairs["rho"].abs() >= threshold].sort_values("rho", key=abs, ascending=False)
    n = df[columns].notna().sum()
    return pairs.assign(n_var1=pairs["var1"].map(n), n_var2=pairs["var2"].map(n)).reset_index(drop=True)


def select_redundant(df: pd.DataFrame, columns: list[str], threshold: float,
                     keep_pairs: list[list[str]] = (), protected: list[str] = ()) -> tuple[dict[str, str], pd.DataFrame]:
    """Choose one variable to drop from each redundant pair (|Spearman rho| >= threshold).

    Pairs are processed from the strongest to the weakest correlation:
    1. pairs listed in `keep_pairs` keep both variables (they measure different biology);
    2. a pair is already resolved if one of its variables was dropped for a stronger pair;
    3. variables in `protected` (pre-specified in the analysis plan) are never dropped: if one
       variable of the pair is protected the other is dropped, if both are, both are kept;
    4. otherwise the variable available for fewer patients is dropped;
    5. on a tie, the variable with the higher mean |rho| with all other `columns`
       (the more redundant overall) is dropped; a remaining tie drops the later name alphabetically.

    Returns {dropped: kept} and a log with the decision and reason for every pair.
    """
    corr = df[columns].corr(method="spearman").abs()
    mean_abs = corr.where(~np.eye(len(corr), dtype=bool)).mean()
    n = df[columns].notna().sum()
    pairs = correlated_pairs(df, columns, threshold)
    kept_set = {frozenset(p) for p in keep_pairs}
    protected = set(protected)

    drop: dict[str, str] = {}
    decisions, reasons = [], []
    for a, b in zip(pairs["var1"], pairs["var2"]):
        if frozenset((a, b)) in kept_set:
            decisions.append("kept"); reasons.append("kept on purpose (keep_pairs)")
            continue
        if a in drop or b in drop:
            gone = a if a in drop else b
            decisions.append("resolved"); reasons.append(f"{gone} already dropped")
            continue
        if a in protected and b in protected:
            decisions.append("kept"); reasons.append("both pre-specified (protected)")
            continue
        if a in protected or b in protected:
            loser, winner = (b, a) if a in protected else (a, b)
            reason = f"{winner} pre-specified (protected)"
        elif n[a] != n[b]:
            loser, winner = (a, b) if n[a] < n[b] else (b, a)
            reason = f"fewer patients ({n[loser]} vs {n[winner]})"
        elif not np.isclose(mean_abs[a], mean_abs[b]):
            loser, winner = (a, b) if mean_abs[a] > mean_abs[b] else (b, a)
            reason = f"same patients; higher mean |rho| ({mean_abs[loser]:.2f} vs {mean_abs[winner]:.2f})"
        else:
            loser, winner = sorted((a, b))[::-1]
            reason = "full tie; alphabetical"
        drop[loser] = winner
        decisions.append(f"drop {loser}"); reasons.append(reason)
    log = pairs.assign(decision=decisions, reason=reasons)
    return drop, log
