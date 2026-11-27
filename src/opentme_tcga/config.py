"""Configuration loading.

Every notebook starts with::

    from opentme_tcga.config import load_config
    cfg = load_config("breast")

which reads an optional ``.env`` file at the repository root (git-ignored, for
local settings such as ``OPENTME_TCGA_DATA_DIR``), then merges ``configs/project.yaml``, ``configs/analysis.yaml`` and
``configs/indications/<indication>.yaml`` (later files override earlier ones) and
resolves all paths. Nothing in the code depends on a particular cancer type:
running the pipeline on another indication only requires a new indication file.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

DATA_DIR_ENV = "OPENTME_TCGA_DATA_DIR"
CONFIG_FILES = ("project.yaml", "analysis.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_repo_root(start: Path | None = None) -> Path:
    """First parent of `start` (default: current directory) holding pyproject.toml and configs/."""
    candidates = [Path(start or Path.cwd()).resolve(), Path(__file__).resolve()]
    for origin in candidates:
        for folder in (origin, *origin.parents):
            if (folder / "pyproject.toml").is_file() and (folder / "configs").is_dir():
                return folder
    raise FileNotFoundError("Repository root not found (no pyproject.toml next to a configs/ folder).")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`; nested dicts are merged, other values replaced."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_dotenv(path: Path) -> dict[str, str]:
    """Read KEY=VALUE lines of a .env file into os.environ (existing variables win).

    Blank lines and lines starting with # are ignored; surrounding quotes are removed.
    Returns the variables found in the file.
    """
    found: dict[str, str] = {}
    if not Path(path).is_file():
        return found
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        found[key] = value
        os.environ.setdefault(key, value)
    return found


def _read_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def indication_fields(cfg: dict) -> dict[str, str]:
    """Values usable in path and file templates: {indication}, {opentme}, {tcga_code}, {tcga_code_lower}..."""
    indication = cfg["indication"]
    fields = {"indication": indication["name"], **{k: str(v) for k, v in indication.items()}}
    if "tcga_code" in indication:
        fields["tcga_code_lower"] = str(indication["tcga_code"]).lower()
    return fields


def resolve_paths(cfg: dict, root: Path) -> dict[str, Path]:
    """Expand the templates of `paths` and `files` into absolute paths.

    Templates may refer to `{data}`, to previously defined paths and to the
    indication fields (`{indication}` is the short name, `{opentme}` the OpenTME name).
    """
    data_dir = Path(os.environ.get(DATA_DIR_ENV) or cfg["paths"]["data"]).expanduser()
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    fields = {**indication_fields(cfg), "data": str(data_dir)}

    paths: dict[str, Path] = {"root": root, "configs": root / "configs", "data": data_dir}
    for name, template in cfg["paths"].items():
        if name == "data":
            continue
        path = Path(template.format(**fields))
        paths[name] = path if path.is_absolute() else root / path
        fields[name] = str(paths[name])
    for name, template in cfg.get("files", {}).items():
        paths[name] = Path(template.format(**fields))
    return paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def available_indications(root: Path | None = None) -> list[str]:
    """Names of the indication files in configs/indications/."""
    root = root or find_repo_root()
    return sorted(p.stem for p in (root / "configs" / "indications").glob("*.yaml"))


def load_config(indication: str, root: Path | None = None) -> dict:
    """Merged configuration for one indication, with resolved paths under ``cfg["paths"]``."""
    root = Path(root) if root else find_repo_root()
    load_dotenv(root / ".env")
    config_dir = root / "configs"
    indication_file = config_dir / "indications" / f"{indication}.yaml"
    if not indication_file.is_file():
        raise FileNotFoundError(
            f"No configuration for indication '{indication}'. "
            f"Available: {available_indications(root)}"
        )

    cfg: dict = {}
    for name in CONFIG_FILES:
        cfg = deep_merge(cfg, _read_yaml(config_dir / name))
    cfg = deep_merge(cfg, _read_yaml(indication_file))

    if cfg.get("indication", {}).get("name") != indication:
        raise ValueError(f"{indication_file.name}: indication.name must be '{indication}'.")
    cfg["paths"] = resolve_paths(cfg, root)
    return cfg


def source_file(cfg: dict, name: str, section: str = "tcga_molecular", folder: str = "raw_tcga") -> dict:
    """Specification of a raw data source with its file path and URL resolved.

    Returns the source entry of ``cfg[section]["sources"][name]`` (or ``cfg[section][name]``
    for indication-specific sources) with ``path`` and, if any, ``url`` filled in.
    """
    block = cfg[section]
    spec = dict(block.get("sources", {}).get(name) or block[name])
    fields = indication_fields(cfg)
    spec["path"] = cfg["paths"][folder] / spec["file"].format(**fields)
    if "url" in spec:
        spec["url"] = spec["url"].format(**fields)
    return spec


def ensure_dirs(cfg: dict, names: tuple[str, ...] = ("interim", "processed", "results", "figures")) -> None:
    """Create the output folders listed in `names` if they do not exist."""
    for name in names:
        cfg["paths"][name].mkdir(parents=True, exist_ok=True)
