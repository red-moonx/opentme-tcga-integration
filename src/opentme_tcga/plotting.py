"""Shared colours so that the same groups look the same in every notebook."""
from __future__ import annotations

# Tertiary lymphoid structures: grey for none, light and dark green by maturity.
TLS_GROUP_COLORS = {"No TLS": "#8a8984", "Immature only": "#9ccfc0", "Mature": "#1b5e57"}
TLS_MATURITY_COLORS = {"Immature": TLS_GROUP_COLORS["Immature only"], "Mature": TLS_GROUP_COLORS["Mature"]}

# Spatial immune phenotypes (hot tumours).
PHENOTYPE_COLORS = {"Inflamed": "#e34948", "Excluded": "#2a78d6"}
HOT_COLD_COLORS = {"Cold": "#2ca02c", "Hot": "#ff7f0e"}
# Median rules (inflamed / excluded / desert), used to compare phenotype definitions.
MEDIAN_RULE_COLORS = {"inflamed": "#e34948", "excluded": "#2a78d6", "desert": "#8a8984"}

# Subtype colours are set per indication (tcga_molecular.subtypes.colors); see category_colors.
def category_colors(categories, known: dict[str, str] | None = None, palette: str = "tab10") -> dict[str, str]:
    """Colour per category: from `known` when available (e.g. the subtype colours of the indication file),
    otherwise from a matplotlib palette, so any number of categories gets distinct colours."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    known = known or {}
    cmap = colormaps[palette]
    others = [c for c in categories if c not in known]
    return {c: known.get(c) or to_hex(cmap(others.index(c) % cmap.N)) for c in categories}
