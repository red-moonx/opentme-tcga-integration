"""Tests for opentme_tcga.plotting."""
from opentme_tcga.plotting import category_colors


def test_category_colors_uses_known_then_palette():
    known = {"LumA": "#1f77b4"}
    colors = category_colors(["LumA", "X", "Y"], known)
    assert colors["LumA"] == "#1f77b4"
    assert colors["X"] != colors["Y"] and colors["X"].startswith("#")


def test_category_colors_without_known_colours():
    colors = category_colors([f"S{i}" for i in range(6)])
    assert len(set(colors.values())) == 6
