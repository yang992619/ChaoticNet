#!/usr/bin/env python3
"""Report-figure quality gate: provenance, numeric claims, and TeX references."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
import baseline_distmass  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    required_images = [
        ROOT / "data/figures/fig_pendulum_schematic.png",
        ROOT / "data/sim/figures/fig_energy_sweep.png",
        ROOT / "data/sim/figures/fig_butterfly_distmass_large.png",
        ROOT / "data/figures/three_way_compare.png",
        ROOT / "data/real_validation/figures/real_overlay_1448.png",
        ROOT / "data/real_validation/figures/multi_horizon_vs_angle.png",
        ROOT / "data/real_validation/figures/error_decomposition.png",
    ]
    for image in required_images:
        require(image.exists(), f"figure asset exists: {image.relative_to(ROOT)}")
        width, height = Image.open(image).size
        require(width >= 1200 and height >= 700,
                f"figure asset is print-resolution: {image.name} ({width}x{height})")

    for tex_name in ("main.tex", "main_匿名投稿版.tex"):
        tex = (ROOT / "paper" / tex_name).read_text(encoding="utf-8")
        require("horizon_compare_50" not in tex and "fig:horizon-50" not in tex,
                f"{tex_name} excludes legacy mixed-source Figure 5")
        for label in ("fig:pendulum", "fig:energy", "fig:butterfly", "fig:three-way",
                      "fig:real-overlay", "fig:real-horizon", "fig:error-decomp"):
            require(label in tex, f"{tex_name} retains {label}")

    y_a = baseline_distmass.simulate(np.deg2rad(60.0), 0.0, t_end=12, fps=120)
    y_b = baseline_distmass.simulate(np.deg2rad(60.1), 0.0, t_end=12, fps=120)
    delta = np.maximum(
        np.abs(np.rad2deg(y_a["th1"] - y_b["th1"])),
        np.abs(np.rad2deg(y_a["th2"] - y_b["th2"])),
    )
    first = y_a["t"].iloc[np.flatnonzero(delta.to_numpy() > 10)[0]]
    require(abs(first - 7.4) < 0.02, f"Figure 3 threshold is reproducible: {first:.2f} s")

    fair = pd.read_csv(ROOT / "data/fair_compare/four_way_horizons.csv")
    fair_subset = fair[(fair["esn"] < 9.9) & (fair["pinn"] < 9.9) & (fair["hybrid"] < 9.9)]
    require(len(fair_subset) == 32, "Figure 4 is explicitly a fixed N=32 visualization")

    horizon = pd.read_csv(ROOT / "data/real_validation/summary_multi.csv")
    require(len(horizon) == 29, "Figure 7 uses the current 29-run real-validation summary")
    require({"rel_deg", "ESN_h", "PINN_h", "Hybrid_h"}.issubset(horizon.columns),
            "Figure 7 source columns are complete")

    decomp = pd.read_csv(ROOT / "data/real_validation/error_decomposition.csv")
    require(len(decomp) == 29, "Figure 8 uses the current 29-run decomposition table")
    require({"h_real_PINN", "h_real_Hybrid", "h_damp_PINN", "h_damp_Hybrid",
             "h_model_PINN", "h_model_Hybrid"}.issubset(decomp.columns),
            "Figure 8 source columns are complete")

    print("\nAll report-figure quality gates passed.")


if __name__ == "__main__":
    main()
