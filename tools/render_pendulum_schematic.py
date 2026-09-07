#!/usr/bin/env python3
"""Render the vector pendulum schematic at print resolution for the report."""

from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "diagrams" / "01_physics_model.svg"
OUT = ROOT / "data" / "figures" / "fig_pendulum_schematic.png"


def main() -> None:
    cairosvg.svg2png(url=str(SRC), write_to=str(OUT), output_width=2280, output_height=1980)
    print(f"rendered {SRC} -> {OUT} (2280x1980)")


if __name__ == "__main__":
    main()
