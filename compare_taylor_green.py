"""Compare Taylor-Green diagnostics against a reference .dat file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ours",
        type=Path,
        default=Path("output_cpp/taylor_green_diagnostics.csv"),
        help="CSV produced by taylor_green_flups",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("spectral_Re1600_512.dat"),
        help="reference whitespace .dat with columns: t E epsilon enstrophy",
    )
    parser.add_argument("--re", type=float, default=1600.0, help="Reynolds number for older CSVs without enstrophy")
    parser.add_argument("--output", type=Path, default=Path("output_cpp/taylor_green_reference_compare.png"))
    parser.add_argument("--title", default="Taylor-Green vortex diagnostics")
    return parser.parse_args()


def load_reference(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"{path} must have at least four columns: t E epsilon enstrophy")
    return {
        "t": data[:, 0],
        "kinetic_energy": data[:, 1],
        "dissipation": data[:, 2],
        "enstrophy": data[:, 3],
    }


def load_ours(path: Path, reynolds: float) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = set(data.dtype.names or ())
    required = {"t", "kinetic_energy", "dissipation"}
    missing = required - names
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    result = {
        "t": np.asarray(data["t"]),
        "kinetic_energy": np.asarray(data["kinetic_energy"]),
        "dissipation": np.asarray(data["dissipation"]),
    }
    if "enstrophy" in names:
        result["enstrophy"] = np.asarray(data["enstrophy"])
    else:
        nu = 1.0 / reynolds
        result["enstrophy"] = result["dissipation"] / (2.0 * nu)
    return result


def main() -> None:
    args = parse_args()
    ref = load_reference(args.reference)
    ours = load_ours(args.ours, args.re)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    series = [
        ("kinetic_energy", "Kinetic energy", "E"),
        ("dissipation", "Dissipation rate", "epsilon"),
        ("enstrophy", "Enstrophy", "0.5 <|omega|^2>"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes, series):
        ax.plot(ref["t"], ref[key], color="black", linewidth=1.8, label="reference 512^3")
        ax.plot(ours["t"], ours[key], color="#1f77b4", linewidth=1.6, linestyle="--", label="ours")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("t")
    fig.suptitle(args.title)
    fig.savefig(args.output, dpi=180)
    print(f"saved comparison plot: {args.output}")


if __name__ == "__main__":
    main()
