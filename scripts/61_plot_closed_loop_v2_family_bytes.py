#!/usr/bin/env python3
"""Publication figure from the immutable repaired-v2 secondary analysis only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/study_b/closed_loop_confirmation_v2/confirmation/secondary_analysis.json"
OUTPUT = ROOT / "docs/figures"
STEM = "bces_v2_family_byte_reduction"
FAMILIES = (
    ("on_ramp_merge", "On-ramp merge"),
    ("pedestrian_crossing", "Pedestrian crossing"),
    ("unprotected_crossing", "Unprotected crossing"),
    ("straight_lead_braking", "Straight lead braking"),
)
METHODS = (
    ("surface", "BCES", "#007f86", 0.17),
    ("scalar_ttl", "Learned scalar TTL", "#ba5b21", -0.17),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    outputs = [OUTPUT / f"{STEM}.{suffix}" for suffix in ("png", "pdf")]
    sidecar = OUTPUT / f"{STEM}_manifest.json"
    if any(path.exists() for path in (*outputs, sidecar)):
        raise FileExistsError("figure artifact is immutable; choose a new version")
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    if data["status"] != "COMPLETE" or set(data["families"]) != {key for key, _ in FAMILIES}:
        raise RuntimeError("unexpected repaired-v2 analysis contract")
    if data["weak_straight_braking_reported"] is not True:
        raise RuntimeError("unfavorable family must remain reported")
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    fig.subplots_adjust(left=0.24, right=0.98, top=0.88, bottom=0.23)
    for method, label, color, shift in METHODS:
        centers, lowers, uppers = [], [], []
        for family, _ in FAMILIES:
            row = data["families"][family][method]
            if row["scenarios"] != 300 or row["adverse_scenarios"] != 0:
                raise RuntimeError("family sample count or adverse status changed")
            point = 100 * row["mean_byte_reduction_vs_periodic"]
            lo, hi = (100 * value for value in row["bootstrap_95pct_ci"])
            if not lo <= point <= hi:
                raise RuntimeError("point outside family interval")
            centers.append(point)
            lowers.append(point - lo)
            uppers.append(hi - point)
        y = [index + shift for index in range(len(FAMILIES))]
        ax.errorbar(centers, y, xerr=[lowers, uppers], fmt="o", color=color,
                    ecolor=color, elinewidth=1.8, capsize=3, markersize=7,
                    label=label, zorder=3)
        for center, height in zip(centers, y):
            ax.annotate(f"{center:.1f}%", (center, height), xytext=(0, -14),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5,
                        color=color, fontweight="bold")
    ax.axvline(0, color="#666666", linewidth=1, linestyle="--", zorder=1)
    ax.set_xlim(-10, 102)
    ax.set_ylim(-0.55, 3.55)
    ax.set_yticks(range(4), [label for _, label in FAMILIES])
    ax.invert_yaxis()
    ax.set_xlabel("Mean modeled byte reduction versus periodic payload (%)")
    ax.set_title("Repaired closed-loop communication by scenario family", loc="left", weight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(0.04, 0.045,
             "95% whole-scenario bootstrap intervals; 300 scenarios per family.\n"
             "All four policies had zero registered adverse episodes; intervals describe bytes, not safety.",
             fontsize=8, color="#444444", linespacing=1.5)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest = {
        "status": "REPAIRED_V2_FAMILY_BYTES_DESCRIPTIVE_FIGURE",
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(SOURCE),
        "script_sha256": sha256(Path(__file__)),
        "output_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in outputs},
        "interval": "95% whole-scenario bootstrap; descriptive family means",
        "independent_unit": "scenario",
        "per_family_scenarios": 300,
        "caution": "no BCES-versus-TTL relative safety claim; straight braking remains weak",
    }
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
