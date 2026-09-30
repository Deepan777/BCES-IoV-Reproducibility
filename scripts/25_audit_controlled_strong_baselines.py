#!/usr/bin/env python3
"""Development-only matched-coverage audit against simple dynamic baselines.

This audit never selects a deployable operating point and never reads calibration
or confirmation outcomes.  Its purpose is to test whether the learned surface's
ranking advantage survives comparison with age and receiver-state-change rules.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.study_b import feature_arrays, load_development, selective_metrics
from bces.models.decision_surface import DecisionSurfaceNet, membership_score
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic


COVERAGES = (0.50, 0.75, 0.80)
BOOTSTRAP_SEED = 250921
BOOTSTRAP_REPLICATES = 5000


def deterministic_rank_subset(scores: np.ndarray, identities: np.ndarray, fraction: float) -> np.ndarray:
    """Select exactly the requested fraction with label-independent tie breaking."""
    scores = np.asarray(scores, dtype=float)
    identities = np.asarray(identities, dtype=str)
    if scores.ndim != 1 or identities.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("finite one-dimensional scores and matching identities required")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    count = max(1, int(len(scores) * fraction))
    tie = np.fromiter(
        (int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") for value in identities),
        dtype=np.uint64,
        count=len(identities),
    )
    order = np.lexsort((tie, -scores))
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:count]] = True
    return selected


def paired_difference(
    valid: np.ndarray,
    learned: np.ndarray,
    comparator: np.ndarray,
    scenarios: np.ndarray,
) -> dict:
    """Scenario-paired descriptive interval for learned minus comparator UAR."""
    valid = np.asarray(valid, dtype=bool)
    scenarios = np.asarray(scenarios)
    names = np.unique(scenarios)
    counts = np.empty((2, len(names), 2), dtype=np.int64)
    for method, accepted in enumerate((learned, comparator)):
        accepted = np.asarray(accepted, dtype=bool)
        for index, name in enumerate(names):
            member = scenarios == name
            counts[method, index] = (accepted[member].sum(), (accepted & ~valid)[member].sum())
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(names), size=(BOOTSTRAP_REPLICATES, len(names))
    )
    sampled = counts[:, draws, :].sum(axis=2)
    usable = (sampled[:, :, 0] > 0).all(axis=0)
    rates = sampled[:, usable, 1] / sampled[:, usable, 0]
    difference = rates[0] - rates[1]
    totals = counts.sum(axis=1)
    observed = totals[:, 1] / totals[:, 0]
    return {
        "learned_minus_comparator_uar": float(observed[0] - observed[1]),
        "paired_scenario_bootstrap_interval_descriptive": np.quantile(
            difference, (0.025, 0.975)
        ).tolist(),
        "bootstrap_replicates_used": int(usable.sum()),
        "scenario_count": int(len(names)),
        "confirmatory_significance_claim": False,
    }


def load_surface_scores(manifest: dict, model_root: Path, arrays: dict) -> tuple[dict, dict]:
    """Recompute all surface scores and verify every linked checkpoint digest."""
    scores: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    torch.set_num_threads(2)
    for view in ("frozen_inputs", "reference_margins"):
        reference = arrays["base_reference"] if view == "frozen_inputs" else arrays["margin_reference"]
        group = manifest["runs"][f"{view}:surface"]
        for run in group["runs"]:
            seed = int(run["seed"])
            path = model_root / f"{view}_surface_{seed}.pt"
            digest = sha256_file(path)
            if digest != run["checkpoint_sha256"] or digest != manifest["artifacts"][path.name]:
                raise ValueError("checkpoint hash mismatch")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = DecisionSurfaceNet(checkpoint["feature_count"], scalar=False)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            normalized = ((reference - checkpoint["mean"].numpy()) / checkpoint["std"].numpy()).clip(
                -10, 10
            ).astype(np.float32)
            with torch.no_grad():
                offsets, logits = model(torch.from_numpy(normalized))
                index = torch.from_numpy(arrays["reference_index"])
                score = membership_score(
                    offsets[index], logits[index], torch.from_numpy(arrays["drift"]), scalar=False
                ).numpy()
            scores[f"{view}:{seed}"] = score
            digests[path.name] = digest
    return scores, digests


def main() -> int:
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("clean committed audit revision required")
    base = ROOT / "outputs" / "study_b"
    data_root = base / "controlled_development_v2"
    model_root = base / "decision_surface_controlled_v2"
    destination = base / "controlled_strong_baseline_audit_v1.json"
    if destination.exists():
        raise FileExistsError("completed baseline audit is immutable")
    manifest_path = model_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["data_manifest_sha256"] != sha256_file(data_root / "run_manifest.json"):
        raise ValueError("training/data mismatch")
    if manifest["scope"] != "development_only" or manifest["original_test_accessed"]:
        raise PermissionError("development-only source required")

    references, points, _ = load_development(data_root)
    points = [point for point in points if point["split"] == "validation"]
    references = {key: value for key, value in references.items() if value["split"] == "validation"}
    arrays = feature_arrays(references, points)
    valid = np.asarray([point["valid"] for point in points], dtype=bool)
    scenarios = np.asarray([point["scenario_id"] for point in points])
    identities = np.asarray([point["point_id"] for point in points])
    drift = arrays["drift"].astype(float)

    comparator_scores = {
        "age_only": -np.asarray([point["cache_age_s"] for point in points], dtype=float),
        "state_change_linf": -np.max(np.abs(drift[:, :6]), axis=1),
        "state_change_l2": -np.linalg.norm(drift[:, :6], axis=1),
        "joint_age_state_linf": -np.max(np.abs(drift), axis=1),
        "joint_age_state_l2": -np.linalg.norm(drift, axis=1),
    }
    surface_scores, checkpoint_digests = load_surface_scores(manifest, model_root, arrays)

    comparisons = []
    for view in ("frozen_inputs", "reference_margins"):
        seeds = sorted(int(key.rsplit(":", 1)[1]) for key in surface_scores if key.startswith(view + ":"))
        for seed in seeds:
            learned_score = surface_scores[f"{view}:{seed}"]
            for coverage in COVERAGES:
                learned = deterministic_rank_subset(learned_score, identities, coverage)
                for name, comparator_score in comparator_scores.items():
                    comparator = deterministic_rank_subset(comparator_score, identities, coverage)
                    comparisons.append(
                        {
                            "view": view,
                            "seed": seed,
                            "comparator": name,
                            "requested_coverage": coverage,
                            "learned_surface": selective_metrics(valid, learned, scenarios),
                            "comparator_metrics": selective_metrics(valid, comparator, scenarios),
                            **paired_difference(valid, learned, comparator, scenarios),
                        }
                    )

    margin_ablation = []
    for seed in sorted(int(key.rsplit(":", 1)[1]) for key in surface_scores if key.startswith("frozen_inputs:")):
        for coverage in COVERAGES:
            margin = deterministic_rank_subset(surface_scores[f"reference_margins:{seed}"], identities, coverage)
            frozen = deterministic_rank_subset(surface_scores[f"frozen_inputs:{seed}"], identities, coverage)
            margin_ablation.append(
                {
                    "seed": seed,
                    "requested_coverage": coverage,
                    "margin_augmented": selective_metrics(valid, margin, scenarios),
                    "payload_only_reference": selective_metrics(valid, frozen, scenarios),
                    **paired_difference(valid, margin, frozen, scenarios),
                }
            )

    report = {
        "schema_version": 1,
        "scope": "validation_development_only",
        "status": "PASS",
        "scientific_success": False,
        "manuscript_allowed": False,
        "original_test_accessed": False,
        "calibration_or_confirmation_outcomes_accessed": False,
        "selection_contract": "fixed matched coverage with SHA-256 point-id tie breaking; no labels used",
        "coverages": list(COVERAGES),
        "comparisons": comparisons,
        "margin_feature_ablation": margin_ablation,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "training_manifest_sha256": sha256_file(manifest_path),
        "data_manifest_sha256": sha256_file(data_root / "run_manifest.json"),
        "checkpoint_sha256": checkpoint_digests,
        "limitations": [
            "development validation comparison, not independent confirmation",
            "matched-coverage ranking is descriptive and is not a deployable threshold",
            "bootstrap intervals are descriptive and unadjusted for repeated comparisons",
            "simple rules use normalized drift available at the receiver; they do not estimate decision regret",
            "behavior removal requires separately trained models and is not claimed by this audit",
        ],
        "git": state,
        "completed_utc": utc_now(),
    }
    write_json_atomic(destination, report)
    favorable = [
        row for row in comparisons
        if row["learned_minus_comparator_uar"] < 0
        and row["paired_scenario_bootstrap_interval_descriptive"][1] < 0
    ]
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(destination),
                "comparisons": len(comparisons),
                "descriptively_favorable_with_interval_below_zero": len(favorable),
                "confirmatory_significance_claim": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
