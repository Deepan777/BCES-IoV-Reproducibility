#!/usr/bin/env python3
"""Development-only screen for vacuity of a new-action zero cost lower bound.

It uses label-world reference costs for mathematical diagnosis only, never as
receiver inputs. It does not construct a valid uniform envelope or certify BCES.
"""

from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.oracle.planner import PlannerConfig
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import ControlledDecisionPlanner, route_for
from bces.utils.hashing import sha256_file
from scripts import __path__ as _scripts_path  # noqa: F401

# Import the previous diagnostic's identical action enumeration without editing it.
import importlib.util


def load_candidates():
    path = ROOT / "scripts/31_audit_regret_premises.py"
    spec = importlib.util.spec_from_file_location("premise_audit_frozen", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.candidates


def main() -> None:
    output = ROOT / "outputs/study_b/changing_action_bound_screen_v1.json"
    if output.exists():
        raise FileExistsError(output)
    inputs = ROOT / "outputs/study_b/controlled_development_v2"
    manifest_path = inputs / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs/evaluation/study_b_controlled_v2.yaml").read_text(encoding="utf-8"))
    tolerance = float(yaml.safe_load((ROOT / "configs/oracle/validity_v1.yaml").read_text(encoding="utf-8"))["max_cost_regret"])
    planner = ControlledDecisionPlanner(PlannerConfig.from_yaml(ROOT / config["planner"]))
    costs_for = load_candidates()
    counts = Counter()
    by_family = {}
    references_costs = []
    for path in sorted(inputs.glob("validation_*.json.gz")):
        if sha256_file(path) != manifest["artifact_sha256"][path.name]:
            raise RuntimeError("development input changed: " + path.name)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            scene = json.load(handle)
        scene_counts = Counter()
        references = {row["reference_id"]: row for row in scene["references"]}
        cache = {}
        for point in scene["points"]:
            reference = references[point["reference_id"]]
            variant = str(point["receiver_drift_variant"])
            ref_time = str(reference["reference_timestamp_ms"])
            now_time = str(point["current_timestamp_ms"])
            frames = scene["traces"][variant]
            if ref_time not in frames or now_time not in frames:
                continue
            key = (reference["reference_id"], variant)
            if key not in cache:
                frame = frames[ref_time]
                ego = ego_from_sample(frame["receiver"])
                route = route_for(ego, reference["behavior"], planner)
                reference_costs = costs_for(planner, ego, route, frame["objects"], int(ref_time))
                if not reference_costs:
                    continue
                chosen = min(reference_costs, key=lambda action: (
                    reference_costs[action].collision,
                    reference_costs[action].ttc,
                    reference_costs[action].total,
                    action.split(":")[0], float(action.split(":")[1]),
                ))
                cache[key] = (set(reference_costs), chosen, reference_costs[chosen].total)
                references_costs.append(reference_costs[chosen].total)
            reference_set, chosen, chosen_cost = cache[key]
            current = frames[now_time]
            ego = ego_from_sample(current["receiver"])
            route = route_for(ego, reference["behavior"], planner)
            feasible = {
                f"{maneuver}:{acceleration:g}"
                for maneuver in planner.config.lateral_maneuvers
                for acceleration in planner.config.longitudinal_accelerations_mps2
                if planner._candidate(ego, route, acceleration, maneuver) is not None
            }
            scene_counts["points"] += 1
            if chosen not in feasible:
                scene_counts["cached_action_infeasible"] += 1
            new = feasible - reference_set
            if new:
                scene_counts["new_action_points"] += 1
                if chosen in feasible:
                    scene_counts["new_action_and_cached_feasible"] += 1
                    # Even with zero error in chosen cost, L_new=0 only gives
                    # a bound of at least c_chosen. This is optimistic, not a certificate.
                    if chosen_cost > tolerance:
                        scene_counts["generic_zero_lower_bound_vacuous_even_with_zero_chosen_error"] += 1
        counts.update(scene_counts)
        by_family.setdefault(scene["family"], Counter()).update(scene_counts)
        print(f"checked {path.name}", flush=True)
    if not references_costs:
        raise RuntimeError("no reference costs audited")
    result = {
        "status": "DEVELOPMENT_ONLY_MATHEMATICAL_VACUITY_SCREEN_NOT_CERTIFICATION",
        "scene_count": len(list(inputs.glob("validation_*.json.gz"))),
        "max_cost_regret": tolerance,
        "counts": dict(counts),
        "by_family": {family: dict(value) for family, value in sorted(by_family.items())},
        "reference_evaluations": len(references_costs),
        "reference_chosen_cost_above_tolerance": sum(value > tolerance for value in references_costs),
        "input_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": sha256_file(Path(__file__)),
        "interpretation": "Zero lower bound for a newly feasible nonnegative-cost action is often too loose; no interval validity or neural-surface containment is proved.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("counts", "reference_evaluations", "reference_chosen_cost_above_tolerance")}, indent=2))


if __name__ == "__main__":
    main()
