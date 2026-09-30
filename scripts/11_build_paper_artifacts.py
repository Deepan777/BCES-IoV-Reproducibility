#!/usr/bin/env python3
"""Generate traceable Phase-11 paper tables, figures, cases, and claims."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.artifacts import artifact_hashes, prohibited_claims
from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/evaluation/paper_artifacts_v1.yaml"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(), writer.writerows(rows)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _save_figure(fig: plt.Figure, output: Path, name: str, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("paper_artifacts_v1 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("paper artifacts require a clean committed revision")
    source_keys = ("phase8_gate", "surface_manifest", "scalar_manifest", "validity_mlp_manifest", "offline_manifest", "offline_decisions", "network_manifest", "geometry_audit", "object_uncertainty", "second_planner", "sumo_source_ablation", "phase9_gate", "sumo_manifest", "sumo_routes", "phase10_gate", "statistics")
    sources = {key: ROOT / config[key] for key in source_keys}
    source_hashes = {key: sha256_file(path) for key, path in sources.items()}
    for gate in ("phase8_gate", "phase9_gate", "phase10_gate"):
        if _load(sources[gate])["status"] != "PASS":
            raise RuntimeError(f"upstream gate failed: {gate}")
    offline, statistics = _load(sources["offline_manifest"]), _load(sources["statistics"])
    sumo, network = _load(sources["sumo_manifest"]), _load(sources["network_manifest"])
    geometry, uncertainty = _load(sources["geometry_audit"]), _load(sources["object_uncertainty"])
    second, source_ablation = _load(sources["second_planner"]), _load(sources["sumo_source_ablation"])
    decisions = _rows(sources["offline_decisions"])
    routes = _rows(sources["sumo_routes"])
    if sha256_file(sources["offline_decisions"]) != offline["raw_sha256"] or sha256_file(sources["sumo_routes"]) != sumo["raw_sha256"]:
        raise ValueError("paper input raw hash mismatch")
    output.mkdir(parents=True, exist_ok=False)

    operating_rows = []
    for name, value in offline["operating_point_results"].items():
        if "overall" not in value:
            continue
        row = value["overall"]
        operating_rows.append({"method": name, "accepted": row["accepted"], "coverage": row["coverage"], "unsafe_accepted": row["unsafe_accepted"], "unsafe_accept_rate": row["unsafe_accept_rate"], "unsafe_accept_upper_95": row["unsafe_accept_upper"], "valid_coverage": row["valid_coverage"], "target_met": row["target_met"], "risk_coverage_area": value.get("risk_coverage_area")})
    _csv(output / "table_offline_operating_points.csv", list(operating_rows[0]), operating_rows)

    matched_rows = []
    for name, value in offline["matched_coverage_results"].items():
        row = value["overall"]
        matched_rows.append({"method": name, "coverage_target": offline["matched_coverage_fraction"], "accepted": row["accepted"], "unsafe_accepted": row["unsafe_accepted"], "unsafe_accept_rate": row["unsafe_accept_rate"], "unsafe_accept_upper_95": row["unsafe_accept_upper"], "valid_coverage": row["valid_coverage"]})
    _csv(output / "table_matched_coverage.csv", list(matched_rows[0]), matched_rows)

    ablation_names = ("surface", "k8", "k12", "k16", "k24", "k32", "remove_behavior", "no_object_context", "no_unsafe_loss", "no_quantization_simulation", "no_calibration", "learned_scalar_ttl")
    ablation_rows = []
    for name in ablation_names:
        value = offline["operating_point_results"][name]["overall"]
        ablation_rows.append({"variant": name, "accepted": value["accepted"], "coverage": value["coverage"], "unsafe_accept_rate": value["unsafe_accept_rate"], "unsafe_accept_upper_95": value["unsafe_accept_upper"], "target_met": value["target_met"]})
    _csv(output / "table_ablations.csv", list(ablation_rows[0]), ablation_rows)

    sumo_rows = []
    for condition, methods in sumo["summaries"].items():
        for method, value in methods.items():
            sumo_rows.append({"condition": condition, "method": method, **value})
    _csv(output / "table_closed_loop_sumo.csv", list(sumo_rows[0]), sumo_rows)

    statistical_rows = []
    offline_ci = statistics["offline_matched_coverage"]["comparisons"]["surface_minus_learned_scalar_ttl"]["clustered_uar_difference"]
    statistical_rows.append({"endpoint": "matched_coverage_uar_surface_minus_scalar", "estimate": offline_ci["estimate_first_minus_second"], "lower_95": offline_ci["lower"], "upper_95": offline_ci["upper"], "unit_count": offline_ci["cluster_count"], "interpretation": "interval_crosses_zero" if offline_ci["upper"] >= 0 else "surface_lower"})
    for comparator in ("learned_scalar_ttl", "always_fresh"):
        value = statistics["sumo_paired_comparisons"]["clean"][f"surface_every_tick_minus_{comparator}"]["application_bytes"]["bootstrap_ci"]
        statistical_rows.append({"endpoint": f"clean_sumo_bytes_surface_minus_{comparator}", "estimate": value["estimate"], "lower_95": value["lower"], "upper_95": value["upper"], "unit_count": value["n_pairs"], "interpretation": "surface_more_bytes" if value["lower"] > 0 else "interval_includes_no_increase"})
    _csv(output / "table_primary_statistics.csv", list(statistical_rows[0]), statistical_rows)

    surface_resource, scalar_resource, mlp_resource = _load(sources["surface_manifest"]), _load(sources["scalar_manifest"]), _load(sources["validity_mlp_manifest"])
    resource_rows = [
        {"component": "SurfaceNet-Lite", "parameters": surface_resource["parameter_count"], "peak_cuda_bytes": surface_resource["peak_cuda_allocated_bytes"]},
        {"component": "learned_scalar_ttl", "parameters": scalar_resource["parameter_count"], "peak_cuda_bytes": scalar_resource["peak_cuda_allocated_bytes"]},
        {"component": "validity_mlp", "parameters": mlp_resource["parameter_count"], "peak_cuda_bytes": mlp_resource["peak_cuda_allocated_bytes"]},
        {"component": "phase9_closed_loop", "parameters": "NA", "peak_cuda_bytes": sumo["peak_cuda_allocated_bytes"]},
    ]
    _csv(output / "table_compute_resources.csv", list(resource_rows[0]), resource_rows)
    failure_rows = [{"category": key, "count": value} for key, value in statistics["failure_taxonomy"]["scientific"].items()]
    failure_rows.extend({"category": f"execution:{key}", "count": value} for key, value in statistics["failure_taxonomy"]["phase9_execution"].items())
    _csv(output / "table_failure_taxonomy.csv", ["category", "count"], failure_rows)

    colors = {"surface": "#2166ac", "learned_scalar_ttl": "#b2182b", "validity_mlp": "#4d9221"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    valid = np.asarray([row["valid"] for row in decisions], dtype=bool)
    for name in colors:
        score = np.asarray([row["scores"][name] for row in decisions])
        order = np.argsort(score, kind="stable")
        risk = np.cumsum(~valid[order]) / np.arange(1, len(valid) + 1)
        coverage = np.arange(1, len(valid) + 1) / len(valid)
        indices = np.linspace(0, len(valid) - 1, int(config["risk_curve_points"]), dtype=int)
        ax.plot(coverage[indices], risk[indices], label=name.replace("_", " "), color=colors[name], linewidth=2)
    ax.axhline(.05, color="black", linestyle="--", linewidth=1, label="registered UAR target")
    ax.set(xlabel="Coverage", ylabel="Unsafe-accept rate", title="Locked-test risk–coverage curves", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    _save_figure(fig, output, "figure_risk_coverage", int(config["figure_dpi"]))

    selected = [row for row in operating_rows if row["method"] in {"surface", "learned_scalar_ttl", "validity_mlp", "fixed_ttl", "aoii_state_error_gate"}]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for row in selected:
        ax.scatter(row["coverage"], row["unsafe_accept_rate"], s=55)
        ax.annotate(row["method"].replace("_", " "), (row["coverage"], row["unsafe_accept_rate"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(.05, color="black", linestyle="--", linewidth=1)
    ax.axvline(.03, color="gray", linestyle=":", linewidth=1)
    ax.set(xlabel="Coverage", ylabel="Unsafe-accept rate", title="Frozen operating points", xlim=(0, 1), ylim=(0, 1))
    _save_figure(fig, output, "figure_operating_points", int(config["figure_dpi"]))

    clean = sumo["summaries"]["clean"]
    method_order = ["surface_every_tick", "learned_scalar_ttl", "always_fresh", "local_only"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(range(len(method_order)), [clean[name]["mean_application_bytes"] for name in method_order], color=["#2166ac", "#b2182b", "#777777", "#4d9221"])
    ax.set_xticks(range(len(method_order)), [name.replace("_", "\n") for name in method_order], fontsize=8)
    ax.set(ylabel="Application bytes per seed", title="Clean closed-loop communication")
    _save_figure(fig, output, "figure_sumo_communication", int(config["figure_dpi"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    values = [row["unsafe_accept_rate"] if row["unsafe_accept_rate"] is not None else 0 for row in ablation_rows]
    coverage_values = [row["coverage"] for row in ablation_rows]
    x = np.arange(len(ablation_rows))
    ax.bar(x - .2, values, width=.4, label="UAR", color="#b2182b")
    ax.bar(x + .2, coverage_values, width=.4, label="coverage", color="#2166ac")
    ax.set_xticks(x, [row["variant"].replace("_", "\n") for row in ablation_rows], fontsize=6)
    ax.set(ylabel="Fraction", title="Locked-test ablations", ylim=(0, 1))
    ax.legend(frameon=False)
    _save_figure(fig, output, "figure_ablations", int(config["figure_dpi"]))

    agreement = second["oracle_label_agreement"]["overall"]
    matrix = np.asarray([[agreement["both_valid"], agreement["policy_a_only_valid"]], [agreement["policy_b_only_valid"], agreement["both_invalid"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, str(value), ha="center", va="center")
    ax.set_xticks((0, 1), ("Policy B valid", "Policy B invalid"))
    ax.set_yticks((0, 1), ("Policy A valid", "Policy A invalid"))
    ax.set_title("Matched oracle labels across planners")
    fig.colorbar(image, ax=ax, fraction=.046)
    _save_figure(fig, output, "figure_second_planner", int(config["figure_dpi"]))

    categories = {
        "surface_unsafe_accept": lambda row: row["decisions"]["surface"] and not row["valid"],
        "surface_safe_accept": lambda row: row["decisions"]["surface"] and row["valid"],
        "surface_safe_reject": lambda row: not row["decisions"]["surface"] and row["valid"],
    }
    cases = {name: [row for row in decisions if predicate(row)][: int(config["case_count_per_category"])] for name, predicate in categories.items()}
    write_json_atomic(output / "selected_cases.json", cases)

    summary = f"""# Frozen Results Summary

The Phase-8 engineering gate passed, but the registered scientific success conditions did not.

On {offline['data']['point_count']:,} locked-test decisions, SurfaceNet accepted {offline['operating_point_results']['surface']['overall']['accepted']:,} decisions (coverage {_fmt(offline['operating_point_results']['surface']['overall']['coverage'])}) with unsafe-accept rate {_fmt(offline['operating_point_results']['surface']['overall']['unsafe_accept_rate'])}. The learned scalar TTL accepted {offline['operating_point_results']['learned_scalar_ttl']['overall']['accepted']:,} decisions with unsafe-accept rate {_fmt(offline['operating_point_results']['learned_scalar_ttl']['overall']['unsafe_accept_rate'])}.

At the frozen matched coverage of {_fmt(offline['matched_coverage_fraction'])}, the Surface-minus-scalar UAR estimate was {_fmt(offline_ci['estimate_first_minus_second'])}, with clustered 95% interval [{_fmt(offline_ci['lower'])}, {_fmt(offline_ci['upper'])}]. The interval crosses zero.

The 30-seed closed-loop run completed all {sumo['registered_branch_count']} branches without execution failures. No evaluated method produced a collision or registered critical event, so comparative closed-loop safety is not identifiable in these scenarios. Surface used an average of {_fmt(clean['surface_every_tick']['mean_application_bytes'], 1)} application bytes per clean seed, compared with {_fmt(clean['always_fresh']['mean_application_bytes'], 1)} for always-fresh and {_fmt(clean['learned_scalar_ttl']['mean_application_bytes'], 1)} for scalar TTL.

The evidence supports an implementation and falsification study under the frozen policy and evaluated distributions. It does not support the proposed geometric-superiority or communication-savings hypotheses.
"""
    limitations = f"""# Limitations

- The locked-test Surface operating point has {_fmt(offline['operating_point_results']['surface']['overall']['coverage'])} coverage and does not attain the registered 5% unsafe-accept upper-bound target.
- The matched-coverage Surface-versus-scalar interval crosses zero.
- All closed-loop critical-event outcomes are zero, leaving safety-effect power unresolved despite {sumo['matched_seed_count']} matched seeds.
- Complete application accounting shows higher Surface traffic than always-fresh in the clean confirmatory run.
- The second-planner oracle agreement is {_fmt(agreement['agreement'])} with kappa {_fmt(agreement['cohen_kappa'])}; forced transfer does not attain the safety target.
- The directional audit contains {geometry['boundary_audit']['infeasible_probe_count']} infeasible probes, so absence of directional re-entry does not establish global convexity.
- The SUMO-only label-source ablation has {source_ablation['calibration']['pair_count']} calibration pairs and is underpowered for the registered minimum acceptance requirement.
- Cooperative-track perturbations are controlled synthetic corruptions of trajectory-derived objects, not production CPM sensor errors.
- The claims are bounded to the frozen planner, registered thresholds, selected V2X-Traj intersections, and compact SUMO scenarios.
"""
    ledger = f"""# Final Claim Ledger

## Supported

- The software implements a behavior- and policy-bound 48-byte expiry-surface protocol with deterministic receiver checks.
- The real-data adequacy and engineering reproducibility gates pass for the recorded artifacts.
- Every-tick checking reduces unsafe acceptance relative to receipt-only checking in the locked offline ablation.

## Not supported

- A statistically resolved UAR improvement over the learned behavior-conditioned scalar TTL.
- Attainment of the registered 5% unsafe-accept target with nontrivial coverage.
- Reduced complete-accounting communication versus always-fresh or scalar TTL.
- A closed-loop critical-event improvement in the evaluated SUMO scenarios.
- Planner-independent transfer or broad deployment generalization.

## Publication position

The current evidence is suitable for a transparent negative-result, methods, or benchmark report. It does not satisfy the project’s registered Q1-strength decision gate.
"""
    for name, text in (("RESULTS_SUMMARY.md", summary), ("LIMITATIONS.md", limitations), ("CLAIM_LEDGER.md", ledger)):
        if prohibited_claims(text):
            raise RuntimeError(f"prohibited claim language in {name}: {prohibited_claims(text)}")
        (output / name).write_text(text, encoding="utf-8", newline="\n")

    environment = {"python": sys.version, "torch": torch.__version__, "cuda_build": torch.version.cuda, "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    provenance = {"schema_version": 1, "sources": {key: {"path": config[key], "sha256": source_hashes[key]} for key in source_keys}, "environment": environment, "uv_lock_sha256": sha256_file(ROOT / "uv.lock"), "config_sha256": sha256_file(CONFIG), "git": state, "generated_utc": utc_now()}
    provenance["provenance_sha256"] = canonical_json_hash(provenance)
    write_json_atomic(output / "provenance.json", provenance)
    budget = build_budget_report(Budget(workspace_root=ROOT, data_root=ROOT / "data"))
    write_json_atomic(output / "storage_report.json", budget)
    hashes = artifact_hashes(output, exclude=("artifact_manifest.json",))
    manifest = {"schema_version": 1, "phase": 11, "status": "PASS", "scientific_success": False, "artifact_count": len(hashes), "artifact_sha256": hashes, "all_values_programmatically_generated": True, "prohibited_claim_language_found": False, "source_sha256": source_hashes, "config_sha256": sha256_file(CONFIG), "git": state, "completed_utc": utc_now()}
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "scientific_success": manifest["scientific_success"], "artifact_count": manifest["artifact_count"], "manifest_sha256": manifest["manifest_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
