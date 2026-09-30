#!/usr/bin/env python3
"""Development-only empirical audit of conditional regret-bound premises.

This is a structural and sampled-cost diagnostic. It cannot establish a
uniform cost envelope or prove neural-surface containment between samples.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.oracle.planner import PlannerConfig
from bces.oracle.world import CostVector, GroundTruthWorld, PlannedTrajectory, WorldObject
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import ControlledDecisionPlanner, route_for
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import utc_now, write_json_atomic

INPUT = ROOT / 'outputs/study_b/controlled_development_v2'
OUTPUT = ROOT / 'outputs/study_b/regret_premise_audit_v1'
SPLIT = 'validation'


def candidates(planner, ego, route, objects, timestamp):
    world = GroundTruthWorld(timestamp, tuple(WorldObject(**row) for row in objects))
    result = {}
    for maneuver in planner.config.lateral_maneuvers:
        for acceleration in planner.config.longitudinal_accelerations_mps2:
            points = planner._candidate(ego, route, acceleration, maneuver)
            if points is None:
                continue
            dummy = CostVector(0., 0., 0., 0., 0., 0., 0., None)
            path = PlannedTrajectory(maneuver, acceleration, points, dummy)
            result[f'{maneuver}:{acceleration:g}'] = planner.evaluate(path, world, ego, route)
    return result


def audit_scene(path, planner):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        scene = json.load(handle)
    references = {r['reference_id']: r for r in scene['references']}
    counters = Counter()
    max_cost_shift = 0.
    largest_scalar_gap = 0.
    family = scene['family']
    for point in scene['points']:
        reference = references[point['reference_id']]
        behavior = reference['behavior']
        variant = str(point['receiver_drift_variant'])
        ref_time = str(reference['reference_timestamp_ms'])
        now_time = str(point['current_timestamp_ms'])
        frames = scene['traces'][variant]
        if ref_time not in frames or now_time not in frames:
            counters['missing_trace_points'] += 1
            continue
        ref_frame, now_frame = frames[ref_time], frames[now_time]
        ref_ego = ego_from_sample(ref_frame['receiver'])
        now_ego = ego_from_sample(now_frame['receiver'])
        ref_route = route_for(ref_ego, behavior, planner)
        now_route = route_for(now_ego, behavior, planner)
        reference_costs = candidates(planner, ref_ego, ref_route, ref_frame['objects'], int(ref_time))
        current_costs = candidates(planner, now_ego, now_route, now_frame['objects'], int(now_time))
        counters['points'] += 1
        if not reference_costs or not current_costs:
            counters['empty_action_set_points'] += 1
            continue
        if reference_costs.keys() != current_costs.keys():
            counters['changed_action_set_points'] += 1
        if len(reference_costs) == 1:
            counters['singleton_reference_action_points'] += 1
        if ref_time == now_time:
            counters['origin_points'] += 1
        else:
            common = reference_costs.keys() & current_costs.keys()
            if common:
                shift = max(abs(reference_costs[key].total - current_costs[key].total) for key in common)
                max_cost_shift = max(max_cost_shift, shift)
                counters['nonorigin_points_with_common_action'] += 1
                if shift > 1e-9:
                    counters['nonzero_sampled_cost_shift_points'] += 1
                if any(reference_costs[key].collision != current_costs[key].collision for key in common):
                    counters['collision_indicator_change_points'] += 1
        # The planner selects lexicographically, not by scalar total alone.
        best = min(reference_costs, key=lambda key: (
            reference_costs[key].collision, reference_costs[key].ttc,
            reference_costs[key].total, key.split(':')[0], float(key.split(':')[1])))
        gap = reference_costs[best].total - min(c.total for c in reference_costs.values())
        largest_scalar_gap = max(largest_scalar_gap, gap)
        if gap > 1e-9:
            counters['reference_scalar_nonoptimal_points'] += 1
    return {'scene': path.name, 'family': family, 'counts': dict(counters),
            'largest_sampled_cost_shift': max_cost_shift,
            'largest_reference_scalar_nonoptimality': largest_scalar_gap,
            'input_sha256': sha256_file(path)}


def main():
    manifest_path = INPUT / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    paths = sorted(INPUT.glob(f'{SPLIT}_*.json.gz'))
    if not paths:
        raise RuntimeError('no validation development inputs')
    for path in paths:
        if manifest['artifact_sha256'][path.name] != sha256_file(path):
            raise RuntimeError('development input changed: ' + path.name)
    config = json.loads((ROOT / 'outputs/study_b/slot_confirmation_v1/preregistration.json').read_text())['config']
    planner = ControlledDecisionPlanner(PlannerConfig.from_yaml(ROOT / config['planner']))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(paths, 1):
        result_path = OUTPUT / (path.stem.replace('.json', '') + '.json')
        if result_path.exists():
            row = json.loads(result_path.read_text(encoding='utf-8'))
            if row['input_sha256'] != sha256_file(path):
                raise RuntimeError('preserved premise audit row does not match input')
        else:
            row = audit_scene(path, planner)
            write_json_atomic(result_path, row)
        rows.append(row)
        if index % 10 == 0:
            print(json.dumps({'audited_scenes': index, 'fixed_total': len(paths)}), flush=True)
    overall = Counter()
    by_family = defaultdict(Counter)
    for row in rows:
        overall.update(row['counts'])
        by_family[row['family']].update(row['counts'])
    result = {'status': 'COMPLETE', 'completed_utc': utc_now(),
              'split': SPLIT, 'scenes': len(rows), 'counts': dict(overall),
              'by_family': {family: dict(counts) for family, counts in sorted(by_family.items())},
              'maximum_sampled_cost_shift': max(row['largest_sampled_cost_shift'] for row in rows),
              'maximum_reference_scalar_nonoptimality': max(
                  row['largest_reference_scalar_nonoptimality'] for row in rows),
              'uniform_envelope_proved': False, 'surface_containment_proved': False,
              'meaning': 'sampled simulator-state audit only; unavailable objects and continuous region remain unverified',
              'development_only': True, 'not_confirmatory': True,
              'source_sha256': sha256_file(Path(__file__)),
              'input_manifest_sha256': sha256_file(manifest_path),
              'artifact_sha256': {row['scene'].replace('.json.gz', '.json'): sha256_file(
                  OUTPUT / row['scene'].replace('.json.gz', '.json')) for row in rows}}
    write_json_atomic(OUTPUT / 'result.json', result)
    print(json.dumps({'status': 'COMPLETE', 'scenes': len(rows), 'counts': dict(overall)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
