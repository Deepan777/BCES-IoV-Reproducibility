#!/usr/bin/env python3
"""Paired development diagnosis of autonomous SUMO ego lane changes.

No seed from the completed 900000+ closed-loop confirmation is selected.
This is exploratory engineering work, never replacement confirmatory evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil

from bces.network.events import NetworkCondition
import bces.simulation.controlled_closed_loop as closed_loop
from bces.simulation.controlled_slots import development_generator
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import utc_now, write_json_atomic


def selected_seeds(start: int, count: int, config: dict) -> list[int]:
    if 300_000 <= start < 900_000 and 1 <= count <= 500:
        selected = []
        for seed in range(start, start + 10 * count):
            if development_generator().sample_scenario(seed, config)[0].family == 'unprotected_crossing':
                selected.append(seed)
                if len(selected) == count:
                    return selected
    raise ValueError('development-only seed range or family quota invalid')


def tracking_summary(branch: dict) -> dict:
    violations = [
        {'timestamp_ms': row['timestamp_ms'], 'position_error_m': row['tracking']['position_error_m'],
         'speed_error_mps': row['tracking']['speed_error_mps'],
         'heading_error_rad': row['tracking']['heading_error_rad'],
         'maneuver': row['command']['maneuver']}
        for row in branch['rows'] if 'tracking' in row and (
            row['tracking']['position_error_m'] > .5
            or row['tracking']['speed_error_mps'] > .5
            or row['tracking']['heading_error_rad'] > .1)
    ]
    return {'actuation_contract_passed': branch['actuation_contract_passed'],
            'lane_change_mode': branch['initial_contract']['snapshot']['vehicles']['ego']['lane_change_mode'],
            'outcome': branch['outcome'], 'tracking_violations': branch['counts']['tracking_violations'],
            'illegal_lane_requests': branch['counts']['illegal_lane_requests'],
            'max_position_error_m': max((row['tracking']['position_error_m'] for row in branch['rows']
                                         if 'tracking' in row), default=0.),
            'violations': violations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=300_000)
    parser.add_argument('--cases', type=int, default=300)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/study_b/lane_actuation_development_v1')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'outputs/study_b'):
        raise ValueError('output outside Study B')
    registration_path = ROOT / 'outputs/study_b/closed_loop_confirmation_v1/preregistration.json'
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    config = registration['config']
    seeds = selected_seeds(args.start, args.cases, config)
    if any(seed in {row['seed'] for row in registration['scenario_sampling']['selected']} for seed in seeds):
        raise AssertionError('confirmation seed entered development diagnosis')
    condition = NetworkCondition(**registration['conditions']['nominal'])
    protocol = {'scope': 'paired development engineering diagnosis, not confirmation',
                'created_utc': utc_now(), 'seeds': seeds, 'selected_sha256': canonical_json_hash(seeds),
                'selection': 'first outcome-blind unprotected-crossing family assignments',
                'condition': registration['conditions']['nominal'], 'method': 'local_only',
                'baseline_lane_change_mode': 'SUMO default', 'candidate_lane_change_mode': 0,
                'registration_sha256': sha256_file(registration_path),
                'baseline_source_sha256': sha256_file(ROOT / 'bces/simulation/controlled_closed_loop.py'),
                'script_sha256': sha256_file(Path(__file__)),
                'confirmation_seeds_excluded': True}
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        prior = json.loads(protocol_path.read_text(encoding='utf-8'))
        for key in ('seeds', 'selected_sha256', 'condition', 'method', 'candidate_lane_change_mode',
                    'registration_sha256', 'baseline_source_sha256', 'script_sha256'):
            if prior[key] != protocol[key]:
                raise RuntimeError('development diagnosis context changed')
    else:
        write_json_atomic(protocol_path, protocol)

    original_populate = closed_loop._populate

    def repaired_populate(connection, spec):
        original_populate(connection, spec)
        connection.vehicle.setLaneChangeMode('ego', 0)

    for index, seed in enumerate(seeds, start=1):
        path = output / f'seed_{seed}.json'
        if path.exists():
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['seed'] != seed or record['protocol_sha256'] != sha256_file(protocol_path):
                raise RuntimeError('preserved development pair mismatch')
            continue
        if psutil.virtual_memory().available < 1_500_000_000:
            raise RuntimeError('resource pause before development pair')
        kwargs = {'seed': seed, 'method': 'local_only', 'config': config,
                  'condition': condition, 'network_seed': seed + 910_000,
                  'horizon_s': registration['horizon_s']}
        baseline = closed_loop.run_branch(**kwargs)
        with patch.object(closed_loop, '_populate', repaired_populate):
            repaired = closed_loop.run_branch(**kwargs)
        record = {'seed': seed, 'family': 'unprotected_crossing',
                  'protocol_sha256': sha256_file(protocol_path),
                  'baseline': tracking_summary(baseline),
                  'repaired': tracking_summary(repaired)}
        if path.exists():
            raise FileExistsError(path)
        write_json_atomic(path, record)
        if index % 10 == 0:
            print(json.dumps({'completed_pairs': index, 'fixed_total': len(seeds)}), flush=True)

    records = [json.loads((output / f'seed_{seed}.json').read_text(encoding='utf-8')) for seed in seeds]
    result = {'status': 'COMPLETE', 'completed_utc': utc_now(), 'protocol_sha256': sha256_file(protocol_path),
              'pairs': len(records),
              'baseline_failed': sum(not row['baseline']['actuation_contract_passed'] for row in records),
              'repaired_failed': sum(not row['repaired']['actuation_contract_passed'] for row in records),
              'baseline_failed_seeds': [row['seed'] for row in records
                                        if not row['baseline']['actuation_contract_passed']],
              'repaired_failed_seeds': [row['seed'] for row in records
                                        if not row['repaired']['actuation_contract_passed']],
              'baseline_mean_max_position_error_m': mean(row['baseline']['max_position_error_m'] for row in records),
              'repaired_mean_max_position_error_m': mean(row['repaired']['max_position_error_m'] for row in records),
              'confirmation_outcomes_not_used': True, 'not_confirmatory': True,
              'artifact_sha256': {f'seed_{seed}.json': sha256_file(output / f'seed_{seed}.json')
                                  for seed in seeds}}
    write_json_atomic(output / 'result.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'pairs', 'baseline_failed', 'repaired_failed')}),
          flush=True)


if __name__ == '__main__':
    main()
