#!/usr/bin/env python3
"""Engineering replay of four *failed* v1 cases; never confirmatory evidence.

The four seeds were selected from the completed negative confirmation because
they exhibited the lane-position discontinuity. They are deliberately excluded
from the independent v2 confirmation sampling frame.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.network.events import NetworkCondition
import bces.simulation.controlled_closed_loop as closed_loop
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import utc_now, write_json_atomic

OUTPUT = ROOT / 'outputs/study_b/lane_mode_512_debug_v1'
FAILED_V1_SEEDS = (900397, 900416, 900661, 901696)


def main() -> None:
    registration_path = ROOT / 'outputs/study_b/closed_loop_confirmation_v1/preregistration.json'
    old_manifest = ROOT / 'outputs/study_b/closed_loop_confirmation_v1/confirmation/run_manifest.json'
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    assert json.loads(old_manifest.read_text(encoding='utf-8'))['status'] == 'COMPLETE'
    config = registration['config']
    conditions = {'nominal': NetworkCondition(**registration['conditions']['nominal']),
                  'impaired': NetworkCondition(**registration['conditions']['impaired'])}
    protocol = {'status': 'FROZEN_DEBUG_REPLAY', 'created_utc': utc_now(),
                'selection': 'all four known v1 engineering failures; never confirmatory',
                'seeds': list(FAILED_V1_SEEDS), 'methods': ['local_only', 'periodic_payload'],
                'conditions': list(conditions), 'candidate_lane_change_mode': 512,
                'sumo_mode_meaning': 'no autonomous change; TraCI requests retain collision and gap checks',
                'v1_registration_sha256': sha256_file(registration_path),
                'v1_manifest_sha256': sha256_file(old_manifest),
                'source_sha256': sha256_file(ROOT / 'bces/simulation/controlled_closed_loop.py'),
                'script_sha256': sha256_file(Path(__file__)), 'not_confirmatory': True}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT / 'protocol.json'
    if protocol_path.exists():
        prior = json.loads(protocol_path.read_text(encoding='utf-8'))
        for key in ('seeds', 'candidate_lane_change_mode', 'v1_registration_sha256',
                    'v1_manifest_sha256', 'source_sha256', 'script_sha256'):
            if prior[key] != protocol[key]:
                raise RuntimeError('debug protocol changed')
    else:
        write_json_atomic(protocol_path, protocol)
    original = closed_loop._populate

    def safety_preserving_populate(connection, spec):
        original(connection, spec)
        connection.vehicle.setLaneChangeMode('ego', 512)

    for seed in FAILED_V1_SEEDS:
        for condition_name, condition in conditions.items():
            for method in ('local_only', 'periodic_payload'):
                path = OUTPUT / f'{seed}_{condition_name}_{method}.json'
                if path.exists():
                    continue
                kwargs = {'seed': seed, 'method': method, 'config': config,
                          'condition': condition, 'network_seed': seed + 910000,
                          'horizon_s': registration['horizon_s']}
                with patch.object(closed_loop, '_populate', safety_preserving_populate):
                    branch = closed_loop.run_branch(**kwargs)
                row = {'seed': seed, 'condition': condition_name, 'method': method,
                       'lane_change_mode': branch['initial_contract']['snapshot']['vehicles']['ego']['lane_change_mode'],
                       'actuation_contract_passed': branch['actuation_contract_passed'],
                       'tracking_violations': branch['counts'].get('tracking_violations', 0),
                       'illegal_lane_requests': branch['counts'].get('illegal_lane_requests', 0),
                       'max_position_error_m': max((r['tracking']['position_error_m'] for r in branch['rows']
                                                    if 'tracking' in r), default=0.),
                       'outcome': branch['outcome'], 'not_confirmatory': True}
                write_json_atomic(path, row)
                print(json.dumps(row), flush=True)
    rows = [json.loads(path.read_text(encoding='utf-8'))
            for path in sorted(OUTPUT.glob('[0-9]*_*.json'))]
    result = {'status': 'COMPLETE', 'completed_utc': utc_now(), 'protocol_sha256': sha256_file(protocol_path),
              'branches': len(rows), 'failed_actuation': sum(not row['actuation_contract_passed'] for row in rows),
              'maximum_position_error_m': max(row['max_position_error_m'] for row in rows),
              'v1_failure_cases_replayed_for_engineering_only': True, 'not_confirmatory': True,
              'artifact_sha256': {path.name: sha256_file(path) for path in sorted(OUTPUT.glob('[0-9]*_*.json'))}}
    write_json_atomic(OUTPUT / 'result.json', result)
    print(json.dumps({'status': result['status'], 'branches': result['branches'],
                      'failed_actuation': result['failed_actuation']}), flush=True)


if __name__ == '__main__':
    main()
