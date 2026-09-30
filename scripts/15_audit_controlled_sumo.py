#!/usr/bin/env python3
"""Verify receiver/payload availability and checkpoint continuations, not efficacy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bces.simulation.branching import _populate
from bces.simulation.scenario_builder import FAMILIES,scenario_spec
from bces.simulation.sumo_adapter import sumo_version
from bces.simulation.controlled_contract import (start_controlled,checkpoint_snapshot,
    compare_snapshots,receiver_state,centered_world,make_message,payload_objects,restore_control_modes)
from bces.utils.budget import Budget,require_headroom,build_budget_report
from bces.utils.hashing import canonical_json_hash,sha256_file
from bces.utils.reproducibility import git_state,utc_now,write_json_atomic


def rollout(connection,spec,previous,steps=15):
    if spec.uncontrolled_signal:
        for signal in connection.trafficlight.getIDList():
            value=connection.trafficlight.getRedYellowGreenState(signal)
            connection.trafficlight.setRedYellowGreenState(signal,'G'*len(value))
    if spec.braking_event and spec.actor_id in connection.vehicle.getIDList():
        connection.vehicle.slowDown(spec.actor_id,0.,1.)
    frames=[]
    for _ in range(steps):
        connection.simulationStep()
        previous=receiver_state(connection,previous)
        message=make_message(centered_world(connection),previous['timestamp_ms'])
        frames.append({'snapshot':checkpoint_snapshot(connection),'receiver':previous,
                       'payload_sha256':canonical_json_hash(message.to_wire_dict()),
                       'payload_bytes_actual_json':len(message.encode()),
                       'reconstructed_object_count':len(payload_objects(message))})
    return frames


def main():
    output=ROOT/'outputs/study_b/controlled_contract_v2'
    state=git_state(ROOT)
    if state['dirty']:
        raise RuntimeError('commit controlled contract before auditing')
    if (output/'audit.json').exists():
        raise FileExistsError('completed audit is immutable')
    if sumo_version()!='1.27.1':
        raise RuntimeError('pinned SUMO 1.27.1 required')
    budget=Budget(workspace_root=ROOT,data_root=ROOT/'data')
    require_headroom(budget,incoming_bytes=10_000_000)
    output.mkdir(parents=True,exist_ok=True)
    network=ROOT/'sumo/registered_grid_v1.net.xml'
    results=[]
    for index,family in enumerate(FAMILIES):
        seed=150926+index; spec=scenario_spec(family)
        checkpoint=output/f'{family}.xml.gz'
        if checkpoint.exists():
            raise FileExistsError('partial checkpoint retained; use a new audit version after investigating')
        connection=start_controlled(network=network,label=f'contract_original_{seed}',seed=seed,step_s=.2)
        try:
            _populate(connection,spec)
            previous=None
            for _ in range(25):
                connection.simulationStep()
                previous=receiver_state(connection,previous)
            origin=checkpoint_snapshot(connection)
            origin_receiver=previous
            connection.simulation.saveState(str(checkpoint))
            original=rollout(connection,spec,previous)
        finally:
            connection.close()
        connection=start_controlled(network=network,label=f'contract_restored_{seed}',seed=seed,step_s=.2,state=checkpoint)
        try:
            restore_control_modes(connection,origin)
            restored_origin=checkpoint_snapshot(connection)
            restored=rollout(connection,spec,origin_receiver)
        finally:
            connection.close()
        initial=compare_snapshots(origin,restored_origin)
        # A numeric-tolerance replay audit must not demand identical JSON float
        # spellings/hashes after checkpoint serialization. Record both hashes,
        # but compare measured state and derived receiver inputs numerically.
        comparison_keys=('snapshot','receiver','reconstructed_object_count')
        comparisons=[compare_snapshots({k:a[k] for k in comparison_keys},
                                       {k:b[k] for k in comparison_keys})
                     for a,b in zip(original,restored)]
        # Independently repeat the complete pre-action history. This preserves
        # unsaved simulator/controller history rather than trusting a snapshot.
        connection=start_controlled(network=network,label=f'contract_prefix_{seed}',seed=seed,step_s=.2)
        try:
            _populate(connection,spec)
            previous=None
            for _ in range(25):
                connection.simulationStep()
                previous=receiver_state(connection,previous)
            prefix_origin=checkpoint_snapshot(connection)
            prefix=rollout(connection,spec,previous)
        finally:
            connection.close()
        prefix_initial=compare_snapshots(origin,prefix_origin)
        prefix_comparisons=[compare_snapshots({k:a[k] for k in comparison_keys},
                                              {k:b[k] for k in comparison_keys})
                            for a,b in zip(original,prefix)]
        result={'family':family,'seed':seed,'initial_state':initial,
            'continuation_equal':all(c['equal_within_tolerance'] for c in comparisons),
            'continuation_comparisons':comparisons,'original':original,'restored':restored,
            'prefix_initial_state':prefix_initial,'prefix_continuation':prefix,
            'prefix_continuation_comparisons':prefix_comparisons,
            'prefix_continuation_equal':all(c['equal_within_tolerance'] for c in prefix_comparisons),
            'checkpoint_sha256':sha256_file(checkpoint),
            'maximum_abs_curvature':max(abs(f['receiver']['curvature_inv_m']) for f in original)}
        results.append(result)
        print(json.dumps({'family':family,'initial_equal':initial['equal_within_tolerance'],
                          'continuation_equal':result['continuation_equal'],
                          'prefix_equal':prefix_initial['equal_within_tolerance'] and result['prefix_continuation_equal']}),flush=True)
    passed=all(r['prefix_initial_state']['equal_within_tolerance'] and r['prefix_continuation_equal'] for r in results)
    report={'schema_version':2,'status':'PASS' if passed else 'FAIL','scope':'engineering_contract_only',
        'approved_branch_initialization':'deterministic_complete_prefix_replay' if passed else None,
        'checkpoint_restore_all_families_passed':all(r['initial_state']['equal_within_tolerance'] and r['continuation_equal'] for r in results),
        'scientific_success':False,'manuscript_allowed':False,'sumo_version':sumo_version(),
        'families':results,'network_sha256':sha256_file(network),
        'contract_source_sha256':sha256_file(ROOT/'bces/simulation/controlled_contract.py'),
        'git':state,'completed_utc':utc_now(),'budget':build_budget_report(budget)['measurements'],
        'limitations':['checkpoint equality is checked only for these deterministic continuations',
            'state reload does not by itself guarantee every hidden car-following/lane-change state',
            'ideal local observations and simulated state are not measured real-world sensing',
            'payload moving heading assumes no side slip; stopped orientation uses enclosing square',
            'no BCES efficacy, uncertainty calibration or risk control was tested in this audit']}
    report['report_sha256']=canonical_json_hash(report)
    write_json_atomic(output/'audit.json',report)
    print(json.dumps({'status':report['status'],'families':len(results),'output':str(output)}))
    return 0 if passed else 2


if __name__=='__main__':
    raise SystemExit(main())
