#!/usr/bin/env python3
"""Replay frozen validation checkpoints through actual bound query/response bytes."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from bces.evaluation.study_b import reference_features, selective_metrics
from bces.models.bound_policy import FrozenWirePolicy, reference_query, bind_reference, wire_features
from bces.network.events import NetworkCondition
from bces.network.wire_channel import WireChannel
from bces.oracle.world import WorldObject
from bces.protocol.bindings import sender_id_hash
from bces.protocol.bound_exchange import BoundReceiver
from bces.protocol.receiver import DecisionState
from bces.simulation.controlled_contract import make_message, ego_from_sample
from bces.simulation.controlled_decisions import kinematic
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic


def main():
    state = git_state(ROOT)
    if state['dirty']: raise RuntimeError('clean source required')
    base = ROOT/'outputs/study_b'
    output = base/'controlled_wire_audit_v2.json'
    if output.exists(): raise FileExistsError('completed wire audit is immutable')
    data_root, model_root = base/'controlled_development_v2', base/'decision_surface_controlled_v2'
    data = json.loads((data_root/'run_manifest.json').read_text())
    training = json.loads((model_root/'run_manifest.json').read_text())
    if training['data_manifest_sha256'] != sha256_file(data_root/'run_manifest.json'):
        raise ValueError('training/data mismatch')
    torch.set_num_threads(2)
    models = {}
    for family in ('surface', 'scalar_ttl'):
        group = training['runs']['reference_margins:'+family]
        seed = group['selected_seed']
        checkpoint = model_root/f'reference_margins_{family}_{seed}.pt'
        expected = next(r['checkpoint_sha256'] for r in group['runs'] if r['seed']==seed)
        models[family] = FrozenWirePolicy(checkpoint, expected_sha256=expected, policy_hash=data['policy_hash'])
    expected_scores = {}
    if sha256_file(model_root/'predictions.jsonl.gz') != training['artifacts']['predictions.jsonl.gz']:
        raise ValueError('saved prediction artifact mismatch')
    with gzip.open(model_root/'predictions.jsonl.gz', 'rt') as f:
        for line in f:
            row = json.loads(line)
            if row['split']=='validation': expected_scores[row['point_id']] = row['score']
    decisions = defaultdict(list); sizes = defaultdict(list); latency = defaultdict(list)
    counts = Counter(); network_cases = []
    for name, digest in sorted(data['artifact_sha256'].items()):
        if not name.startswith('validation_'): continue
        path = data_root/name
        if sha256_file(path) != digest: raise ValueError('source artifact mismatch')
        with gzip.open(path, 'rt') as f: scene = json.load(f)
        by_ref = defaultdict(list)
        for point in scene['points']: by_ref[point['reference_id']].append(point)
        for index, reference in enumerate(scene['references']):
            timestamp = reference['reference_timestamp_ms']
            frame = scene['traces']['0'][str(timestamp)]
            ego = ego_from_sample(frame['receiver'])
            objects = tuple(WorldObject(**o) for o in frame['objects'])
            cooperative = tuple(o for o in objects if math.hypot(o.x_m-ego.x_m, o.y_m-ego.y_m)<=120.)
            message = make_message(cooperative, timestamp, message_id=int(scene['scenario_id']))
            payload = message.encode()
            q = reference_query(reference, query_id=index+1, sender_hash=sender_id_hash(message.sender_id))
            for family, model in models.items():
                bound = bind_reference(reference, q, payload, model_sha256=model.sha256,
                    view=model.view, family=family, shrinkage=0.)
                expected = np.concatenate((reference_features(reference), np.asarray(reference['observable_margin_features'],np.float32)))
                if not np.array_equal(wire_features(bound,payload),expected):
                    raise AssertionError('wire features changed the frozen model inputs')
                wire = bound.encode()
                start = time.perf_counter_ns()
                response = model.issue(wire, payload)
                latency[family].append((time.perf_counter_ns()-start)/1e6)
                receiver = BoundReceiver(); receiver.register(wire); receiver.install(response,payload)
                sizes[family].append({'payload':len(payload),'query':len(wire),'response':len(response)})
                for point in by_ref[reference['reference_id']]:
                    sample = scene['traces'][str(point['receiver_drift_variant'])][str(point['current_timestamp_ms'])]['receiver']
                    current = kinematic(ego_from_sample(sample), point['current_timestamp_ms'])
                    accepted = receiver.evaluate(current,q.behavior,q.policy_hash,message.sender_id).state==DecisionState.ACCEPT_REUSE
                    prior = expected_scores[point['point_id']]['reference_margins:'+family]>=0
                    counts[family+':cuda_wire_decision_mismatches'] += accepted != prior
                    decisions[family].append((point['valid'],accepted,point['scenario_id']))
                # One reference per scenario: actual payload receipt precedes query
                # generation, then response issuance. This is a transport fixture,
                # not a closed-loop safety or bandwidth-benefit experiment.
                if index==0:
                    channel = WireChannel(NetworkCondition(50,.1,1,duplicate_probability=.1,reorder_probability=.1),
                        seed=int(scene['scenario_id']), security_bytes=64)
                    channel.send('payload',payload,timestamp)
                    handled = set(); generated_query=None
                    while (event:=channel.next_delivery(timestamp+2000)) is not None:
                        if event.kind in handled:
                            channel.discard(event); continue
                        handled.add(event.kind)
                        if event.kind=='payload':
                            arrival = math.ceil(event.delivered_ms)
                            generated_query = replace(bound, generated_ms=arrival,payload_received_ms=arrival,
                                context_available_ms=arrival).encode()
                            channel.send('query',generated_query,arrival)
                        elif event.kind=='query':
                            reply = model.issue(event.body,payload)
                            channel.send('response',reply,event.delivered_ms)
                    report = channel.report()
                    if not report['byte_conservation_ok'] or not report['component_conservation_ok']:
                        raise AssertionError('wire accounting failed')
                    network_cases.append({'scenario_id':scene['scenario_id'],'family':family,**report})
            counts['references'] += 1
        counts['scenarios'] += 1
    metrics = {family:selective_metrics(*zip(*rows)) for family,rows in decisions.items()}
    report = {'status':'PASS' if not any(v for k,v in counts.items() if 'mismatches' in k) else 'NUMERICAL_REVIEW_REQUIRED',
        'scope':'validation_development_only','scientific_success':False,'manuscript_allowed':False,
        'counts':dict(counts),'wire_metrics':metrics,
        'mean_exchange_application_bytes':{key:{part:float(np.mean([r[part] for r in rows])) for part in ('payload','query','response')} for key,rows in sizes.items()},
        'sender_issue_latency_ms':{key:{'p50':float(np.quantile(rows,.5)),'p95':float(np.quantile(rows,.95))} for key,rows in latency.items()},
        'network_cases':network_cases,'model_sha256':{k:m.sha256 for k,m in models.items()},
        'data_manifest_sha256':sha256_file(data_root/'run_manifest.json'),
        'training_manifest_sha256':sha256_file(model_root/'run_manifest.json'),
        'git':state,'completed_utc':utc_now(),
        'limitations':['transport fixtures are not a closed-loop communication savings measurement',
            'security bytes are modeled, not authentication or physical radio overhead',
            'sender computation wall time is measured but not injected into fixture network latency',
            'no fresh confirmation or final-test data accessed']}
    write_json_atomic(output,report)
    print(json.dumps({k:report[k] for k in ('status','counts','wire_metrics','mean_exchange_application_bytes','sender_issue_latency_ms')},indent=2))
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__': raise SystemExit(main())
