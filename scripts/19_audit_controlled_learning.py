#!/usr/bin/env python3
"""All-seed, paired validation ranking audit; no calibration or model tuning."""
from __future__ import annotations

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

COVERAGES = (.5, .75, .8)
BOOTSTRAP_SEED = 150926
BOOTSTRAP_REPLICATES = 2000


def rank_subset(scores, fraction):
    """Never include points rejected by the unshrunk surface/origin gate.

    Ranking tie breaks are descriptive and may not correspond to a wire rule.
    This is not calibration, a deployment policy or a matched-safety claim.
    """
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or not 0 < fraction <= 1:
        raise ValueError('finite scores and a fraction in (0,1] required')
    count = int(len(scores) * fraction)
    if count < 1 or np.count_nonzero(scores >= 0) < count:
        return None
    indices = np.argsort(-scores, kind='stable')[:count]
    selected = np.zeros(len(scores), bool)
    selected[indices] = True
    return selected


def paired_difference(valid, surface, ttl, scenarios):
    valid, surface, ttl, scenarios = map(np.asarray, (valid, surface, ttl, scenarios))
    names = np.unique(scenarios)
    counts = np.array([[mask[scenarios == name].sum(), (mask & ~valid)[scenarios == name].sum()]
                       for mask in (surface, ttl) for name in names]).reshape(2, len(names), 2)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(names), size=(BOOTSTRAP_REPLICATES, len(names)))
    sums = counts[:, draws, :].sum(axis=2)
    usable = (sums[:, :, 0] > 0).all(axis=0)
    difference = sums[0, usable, 1] / sums[0, usable, 0] - sums[1, usable, 1] / sums[1, usable, 0]
    totals = counts.sum(axis=1)
    return {'surface_minus_ttl_uar': float(totals[0, 1] / totals[0, 0] - totals[1, 1] / totals[1, 0]),
            'paired_scenario_bootstrap_interval_descriptive': np.quantile(difference, [.025, .975]).tolist(),
            'bootstrap_replicates_used': int(usable.sum()), 'scenario_count': len(names),
            'confirmatory_significance_claim': False}


def main():
    state = git_state(ROOT)
    if state['dirty']:
        raise RuntimeError('clean audit revision required')
    base = ROOT / 'outputs/study_b'
    data_root, model_root = base / 'controlled_development_v2', base / 'decision_surface_controlled_v2'
    destination = base / 'controlled_learning_audit_v2.json'
    if destination.exists():
        raise FileExistsError('completed learning audit is immutable')
    manifest = json.loads((model_root / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['data_manifest_sha256'] != sha256_file(data_root / 'run_manifest.json'):
        raise ValueError('training/data mismatch')
    if manifest['original_test_accessed'] or manifest['scope'] != 'development_only':
        raise ValueError('development models required')
    refs, points, _ = load_development(data_root)
    # No new calibration analysis: the comparison is restricted to validation.
    points = [p for p in points if p['split'] == 'validation']
    refs = {key: ref for key, ref in refs.items() if ref['split'] == 'validation'}
    arrays = feature_arrays(refs, points)
    valid = np.array([p['valid'] for p in points], bool)
    scenarios = np.array([p['scenario_id'] for p in points])
    scores, results, checkpoints = {}, {}, {}
    torch.set_num_threads(2)
    for key, group in manifest['runs'].items():
        view, family = key.split(':')
        reference = arrays['base_reference'] if view == 'frozen_inputs' else arrays['margin_reference']
        for run in group['runs']:
            seed = run['seed']
            path = model_root / f'{view}_{family}_{seed}.pt'
            digest = sha256_file(path)
            if digest != run['checkpoint_sha256'] or digest != manifest['artifacts'][path.name]:
                raise ValueError('checkpoint hash mismatch')
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            model = DecisionSurfaceNet(checkpoint['feature_count'], scalar=checkpoint['scalar'])
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            x = ((reference - checkpoint['mean'].numpy()) / checkpoint['std'].numpy()).clip(-10, 10).astype(np.float32)
            with torch.no_grad():
                offsets, logits = model(torch.from_numpy(x))
                indices = torch.from_numpy(arrays['reference_index'])
                score = membership_score(offsets[indices], logits[indices], torch.from_numpy(arrays['drift']),
                                         scalar=checkpoint['scalar']).numpy()
            run_key = f'{key}:{seed}'
            scores[run_key] = score
            results[run_key] = selective_metrics(valid, score >= 0, scenarios)
            checkpoints[path.name] = digest
    comparisons = []
    for view in ('frozen_inputs', 'reference_margins'):
        surface_seeds = {r['seed'] for r in manifest['runs'][f'{view}:surface']['runs']}
        ttl_seeds = {r['seed'] for r in manifest['runs'][f'{view}:scalar_ttl']['runs']}
        if surface_seeds != ttl_seeds:
            raise ValueError('all-seed paired comparison requires matched seeds')
        for seed in sorted(surface_seeds):
            for coverage in COVERAGES:
                surface = rank_subset(scores[f'{view}:surface:{seed}'], coverage)
                ttl = rank_subset(scores[f'{view}:scalar_ttl:{seed}'], coverage)
                row = {'view': view, 'seed': seed, 'requested_coverage': coverage,
                       'both_models_have_enough_nonnegative_scores': surface is not None and ttl is not None}
                if row['both_models_have_enough_nonnegative_scores']:
                    row.update(paired_difference(valid, surface, ttl, scenarios))
                    row['surface'] = selective_metrics(valid, surface, scenarios)
                    row['ttl'] = selective_metrics(valid, ttl, scenarios)
                comparisons.append(row)
    report = {'scope': 'validation_development_only', 'scientific_success': False, 'manuscript_allowed': False,
              'original_test_accessed': False, 'calibration_used_for_this_audit': False,
              'run_metrics': results, 'paired_ranking_comparisons': comparisons,
              'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_replicates': BOOTSTRAP_REPLICATES,
              'training_manifest_sha256': sha256_file(model_root / 'run_manifest.json'),
              'data_manifest_sha256': sha256_file(data_root / 'run_manifest.json'), 'checkpoint_sha256': checkpoints,
              'limitations': ['ranking tie breaks are not deployable receiver decisions',
                             'descriptive validation comparison, not independent confirmation or simultaneous risk control',
                             'CPU recomputation may differ numerically from CUDA at a boundary',
                             'all matched training seeds retained; no seed chosen from these comparisons'],
              'git': state, 'completed_utc': utc_now()}
    write_json_atomic(destination, report)
    print(json.dumps({'output': str(destination), 'paired_comparisons': comparisons}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
