#!/usr/bin/env python3
"""Frozen scenario-clustered secondary analysis; run only after v2 completion."""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.closed_loop_confirmation import CONDITIONS, FAMILIES, METHODS, _severe
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import utc_now, write_json_atomic

REGISTRATION = ROOT / 'outputs/study_b/closed_loop_confirmation_v2/preregistration.json'
OUTPUT = REGISTRATION.parent / 'confirmation'
BOOTSTRAP_REPLICATES = 20_000
RANDOM_SEED = 20260922


def interval(values, groups, rng):
    """Stratified scenario bootstrap keeps the frozen equal-family mixture."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    positions = [np.flatnonzero(groups == family) for family in FAMILIES if np.any(groups == family)]
    estimates = np.empty(BOOTSTRAP_REPLICATES)
    for start in range(0, BOOTSTRAP_REPLICATES, 500):
        stop = min(start + 500, BOOTSTRAP_REPLICATES)
        samples = [rng.choice(pos, size=(stop - start, len(pos)), replace=True) for pos in positions]
        estimates[start:stop] = np.concatenate([values[s] for s in samples], axis=1).mean(axis=1)
    return [float(q) for q in np.quantile(estimates, [.025, .975])]


def signflip_paired(values, rng):
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    exceed = 0
    for start in range(0, BOOTSTRAP_REPLICATES, 500):
        count = min(500, BOOTSTRAP_REPLICATES - start)
        signs = rng.choice((-1., 1.), size=(count, len(values)))
        exceed += int(np.count_nonzero(abs((signs * values).mean(axis=1)) >= observed - 1e-15))
    return (exceed + 1) / (BOOTSTRAP_REPLICATES + 1)


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    corrected = {}
    running = 0.
    for index, name in enumerate(ordered):
        running = max(running, min(1., pvalues[name] * (len(ordered) - index)))
        corrected[name] = running
    return corrected


def load_branch(registration_digest, artifact_digests, seed, family, condition, method):
    name = f'branch_{seed}_{condition}_{method}.json.gz'
    path = OUTPUT / name
    if sha256_file(path) != artifact_digests[name]:
        raise RuntimeError('branch artifact changed: ' + name)
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        wrapper = json.load(handle)
    if (wrapper['registration_sha256'] != registration_digest or wrapper['seed'] != seed
            or wrapper['family'] != family or wrapper['condition'] != condition
            or wrapper['method'] != method
            or wrapper['branch_sha256'] != canonical_json_hash(wrapper['branch'])):
        raise RuntimeError('branch wrapper identity/digest mismatch: ' + name)
    return wrapper['branch']


def main():
    registration = json.loads(REGISTRATION.read_text(encoding='utf-8'))
    manifest_path = OUTPUT / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    digest = sha256_file(REGISTRATION)
    if manifest['status'] != 'COMPLETE' or manifest['registration_sha256'] != digest:
        raise RuntimeError('v2 confirmation incomplete or registration mismatch')
    result_path = OUTPUT / 'secondary_analysis.json'
    if result_path.exists():
        raise FileExistsError('secondary analysis is immutable')
    scenario_rows = []
    for assignment in registration['scenario_sampling']['selected']:
        seed, family = assignment['seed'], assignment['family']
        block = {(condition, method): load_branch(digest, manifest['artifact_sha256'],
                                                    seed, family, condition, method)
                 for condition in CONDITIONS for method in METHODS}
        row = {'seed': seed, 'family': family, 'methods': {}}
        for method in METHODS:
            branches = [block[(condition, method)] for condition in CONDITIONS]
            periodic = [block[(condition, 'periodic_payload')] for condition in CONDITIONS]
            reductions = [1. - b['traffic']['generated_bytes'] / p['traffic']['generated_bytes']
                          for b, p in zip(branches, periodic)]
            progress_deficit = any(
                b['route_distance_lower_bound_m'] - p['route_distance_lower_bound_m'] < -5.
                for b, p in zip(branches, periodic))
            adverse = any(_severe(b) for b in branches) or (method != 'periodic_payload' and progress_deficit)
            reuse = sum(b['counts'].get('reuse_decisions', 0) for b in branches)
            local = sum(b['counts'].get('local_only_decisions', 0) for b in branches)
            row['methods'][method] = {
                'mean_byte_reduction_vs_periodic': sum(reductions) / len(reductions),
                'byte_reductions_by_condition': reductions,
                'generated_bytes': sum(b['traffic']['generated_bytes'] for b in branches),
                'periodic_bytes': sum(p['traffic']['generated_bytes'] for p in periodic),
                'condition_pairs_using_more_bytes_than_periodic': sum(value < 0 for value in reductions),
                'reuse_fraction': reuse / (reuse + local) if reuse + local else 0.,
                'reuse_decisions': reuse,
                'adverse_episode': adverse,
                'actuation_failure': any(not b['actuation_contract_passed'] for b in branches),
                'geometric_overlap_ticks': sum(b['counts'].get('geometric_overlap_ticks', 0) for b in branches),
                'outcome_or_duration_mismatch_vs_periodic': any(
                    b['outcome'] != p['outcome'] or b['duration_s'] != p['duration_s']
                    for b, p in zip(branches, periodic)),
            }
        scenario_rows.append(row)
    rng = np.random.default_rng(RANDOM_SEED)
    families = [row['family'] for row in scenario_rows]
    family_table = {}
    for family in FAMILIES:
        subset = [row for row in scenario_rows if row['family'] == family]
        family_table[family] = {}
        for method in METHODS:
            data = [row['methods'][method] for row in subset]
            reductions = [d['mean_byte_reduction_vs_periodic'] for d in data]
            family_table[family][method] = {
                'scenarios': len(data), 'mean_byte_reduction_vs_periodic': float(np.mean(reductions)),
                'bootstrap_95pct_ci': interval(reductions, [family] * len(data), rng),
                'generated_bytes': sum(d['generated_bytes'] for d in data),
                'periodic_bytes': sum(d['periodic_bytes'] for d in data),
                'condition_pairs_using_more_bytes_than_periodic': sum(
                    d['condition_pairs_using_more_bytes_than_periodic'] for d in data),
                'adverse_scenarios': sum(d['adverse_episode'] for d in data),
                'actuation_failure_scenarios': sum(d['actuation_failure'] for d in data),
                'geometric_overlap_ticks': sum(d['geometric_overlap_ticks'] for d in data),
                'outcome_or_duration_mismatch_scenarios': sum(
                    d['outcome_or_duration_mismatch_vs_periodic'] for d in data),
                'mean_reuse_fraction': float(np.mean([d['reuse_fraction'] for d in data])),
            }
    contrasts = {}
    for name, key in [('byte_reduction', 'mean_byte_reduction_vs_periodic'),
                      ('adverse_indicator', 'adverse_episode'),
                      ('reuse_fraction', 'reuse_fraction')]:
        differences = [float(row['methods']['surface'][key]) - float(row['methods']['scalar_ttl'][key])
                       for row in scenario_rows]
        if name == 'adverse_indicator':
            discordant_surface_only = sum(d == 1 for d in differences)
            discordant_ttl_only = sum(d == -1 for d in differences)
            discordant = discordant_surface_only + discordant_ttl_only
            pvalue = binomtest(discordant_surface_only, discordant, .5).pvalue if discordant else 1.
        else:
            pvalue = signflip_paired(differences, rng)
        contrasts[name] = {'surface_minus_scalar_ttl_mean': float(np.mean(differences)),
                           'scenario_clustered_95pct_ci': interval(differences, families, rng),
                           'two_sided_p_uncorrected': float(pvalue)}
    adjusted = holm({name: value['two_sided_p_uncorrected'] for name, value in contrasts.items()})
    for name, value in contrasts.items():
        value['holm_adjusted_p'] = adjusted[name]
    report = {'status': 'COMPLETE', 'completed_utc': utc_now(),
              'registration_sha256': digest, 'confirmation_manifest_sha256': sha256_file(manifest_path),
              'analysis_source_sha256': sha256_file(Path(__file__)),
              'independent_unit': 'scenario; two conditions nested; equal-family stratified bootstrap',
              'bootstrap_replicates': BOOTSTRAP_REPLICATES, 'random_seed': RANDOM_SEED,
              'families': family_table, 'surface_vs_learned_scalar_ttl': contrasts,
              'weak_straight_braking_reported': True,
              'limitations': ['secondary analysis does not alter primary preregistered gates',
                              'closed-loop reuse fraction is not oracle-valid reuse coverage',
                              'simulated bytes and adverse episodes are not public-road safety'],
              'scenario_rows_sha256': canonical_json_hash(scenario_rows)}
    write_json_atomic(result_path, report)
    print(json.dumps({'status': report['status'], 'contrasts': contrasts,
                      'straight_lead_braking': family_table['straight_lead_braking']}, indent=2))


if __name__ == '__main__':
    main()
