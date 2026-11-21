#!/usr/bin/env python3
"""Post hoc diagnostic for the closed-loop v2 BCES-minus-TTL byte-reduction contrast.

Explains why the registered family-stratified bootstrap interval excludes zero while the
registered paired sign-flip test does not reject. Reads the frozen branch artifacts (hash-checked
against the run manifest), leaves the registered secondary analysis untouched, and writes a
separate JSON report beneath reproduced/. This analysis was not prespecified.

Run after materializing the data: python reproduce.py run scripts/157_diagnose_closed_loop_v2_byte_contrast.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.closed_loop_confirmation import CONDITIONS, FAMILIES
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import utc_now, write_json_atomic

STUDY = ROOT / 'outputs/study_b/closed_loop_confirmation_v2'
REGISTRATION = STUDY / 'preregistration.json'
BRANCHES = STUDY / 'confirmation'
OUTPUT = ROOT / 'reproduced' / 'source_query_byte_contrast_diagnostic.json'
REPLICATES = 20_000
RANDOM_SEED = 20261117


def load(manifest, seed, condition, method):
    name = f'branch_{seed}_{condition}_{method}.json.gz'
    path = BRANCHES / name
    if sha256_file(path) != manifest['artifact_sha256'][name]:
        raise RuntimeError('branch artifact changed: ' + name)
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)['branch']


def reduction(manifest, seed, method):
    values = []
    for condition in CONDITIONS:
        branch = load(manifest, seed, condition, method)
        periodic = load(manifest, seed, condition, 'periodic_payload')
        values.append(1. - branch['traffic']['generated_bytes'] / periodic['traffic']['generated_bytes'])
    return sum(values) / len(values)


def main():
    registration = json.loads(REGISTRATION.read_text(encoding='utf-8'))
    manifest = json.loads((BRANCHES / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'COMPLETE' or manifest['registration_sha256'] != sha256_file(REGISTRATION):
        raise RuntimeError('v2 confirmation incomplete or registration mismatch')
    families, differences = [], []
    for assignment in registration['scenario_sampling']['selected']:
        seed = assignment['seed']
        families.append(assignment['family'])
        differences.append(reduction(manifest, seed, 'surface') - reduction(manifest, seed, 'scalar_ttl'))
    families = np.asarray(families)
    d = np.asarray(differences)
    n = len(d)
    present = [f for f in FAMILIES if np.any(families == f)]
    per_family = {}
    stratified_variance = 0.
    between_sum_of_squares = 0.
    for family in present:
        x = d[families == family]
        weight = len(x) / n
        stratified_variance += weight ** 2 * x.var(ddof=1) / len(x)
        between_sum_of_squares += len(x) * x.mean() ** 2
        per_family[family] = {'scenarios': int(len(x)), 'mean_pp': 100 * float(x.mean()),
                              'sd_pp': 100 * float(x.std(ddof=1))}
    rng = np.random.default_rng(RANDOM_SEED)
    positions = [np.flatnonzero(families == f) for f in present]
    stratified = np.empty(REPLICATES)
    pooled = np.empty(REPLICATES)
    null = np.empty(REPLICATES)
    for start in range(0, REPLICATES, 500):
        k = min(500, REPLICATES - start)
        stratified[start:start + k] = np.concatenate(
            [d[rng.choice(p, size=(k, len(p)))] for p in positions], axis=1).mean(axis=1)
        pooled[start:start + k] = d[rng.choice(n, size=(k, n))].mean(axis=1)
        null[start:start + k] = (rng.choice((-1., 1.), size=(k, n)) * d).mean(axis=1)
    report = {
        'status': 'COMPLETE', 'completed_utc': utc_now(), 'prespecified': False,
        'purpose': 'explain bootstrap/sign-flip disagreement for the byte-reduction contrast',
        'registration_sha256': manifest['registration_sha256'],
        'analysis_source_sha256': sha256_file(Path(__file__)),
        'scenarios': n, 'mean_difference_pp': 100 * float(d.mean()), 'families': per_family,
        'stratified_standard_error_pp': 100 * float(np.sqrt(stratified_variance)),
        'signflip_null_sd_pp': 100 * float(np.sqrt((d ** 2).sum()) / n),
        'unstratified_standard_error_pp': 100 * float(d.std(ddof=1) / np.sqrt(n)),
        'between_family_share_of_sum_of_squares': float(between_sum_of_squares / (d ** 2).sum()),
        'stratified_bootstrap_95pct_ci_pp': [100 * float(q) for q in np.quantile(stratified, [.025, .975])],
        'unstratified_bootstrap_95pct_ci_pp': [100 * float(q) for q in np.quantile(pooled, [.025, .975])],
        'signflip_two_sided_p': float((np.sum(np.abs(null) >= abs(d.mean()) - 1e-15) + 1) / (REPLICATES + 1)),
        'random_seed': RANDOM_SEED, 'replicates': REPLICATES,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
