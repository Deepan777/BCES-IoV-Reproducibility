#!/usr/bin/env python3
"""Build a development report from hash-bound results, never manuscript claims."""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic


def main():
    state = git_state(ROOT)
    if state['dirty']:
        raise RuntimeError('clean committed reporting source required')
    base = ROOT / 'outputs/study_b'
    paths = {
        'generation': base / 'controlled_development_v2/run_manifest.json',
        'diagnostics': base / 'diagnostics_controlled_v2/diagnostics.json',
        'audit': base / 'controlled_information_audit_v2.json',
    }
    reports = {key: json.loads(path.read_text(encoding='utf-8')) for key, path in paths.items()}
    data, diagnostic, audit = (reports[key] for key in ('generation', 'diagnostics', 'audit'))
    for report in (diagnostic, audit):
        if report['source_manifest_sha256'] != sha256_file(paths['generation']):
            raise ValueError('source generation mismatch')
    if audit['diagnostics_sha256'] != sha256_file(paths['diagnostics']):
        raise ValueError('audit/diagnostic mismatch')
    config_path = ROOT / 'configs/training/decision_surface_controlled_v2.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    g = diagnostic['geometry']
    information = max(diagnostic['information'][key]['validation']['auroc']
                      for key in ('frozen_inputs', 'reference_margins'))
    ready = (information >= config['minimum_diagnostic_auroc'] and
             g['surface_accepted_valid'] - g['ttl_accepted_valid'] >= config['minimum_geometry_extra_valid_points'])
    lines = ['# Controlled Study B development results', '',
             'Engineering and diagnostic evidence only. Scientific success and manuscript gates remain unmet.', '',
             '## Generated evidence', '',
             '| Partition | Scenarios | References | Evaluated | Valid | Invalid |',
             '|---|---:|---:|---:|---:|---:|']
    for split in ('train', 'validation', 'calibration'):
        c = data['counts']
        n, y = c[f'{split}:points'], c[f'{split}:valid']
        lines.append(f"| {split} | {c[f'{split}:scenarios']} | {c[f'{split}:references']} | {n} | {y} | {n-y} |")
    lines.extend(['', f"Excluded requests: `{json.dumps(data['rejections'], sort_keys=True)}`.",
                  'Coverage is conditional on evaluated feasible requests. Three continuations remain one statistical scenario.', '',
                  f"Overlapping failure counts: `{json.dumps(diagnostic['violation_taxonomy'], sort_keys=True)}`.",
                  'A regret-only bound would not establish the separate absolute-risk and trajectory-deviation requirements.', '',
                  '## Information', '',
                  '| Feature view | Validation AUROC | Validation AUPRC | Calibration coverage | Calibration UAR | Finite-sample target met |',
                  '|---|---:|---:|---:|---:|---|'])
    for key, result in diagnostic['information'].items():
        v, c = result['validation'], result['calibration_fixed_rule']
        uar = 'undefined' if c['uar'] is None else f"{c['uar']:.6f}"
        lines.append(f"| {key} | {v['auroc']:.6f} | {v['auprc']:.6f} | {c['coverage']:.6f} | {uar} | {c['finite_sample_target_met']} |")
    lines.extend(['', 'Models were selected by validation AUPRC using the frozen grid. Current-receiver summaries are not sender-time surface inputs.', '',
                  '## Geometry and ambiguity', '',
                  f"Label-informed fits: {g['references']} references, {g['points']} points, {g['valid']} valid points.",
                  f"Ideal fixed-normal surface accepts {g['surface_accepted_valid']} valid points; ideal scalar TTL accepts {g['ttl_accepted_valid']}.",
                  f"Solver-optimal fits: {g['surface_optimal_solutions']}; surface abstentions: {g['surface_abstentions']}.", '',
                  'These fits use the evaluated labels; they are representability diagnostics, not learned performance.', ''])
    for key, value in audit['receiver_comparison_counts'].items():
        lines.append(f'- {key}: {value}')
    differences = [row['surface_accepted_valid'] - row['ttl_accepted_valid']
                   for row in audit['geometry_per_reference']]
    lines.extend(['', f"Per-reference surface versus TTL: {sum(d > 0 for d in differences)} better, "
                  f"{sum(d == 0 for d in differences)} tied, {sum(d < 0 for d in differences)} worse."])
    lines.extend(['', f"Learning prerequisite: **{'PASS' if ready else 'DIAGNOSTIC_STOP'}**.", '',
                  'The input AUROC and positive geometric-headroom requirements are unchanged. Passing them would authorize only the development learning diagnostic.'])
    learning_path = base / 'decision_surface_controlled_v2/run_manifest.json'
    if learning_path.exists():
        learning = json.loads(learning_path.read_text(encoding='utf-8'))
        if (learning['data_manifest_sha256'] != sha256_file(paths['generation']) or
                learning['diagnostics_sha256'] != sha256_file(paths['diagnostics'])):
            raise ValueError('learning provenance mismatch')
        paths['learning'] = learning_path
        lines.extend(['', '## Learning', '', '| View / family | Seed | Validation coverage | Validation UAR |',
                      '|---|---:|---:|---:|'])
        for key, group in learning['runs'].items():
            for run in group['runs']:
                metric = run['metrics']['validation']
                lines.append(f"| {key} | {run['seed']} | {metric['coverage']:.6f} | {metric['uar']} |")
        lines.extend(['', 'Calibration target attainment: `' + json.dumps(
            {key: value['target_met'] for key, value in learning['calibration'].items()}, sort_keys=True) + '`.'])
        completed_runs = [run for group in learning['runs'].values() for run in group['runs']]
        lines.extend(['', f"Completed training runs: {len(completed_runs)}. "
                      f"Maximum parameter count: {max(r['parameter_count'] for r in completed_runs)}. "
                      f"Peak PyTorch CUDA allocated bytes: {max(r['peak_cuda_allocated_bytes'] for r in completed_runs)}.",
                      f"Workspace bytes measured at learning completion: {learning['budget']['workspace_bytes']}."])
        audit_path = base / 'controlled_learning_audit_v2.json'
        if audit_path.exists():
            paired = json.loads(audit_path.read_text(encoding='utf-8'))
            if paired['training_manifest_sha256'] != sha256_file(learning_path):
                raise ValueError('paired audit/training mismatch')
            paths['paired_learning_audit'] = audit_path
            lines.extend(['', '### All-seed paired validation ranking diagnostic', '',
                          'Not a deployed wire rule, calibrated safety guarantee or confirmatory significance test. Negative differences favor the surface.', '',
                          '| Features | Seed | Coverage | Surface minus TTL invalid-reuse rate | Descriptive paired scenario interval |',
                          '|---|---:|---:|---:|---|'])
            for row in paired['paired_ranking_comparisons']:
                if not row['both_models_have_enough_nonnegative_scores']:
                    continue
                interval = row['paired_scenario_bootstrap_interval_descriptive']
                lines.append(f"| {row['view']} | {row['seed']} | {row['requested_coverage']:.0%} | "
                             f"{row['surface_minus_ttl_uar']:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] |")
    elif ready:
        lines.extend(['', 'Learning has not produced a completed manifest. No trained-model result is claimed.'])
    else:
        lines.extend(['', 'The learning experiment must not run past its failed prerequisite. No new trained-model result is claimed.'])
    test_path = base / 'controlled_tests_v2.xml'
    if test_path.exists():
        suite = ET.parse(test_path).getroot().find('testsuite')
        if suite is None:
            raise ValueError('missing test suite summary')
        paths['tests'] = test_path
        lines.extend(['', '## Software verification', '',
                      f"Tests: {suite.attrib['tests']}; failures: {suite.attrib['failures']}; "
                      f"errors: {suite.attrib['errors']}; skipped: {suite.attrib['skipped']}. "
                      'Software verification is separate from the scientific-success gate.'])
    lines.extend(['', '## Reproduction commands', '', '```text',
                  'python -u scripts/16_generate_controlled_development.py',
                  'python -u scripts/12_information_geometry_diagnostics.py --config configs/evaluation/study_b_controlled_v2.yaml',
                  'python -u scripts/13_train_decision_surfaces.py --config configs/training/decision_surface_controlled_v2.yaml',
                  'python -u scripts/17_audit_controlled_information.py',
                  'python -u scripts/19_audit_controlled_learning.py',
                  'python -u scripts/18_report_controlled_development.py', '```', '',
                  'Use the source revisions recorded by each manifest and fresh versioned output paths; completed outputs are immutable.', '',
                  '## Claim limits and next gate', '',
                  'These are designed-distribution, discrete-time open-loop labels. They do not measure closed-loop collisions, establish real-road safety, or validate uniform uncertainty envelopes.',
                  'Full packet/query and margin-sidecar accounting, independent powered confirmation, matched-risk TTL comparisons and closed-loop evaluation remain required. Calibration cannot count correlated time steps as independent evidence.',
                  'Original test data were not accessed by these runs. Earlier observational and simulator artifacts remain preserved and are not pooled into this result.', '',
                  '## Provenance', ''])
    for key, path in paths.items():
        lines.append(f'- {key}: `{path.relative_to(ROOT).as_posix()}`; SHA-256 `{sha256_file(path)}`')
    destination = base / 'controlled_development_report_v2'
    if destination.exists():
        raise FileExistsError('completed report directory is immutable')
    destination.mkdir()
    (destination / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json_atomic(destination / 'manifest.json', {
        'scope': 'development_only', 'scientific_success': False, 'manuscript_allowed': False,
        'learning_prerequisite_passed': ready, 'source_sha256': {k: sha256_file(p) for k, p in paths.items()},
        'learning_config_sha256': sha256_file(config_path),
        'report_sha256': sha256_file(destination / 'report.md'), 'git': state, 'completed_utc': utc_now()})
    print(destination / 'report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
