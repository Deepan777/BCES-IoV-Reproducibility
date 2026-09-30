"""Fixed-N, scenario-independent audits of actual bound receiver decisions.

An unavailable oracle label is an adverse outcome if reuse was accepted.
This conservative endpoint must not be confused with observed invalidity.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, binomtest
from bces.evaluation.scenario_slot_risk import exact_binomial_upper

from bces.models.bound_policy import bind_reference, reference_query
from bces.protocol.bindings import sender_id_hash
from bces.protocol.bound_exchange import decode_objects, BoundReceiver
from bces.protocol.receiver import DecisionState
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import kinematic


def random_slot(scenario_seed, randomization_seed, phase_id, domain=315):
    """Order-independent sampling, separate from the scenario's own RNG."""
    if any(type(v) is not int or v < 0 for v in (scenario_seed, randomization_seed, phase_id)):
        raise ValueError('nonnegative integer seeds required')
    if type(domain) is not int or domain < 1:
        raise ValueError('positive slot domain required')
    return int(np.random.default_rng(np.random.SeedSequence(
        [randomization_seed, phase_id, scenario_seed])).integers(domain))


def evaluate_record(record, models, operating_rules):
    """Apply frozen bytes; validity is read only AFTER receiver decisions."""
    status = record['status']
    if status not in ('abstain', 'evaluated', 'oracle_unavailable'):
        raise ValueError('unknown slot status')
    row = {k: record[k] for k in ('scenario_id', 'slot', 'family', 'status')}
    row['methods'] = {}
    if status == 'abstain':
        row['reason'] = record['reason']
        row['valid'] = None
        for family in models:
            row['methods'][family] = {'accepted': False, 'adverse': False,
                'unknown_accepted': False, 'application_bytes': 0}
        return row
    payload = bytes.fromhex(record['payload_hex'])
    message = decode_objects(payload)
    query = reference_query(record['reference'], query_id=1,
                            sender_hash=sender_id_hash(message.sender_id))
    current = kinematic(ego_from_sample(record['current_receiver']),
                        query.reference_state.timestamp_ms+round(record['age_s']*1000))
    for family, model in models.items():
        bound = bind_reference(record['reference'], query, payload, model_sha256=model.sha256,
            view=model.view, family=family, shrinkage=operating_rules[family]['shrinkage'])
        wire = bound.encode()
        response = model.issue(wire, payload)
        receiver = BoundReceiver()
        receiver.register(wire)
        receiver.install(response, payload)
        accepted = receiver.evaluate(current, query.behavior, query.policy_hash,
                                      message.sender_id).state == DecisionState.ACCEPT_REUSE
        row['methods'][family] = {'accepted': accepted,
            'application_bytes': len(payload)+len(wire)+len(response)}
    # Neither labels nor oracle status may choose a rule or suppress reuse.
    row['valid'] = bool(record['point']['valid']) if status == 'evaluated' else None
    for result in row['methods'].values():
        result['unknown_accepted'] = result['accepted'] and row['valid'] is None
        result['adverse'] = result['accepted'] and row['valid'] is not True
    return row


def exact_upper(k, n, alpha):
    return exact_binomial_upper(k,n,alpha=alpha)


def paired_difference_interval(first_only, second_only, n, alpha=.05):
    """Conservative simultaneous CP bounds on two discordant probabilities."""
    if n < 1 or min(first_only, second_only) < 0 or first_only+second_only > n:
        raise ValueError('invalid paired counts')
    def limits(k):
        return (0. if k == 0 else float(beta.ppf(alpha/4, k, n-k+1)),
                exact_upper(k, n, alpha/4))
    a, b = limits(first_only), limits(second_only)
    return [a[0]-b[1], a[1]-b[0]]


def summarize(rows, *, expected_n, target=.05, alpha=.05, minimum_accepted=100,
              paired_alpha=.025, coverage_tolerance=.03):
    if len(rows) != expected_n or len({r['scenario_id'] for r in rows}) != expected_n:
        raise ValueError('fixed N and independent scenario identities required')
    methods = ('surface', 'scalar_ttl')
    report = {'scenarios': expected_n, 'methods': {}}
    for family in methods:
        values = [r['methods'][family] for r in rows]
        accepted = sum(bool(v['accepted']) for v in values)
        adverse = sum(bool(v['adverse']) for v in values)
        unknown = sum(bool(v['unknown_accepted']) for v in values)
        if any(v['adverse'] and not v['accepted'] or v['unknown_accepted'] and not v['adverse'] for v in values):
            raise ValueError('inconsistent adverse/unknown acceptance')
        upper = exact_upper(adverse, accepted, alpha/len(methods))
        report['methods'][family] = {'accepted': accepted, 'coverage': accepted/expected_n,
            'adverse_accepted': adverse, 'observed_invalid_accepted': adverse-unknown,
            'unknown_accepted': unknown, 'adverse_reuse_rate': adverse/accepted if accepted else None,
            'simultaneous_risk_upper': upper,
            'risk_gate_passed': accepted >= minimum_accepted and upper <= target}
    discordance = {}
    for key in ('accepted', 'adverse'):
        a = sum(bool(r['methods']['surface'][key]) and not r['methods']['scalar_ttl'][key] for r in rows)
        b = sum(bool(r['methods']['scalar_ttl'][key]) and not r['methods']['surface'][key] for r in rows)
        discordance[key] = {'surface_only': a, 'ttl_only': b, 'difference': (a-b)/expected_n,
                           'difference_interval': paired_difference_interval(a, b, expected_n)}
    a, b = discordance['adverse']['surface_only'], discordance['adverse']['ttl_only']
    p = float(binomtest(a, a+b, .5, alternative='less').pvalue) if a+b else 1.
    interval = discordance['accepted']['difference_interval']
    report.update({'paired': discordance, 'paired_adverse_pvalue': p,
        'paired_adverse_superiority': p <= paired_alpha,
        'coverage_equivalence': interval[0] >= -coverage_tolerance and interval[1] <= coverage_tolerance,
        'both_risk_gates_passed': all(v['risk_gate_passed'] for v in report['methods'].values()),
        'scientific_success': False, 'manuscript_allowed': False})
    return report
