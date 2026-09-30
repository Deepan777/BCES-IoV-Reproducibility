"""Conservative, past-only ID repair for counterfactual replay diagnostics.

Associations are estimates, not identity ground truth. Unresolved detections are
retained. This module does not turn a perception stream into physical truth or
make the forecasting target the actual sensing vehicle.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

from bces.geometry.drift import wrap_angle_rad


KINEMATIC_FIELDS = ('x', 'y', 'v_x', 'v_y', 'theta', 'length', 'width')


@dataclass(frozen=True)
class AssociationConfig:
    history_frames: int = 3
    maximum_frame_gap_ms: int = 150
    maximum_distance_m: float = 2.0
    maximum_lateral_m: float = .9
    maximum_velocity_difference_mps: float = 2.5
    maximum_heading_difference_rad: float = .35
    maximum_relative_size_difference: float = .35

    def __post_init__(self):
        if self.history_frames < 2 or any(value <= 0 for value in asdict(self).values()):
            raise ValueError('positive association gates and at least two frames required')


def complete_kinematics(row):
    try:
        return (all(math.isfinite(float(row[key])) for key in KINEMATIC_FIELDS)
                and float(row['length']) > 0 and float(row['width']) > 0)
    except (KeyError, TypeError, ValueError):
        return False


def compatible(first, second, config):
    """No imputation: missing kinematics cannot authorize deleting a detection."""
    if not complete_kinematics(first) or not complete_kinematics(second):
        return False
    if first.get('type') != second.get('type'):
        return False
    dx, dy = float(second['x'])-float(first['x']), float(second['y'])-float(first['y'])
    if math.hypot(dx, dy) > config.maximum_distance_m:
        return False
    heading = float(first['theta'])
    if abs(-math.sin(heading)*dx+math.cos(heading)*dy) > config.maximum_lateral_m:
        return False
    if math.hypot(float(second['v_x'])-float(first['v_x']),
                  float(second['v_y'])-float(first['v_y'])) > config.maximum_velocity_difference_mps:
        return False
    if abs(wrap_angle_rad(float(second['theta'])-heading)) > config.maximum_heading_difference_rad:
        return False
    return all(abs(float(second[key])-float(first[key]))/max(float(second[key]),float(first[key]))
               <= config.maximum_relative_size_difference for key in ('length','width'))


def _index(snapshots):
    result = {}
    for timestamp, rows in snapshots.items():
        # Real infrastructure rows can reuse one numeric ID across classes.
        # Always include class (not only on colliding frames) for continuity.
        indexed = {f"{row.get('type', '')}:{row['id']}":row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError('duplicate per-view class/ID within a timestamp')
        result[timestamp] = indexed
    return result


def _matches(anchor_history, other_history, times, config):
    """Mutually unique gates now, then corroborate the same IDs in past frames."""
    if len(times) != config.history_frames:
        return {}
    if any(b-a > config.maximum_frame_gap_ms for a,b in zip(times,times[1:])):
        return {}
    anchor, other = anchor_history[times[-1]], other_history[times[-1]]
    candidates = {key:[other_id for other_id,row in other.items() if compatible(value,row,config)]
                  for key,value in anchor.items()}
    frequency = Counter(key for values in candidates.values() for key in values)
    matches = {}
    for key, values in candidates.items():
        if len(values) != 1 or frequency[values[0]] != 1:
            continue
        other_id = values[0]
        if all(key in anchor_history[t] and other_id in other_history[t]
               and compatible(anchor_history[t][key],other_history[t][other_id],config) for t in times):
            matches[other_id] = key
    return matches


def associate_replay_views(ego_snapshots, vehicle_snapshots, infrastructure_snapshots,
                           focal_id, config=None):
    """Return source-namespaced copies and a complete correspondence audit.

Only the focal ego-view trajectory is an association input. Other ego-view
objects (the withheld evaluation proxy) never affect local/remote associations.
The local/remote match at receipt is frozen in the payload's track identifiers;
future matches cannot retroactively rewrite an old message.
"""
    config = config or AssociationConfig()
    focal = {t:{str(row['id']):row for row in rows if str(row['id']) == str(focal_id)}
             for t,rows in ego_snapshots.items()}
    views = {'vehicle':_index(vehicle_snapshots),'infrastructure':_index(infrastructure_snapshots)}
    sanitized, self_ids, audit = {}, {}, {'config':asdict(config),'counts':{},'frames':[]}
    counts = Counter()
    for source, snapshots in views.items():
        sanitized[source], self_ids[source] = {}, {}
        available = sorted(set(snapshots) & set(focal))
        preceding = {t:available[max(0,i-config.history_frames+1):i+1] for i,t in enumerate(available)}
        for t, rows in sorted(snapshots.items()):
            matches = _matches(focal, snapshots, preceding.get(t,[]), config)
            self_ids[source][t] = set(matches)
            sanitized[source][t] = {
                key:{**row,'id':str(focal_id) if key in matches else f'{source}:{key}'}
                for key,row in rows.items()}
            counts[f'{source}:self_matches'] += len(matches)
            counts[f'{source}:frames'] += 1
            audit['frames'].append({'timestamp_ms':t,'source':source,'self_ids':sorted(matches)})
    # Exclude estimated self before mutually unique cross-source association.
    remaining = {source:{t:{key:row for key,row in snapshots.items() if key not in self_ids[source][t]}
                         for t,snapshots in view.items()} for source,view in views.items()}
    available = sorted(set(remaining['vehicle']) & set(remaining['infrastructure']))
    for index,t in enumerate(available):
        times = available[max(0,index-config.history_frames+1):index+1]
        matches = _matches(remaining['vehicle'],remaining['infrastructure'],times,config)
        for remote_id, local_id in matches.items():
            sanitized['infrastructure'][t][remote_id]['id'] = f'vehicle:{local_id}'
        counts['cross_source_matches'] += len(matches)
        audit['frames'].append({'timestamp_ms':t,'source':'cross_source','remote_to_local':matches})
    audit['counts'] = dict(counts)
    result = {source:{t:tuple(rows.values()) for t,rows in view.items()} for source,view in sanitized.items()}
    return result['vehicle'],result['infrastructure'],audit


def audit_receiver_roles(ego_snapshots, focal_id):
    counts = Counter()
    av_ids = set()
    for rows in ego_snapshots.values():
        for row in rows:
            if row.get('tag') == 'AV':
                av_ids.add(str(row['id']))
                counts['av_rows'] += 1
                counts['av_complete_kinematic_rows'] += int(complete_kinematics(row))
            if str(row.get('id')) == str(focal_id):
                counts['focal_rows'] += 1
                counts['focal_is_tagged_av_rows'] += int(row.get('tag') == 'AV')
    return {'counts':dict(counts),'av_ids':sorted(av_ids),'selected_focal_id':str(focal_id),
            'receiver_role_verified':bool(counts['focal_rows']) and
                counts['focal_rows'] == counts['focal_is_tagged_av_rows'],
            'source_is_complete_ground_truth':False}
