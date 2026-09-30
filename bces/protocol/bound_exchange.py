"""Versioned application framing for post-reception expiry queries.

The BCES extension remains exactly 48 bytes. Query/response framing and margin
bytes are additional charged traffic. SHA-256 binds context, not authentication.
The normal V2X security envelope is still required in a deployment.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass

from bces.data.objects import ObjectMessage, PerceivedObject
from bces.geometry.drift import DriftScales, KinematicState
from bces.protocol.bindings import digest32, sender_id_hash
from bces.protocol.codec import decode_packet
from bces.protocol.query import Behavior, PathPoint, ReceiverQuery
from bces.protocol.receiver import ReceiverStateMachine, ReceiverDecision, DecisionState, EvidenceMode, _serial_relation

QUERY_HEADER = struct.Struct('!4sBIH')
RESPONSE_HEADER = struct.Struct('!4sB32s32sBH')
MARGINS = struct.Struct('!29f')
QUERY_IDS = struct.Struct('!IIBBIBBBB')
QUERY_STATE = struct.Struct('!6dQ')
QUERY_PATH = struct.Struct('!72d')
QUERY_SCALES = struct.Struct('!7d')
QUERY_CONTEXT = struct.Struct('!32s32sBBdQddBQQ')
CAP = 65535 / 32768
STEP = CAP / 255
GUARD = .05


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def decode_objects(raw):
    obj = json.loads(raw)
    obj['objects'] = tuple(PerceivedObject(**row) for row in obj['objects'])
    result = ObjectMessage(**obj)
    if result.encode() != raw:
        raise ValueError('noncanonical object payload')
    return result


@dataclass(frozen=True)
class BoundQuery:
    query: ReceiverQuery
    payload_sha256: str
    model_sha256: str
    feature_view: str
    family: str
    shrinkage: float
    margins: tuple[float, ...]
    context_available_ms: int
    occlusion_proxy: float
    estimated_delay_s: float
    map_context_flags: int
    generated_ms: int | None = None
    payload_received_ms: int | None = None

    def __post_init__(self):
        for value in (self.payload_sha256, self.model_sha256):
            if len(value) != 64 or bytes.fromhex(value).hex() != value:
                raise ValueError('canonical SHA-256 required')
        if self.feature_view not in ('frozen_inputs', 'reference_margins') or self.family not in ('surface', 'scalar_ttl'):
            raise ValueError('unsupported feature/model contract')
        if self.generated_ms is None:
            object.__setattr__(self, 'generated_ms', self.query.reference_state.timestamp_ms)
        if self.payload_received_ms is None:
            object.__setattr__(self, 'payload_received_ms', self.query.reference_state.timestamp_ms)
        if (type(self.generated_ms) is not int or type(self.payload_received_ms) is not int
                or not 0 <= self.payload_received_ms <= self.generated_ms
                or self.query.reference_state.timestamp_ms > self.generated_ms):
            raise ValueError('query precedes payload receipt or reference state')
        if type(self.context_available_ms) is not int or not 0 <= self.context_available_ms <= self.generated_ms:
            raise ValueError('future or invalid context availability')
        for name in ('shrinkage', 'occlusion_proxy', 'estimated_delay_s'):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError('finite nonnegative context required')
            object.__setattr__(self, name, value)
        if self.shrinkage > CAP or self.occlusion_proxy > 1 or type(self.map_context_flags) is not int or not 0 <= self.map_context_flags <= 255:
            raise ValueError('context outside schema range')
        if self.feature_view == 'reference_margins':
            if len(self.margins) != 29 or not all(math.isfinite(x) for x in self.margins):
                raise ValueError('29 finite observable margin features required')
            try:
                values = MARGINS.unpack(MARGINS.pack(*self.margins))
            except (OverflowError, struct.error) as exc:
                raise ValueError('margin outside float32 range') from exc
            if not all(math.isfinite(x) for x in values):
                raise ValueError('nonfinite encoded margin')
            object.__setattr__(self, 'margins', values)
        elif self.margins:
            raise ValueError('base-feature query must not include a margin sidecar')

    def encode(self):
        q = self.query
        base = (QUERY_IDS.pack(q.query_id,q.expected_sender_id_hash,int(q.behavior),q.risk_class,
                    q.policy_hash,q.calibration_id,q.protocol_version,q.drift_schema_id,q.normal_codebook_id)
                + QUERY_STATE.pack(*asdict(q.reference_state).values())
                + QUERY_PATH.pack(*(value for point in q.reference_path for value in asdict(point).values()))
                + QUERY_SCALES.pack(*q.drift_scales.as_tuple())
                + QUERY_CONTEXT.pack(bytes.fromhex(self.payload_sha256),bytes.fromhex(self.model_sha256),
                    int(self.feature_view=='reference_margins'),int(self.family=='scalar_ttl'),
                    self.shrinkage,self.context_available_ms,self.occlusion_proxy,self.estimated_delay_s,
                    self.map_context_flags,self.generated_ms,self.payload_received_ms))
        side = MARGINS.pack(*self.margins) if self.margins else b''
        return QUERY_HEADER.pack(b'BQRY', 2, len(base), len(side)) + base + side

    @classmethod
    def decode(cls, raw):
        try:
            return cls._decode(raw)
        except (KeyError, TypeError, UnicodeError, struct.error) as exc:
            raise ValueError('malformed query schema') from exc

    @classmethod
    def _decode(cls, raw):
        if not isinstance(raw, bytes) or len(raw) < QUERY_HEADER.size:
            raise ValueError('truncated query')
        magic, version, n, s = QUERY_HEADER.unpack_from(raw)
        expected_n = sum(p.size for p in (QUERY_IDS,QUERY_STATE,QUERY_PATH,QUERY_SCALES,QUERY_CONTEXT))
        if magic != b'BQRY' or version != 2 or s not in (0, MARGINS.size) or n != expected_n or len(raw) != QUERY_HEADER.size + n + s:
            raise ValueError('invalid query frame')
        cursor = QUERY_HEADER.size
        groups = []
        for packer in (QUERY_IDS,QUERY_STATE,QUERY_PATH,QUERY_SCALES,QUERY_CONTEXT):
            groups.append(packer.unpack_from(raw,cursor)); cursor += packer.size
        ids,state,path,scales,context = groups
        if context[2] not in (0,1) or context[3] not in (0,1):
            raise ValueError('unsupported feature or model family')
        q = ReceiverQuery(ids[0],ids[1],KinematicState(*state),
            tuple(PathPoint(*path[i:i+6]) for i in range(0,72,6)),ids[2],ids[3],ids[4],ids[5],
            DriftScales(*scales),protocol_version=ids[6],drift_schema_id=ids[7],normal_codebook_id=ids[8])
        result = cls(q,context[0].hex(),context[1].hex(),
            'reference_margins' if context[2] else 'frozen_inputs',
            'scalar_ttl' if context[3] else 'surface',context[4],MARGINS.unpack(raw[-s:]) if s else (),
            context[5],context[6],context[7],context[8],context[9],context[10])
        if result.encode() != raw:
            raise ValueError('noncanonical query')
        return result

    def validate_payload(self, payload):
        if hashlib.sha256(payload).hexdigest() != self.payload_sha256:
            raise ValueError('exact payload binding mismatch')
        message = decode_objects(payload)
        if message.timestamp_ms > self.payload_received_ms:
            raise ValueError('payload receipt precedes generation')
        if sender_id_hash(message.sender_id) != self.query.expected_sender_id_hash:
            raise ValueError('sender binding mismatch')
        delay = (self.query.reference_state.timestamp_ms - message.timestamp_ms) / 1000
        if delay < 0 or not math.isclose(delay, self.estimated_delay_s, abs_tol=1e-9, rel_tol=0):
            raise ValueError('message age/context mismatch')
        return message


def encode_response(query_wire, extension):
    bound = BoundQuery.decode(query_wire)
    kind = 1 if bound.family == 'surface' else 2
    if len(extension) != (48 if kind == 1 else 2):
        raise ValueError('incorrect method extension length')
    return RESPONSE_HEADER.pack(b'BRSP', 1, hashlib.sha256(query_wire).digest(),
                                bytes.fromhex(bound.payload_sha256), kind, len(extension)) + extension


def decode_response(raw, query_wire):
    bound = BoundQuery.decode(query_wire)
    if not isinstance(raw, bytes) or len(raw) < RESPONSE_HEADER.size:
        raise ValueError('truncated response')
    magic, version, qhash, phash, kind, n = RESPONSE_HEADER.unpack_from(raw)
    expected = 1 if bound.family == 'surface' else 2
    if (magic != b'BRSP' or version != 1 or qhash != hashlib.sha256(query_wire).digest()
            or phash != bytes.fromhex(bound.payload_sha256) or kind != expected
            or n != (48 if kind == 1 else 2) or len(raw) != RESPONSE_HEADER.size + n):
        raise ValueError('response context/length mismatch')
    extension = raw[RESPONSE_HEADER.size:]
    if kind == 1:
        decode_packet(extension)
    elif extension[0] not in (0, 1):
        raise ValueError('invalid scalar origin flag')
    return extension


def scalar_extension(offset, origin_permitted):
    if not math.isfinite(offset) or not 0 <= offset <= CAP or type(origin_permitted) is not bool:
        raise ValueError('invalid scalar expiry token')
    return bytes((int(origin_permitted), min(255, math.floor(offset / STEP))))


class BoundReceiver:
    """One outstanding monotone query per instance/sender; no query-ID rebinding.

    Query bytes are retained once, not retransmitted on every planning tick.
    A fresh query revokes the previous response until its own reply is installed.
    """
    def __init__(self):
        self.query_wire = None
        self.bound = None
        self.extension = None
        self.payload = None
        self.machine = ReceiverStateMachine(refresh_guard_band=GUARD)

    def register(self, query_wire):
        bound = BoundQuery.decode(query_wire)
        if self.bound is not None:
            if bound.query.expected_sender_id_hash != self.bound.query.expected_sender_id_hash:
                raise ValueError('receiver instance is sender-bound')
            relation = _serial_relation(bound.query.query_id, self.bound.query.query_id)
            if relation < 0 or (relation == 0 and query_wire != self.query_wire):
                raise ValueError('stale or rebound query ID')
            if relation == 0:
                return
        self.bound, self.query_wire = bound, query_wire
        self.extension = self.payload = None
        self.machine = ReceiverStateMachine(refresh_guard_band=GUARD)

    def install(self, response, payload):
        if self.bound is None:
            raise ValueError('no outstanding query')
        message = self.bound.validate_payload(payload)
        extension = decode_response(response, self.query_wire)
        if self.extension is not None and extension != self.extension:
            raise ValueError('conflicting response for immutable query')
        if self.bound.family == 'surface':
            q = self.bound.query
            check = self.machine.evaluate(encoded_packet=extension, cooperative_payload=payload,
                query=q, current_state=q.reference_state, current_behavior=q.behavior,
                current_policy_hash=q.policy_hash, actual_sender_id=message.sender_id)
            if check.state == DecisionState.INVALID:
                raise ValueError(check.reason)
        self.extension, self.payload = extension, payload

    def evaluate(self, current_state, current_behavior, current_policy_hash, actual_sender_id):
        if self.extension is None:
            return ReceiverDecision(DecisionState.REFRESH, EvidenceMode.LOCAL_ONLY, True, 'awaiting_bound_response')
        q = self.bound.query
        if self.bound.family == 'surface':
            return self.machine.evaluate(encoded_packet=self.extension, cooperative_payload=self.payload,
                query=q, current_state=current_state, current_behavior=current_behavior,
                current_policy_hash=current_policy_hash, actual_sender_id=actual_sender_id)
        try:
            sender = sender_id_hash(actual_sender_id)
            behavior = Behavior(current_behavior)
        except (TypeError, ValueError):
            return ReceiverDecision(DecisionState.INVALID, EvidenceMode.LOCAL_ONLY, False, 'invalid_scalar_runtime_context')
        if (type(current_policy_hash) is not int or current_policy_hash != q.policy_hash or behavior != q.behavior
                or sender != q.expected_sender_id_hash
                or current_state.timestamp_ms < q.reference_state.timestamp_ms):
            return ReceiverDecision(DecisionState.INVALID, EvidenceMode.LOCAL_ONLY, False, 'scalar_runtime_binding_mismatch')
        slack = self.extension[1] * STEP - (current_state.timestamp_ms-q.reference_state.timestamp_ms)/1000/q.drift_scales.elapsed_s
        accepted = bool(self.extension[0]) and slack >= GUARD
        return ReceiverDecision(DecisionState.ACCEPT_REUSE if accepted else DecisionState.REFRESH,
            EvidenceMode.COOPERATIVE_REUSE if accepted else EvidenceMode.LOCAL_ONLY,
            not accepted, 'scalar_inside' if accepted else 'scalar_refresh', slack)
