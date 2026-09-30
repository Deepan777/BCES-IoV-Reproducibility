"""Causal byte-exact datagram channel; lost/duplicate frames consume bandwidth."""
from __future__ import annotations

import heapq
import math
import random
import struct
from collections import Counter
from dataclasses import dataclass

from bces.network.events import NetworkCondition
from bces.protocol.bound_exchange import QUERY_HEADER, RESPONSE_HEADER

DATAGRAM_HEADER = struct.Struct('!4sBBI')
KINDS = {'payload': 0, 'query': 1, 'response': 2, 'refresh': 3}
REFRESH = struct.Struct('!4sBII')


def refresh_control(sender_hash, last_query_id):
    return REFRESH.pack(b'BREF', 1, sender_hash, last_query_id)


def application_components(kind, body):
    if kind == 'query':
        _, _, n, s = QUERY_HEADER.unpack_from(body)
        if len(body) != QUERY_HEADER.size+n+s:
            raise ValueError('query accounting length mismatch')
        return {'query_framing': QUERY_HEADER.size, 'query_context': n, 'margin_sidecar': s}
    if kind == 'response':
        _, _, _, _, method, n = RESPONSE_HEADER.unpack_from(body)
        if len(body) != RESPONSE_HEADER.size+n or method not in (1, 2):
            raise ValueError('response accounting length mismatch')
        return {'response_binding_framing': RESPONSE_HEADER.size,
                'bces_extension' if method == 1 else 'ttl_extension': n}
    return {'object_payload' if kind == 'payload' else 'refresh_control': len(body)}


@dataclass(frozen=True)
class WireDelivery:
    serial: int
    kind: str
    body: bytes
    wire: bytes
    generated_ms: float
    delivered_ms: float
    duplicate: bool
    reordered: bool
    attempt: int


class WireChannel:
    """One shared serialized application channel, with a modeled security envelope.

    The security bytes are counted zero-filled placeholders, NOT implemented
    signatures. No physical/MAC framing or radio interference is claimed.
    Calling send only after receiving a query enforces request/response causality.
    """
    def __init__(self, condition: NetworkCondition, *, seed: int, security_bytes: int):
        if type(security_bytes) is not int or not 0 <= security_bytes <= 65536:
            raise ValueError('invalid modeled security envelope')
        if not all(math.isfinite(float(x)) for x in vars(condition).values()):
            raise ValueError('finite network parameters required')
        self.condition, self.security_bytes = condition, security_bytes
        self.rng = random.Random(seed)
        self.queue_ms = self.clock_ms = 0.
        self.scheduled = []
        self.schedule_serial = 0
        self.serial = 0
        self.pending = []
        self.counts, self.components = Counter(), Counter()
        self.bad = False
        self.latencies = []
        self.delivered_serials, self.discarded_serials = set(), set()

    def send(self, kind, body, generated_ms, *, attempt=0):
        if kind not in KINDS or not isinstance(body, bytes) or not body or not math.isfinite(generated_ms) or generated_ms < self.clock_ms:
            raise ValueError('valid bytes and causal generation time required')
        if type(attempt) is not int or not 0 <= attempt <= 0xffffffff:
            raise ValueError('invalid retry attempt')
        if generated_ms > self.clock_ms:
            self.schedule_serial += 1
            heapq.heappush(self.scheduled,(generated_ms,self.schedule_serial,kind,body,attempt))
            return
        self._transmit(kind,body,generated_ms,attempt)

    def _transmit(self,kind,body,generated_ms,attempt):
        parts = application_components(kind, body)
        parts.update(datagram_header=DATAGRAM_HEADER.size, modeled_security=self.security_bytes)
        wire = DATAGRAM_HEADER.pack(b'BDGM', 1, KINDS[kind], attempt) + body + bytes(self.security_bytes)
        if sum(parts.values()) != len(wire):
            raise AssertionError('component accounting mismatch')
        copies = 2 if self.rng.random() < self.condition.duplicate_probability else 1
        for copy in range(copies):
            self.serial += 1
            self.counts['generated_bytes'] += len(wire)
            self.counts['generated_packets'] += 1
            self.counts[f'{kind}_packets'] += 1
            self.counts['duplicate_bytes'] += len(wire) if copy else 0
            self.counts['retry_bytes'] += len(wire) if attempt else 0
            self.components.update(parts)
            self.queue_ms = max(generated_ms, self.queue_ms) + len(wire)*8/(self.condition.bandwidth_mbps*1000)
            if self.bad:
                self.bad = not (self.rng.random() < self.condition.gilbert_bad_to_good)
            else:
                self.bad = self.rng.random() < self.condition.gilbert_good_to_bad
            lost = self.rng.random() < self.condition.loss_probability
            if self.bad:
                lost = lost or self.rng.random() < self.condition.gilbert_bad_loss
            if lost:
                self.counts['dropped_bytes'] += len(wire)
                self.counts['dropped_packets'] += 1
                continue
            reordered = self.rng.random() < self.condition.reorder_probability
            extra = self.rng.uniform(0, max(1., self.condition.latency_ms)) if reordered else 0.
            jitter = self.rng.uniform(-self.condition.jitter_ms, self.condition.jitter_ms)
            arrival = self.queue_ms + max(0., self.condition.latency_ms+jitter) + extra
            delivery = WireDelivery(self.serial, kind, body, wire, generated_ms, arrival, bool(copy), reordered, attempt)
            heapq.heappush(self.pending, (arrival, self.serial, delivery))

    def next_delivery(self, through_ms):
        """Deliver one chronological event, allowing an immediate causal response."""
        if not math.isfinite(through_ms) or through_ms < self.clock_ms:
            raise ValueError('channel clock cannot move backwards')
        while self.scheduled and self.scheduled[0][0] <= through_ms and (
                not self.pending or self.scheduled[0][0] <= self.pending[0][0]):
            generated,_,kind,body,attempt = heapq.heappop(self.scheduled)
            self.clock_ms = generated
            self._transmit(kind,body,generated,attempt)
        if not self.pending or self.pending[0][0] > through_ms:
            self.clock_ms = through_ms
            return None
        time, serial, delivery = heapq.heappop(self.pending)
        self.clock_ms = time
        self.delivered_serials.add(serial)
        self.counts['delivered_bytes'] += len(delivery.wire)
        self.counts['delivered_packets'] += 1
        self.counts['reordered_bytes'] += len(delivery.wire) if delivery.reordered else 0
        self.latencies.append(time-delivery.generated_ms)
        return delivery

    def discard(self, delivery):
        if delivery.serial not in self.delivered_serials or delivery.serial in self.discarded_serials:
            raise ValueError('only a delivered frame can be discarded once')
        self.discarded_serials.add(delivery.serial)
        self.counts['discarded_bytes'] += len(delivery.wire)
        self.counts['discarded_packets'] += 1

    def report(self):
        pending = sum(len(row[2].wire) for row in self.pending)
        counts = {key: self.counts[key] for key in (
            'generated_bytes', 'generated_packets', 'delivered_bytes', 'delivered_packets',
            'dropped_bytes', 'dropped_packets', 'duplicate_bytes', 'retry_bytes',
            'reordered_bytes', 'discarded_bytes', 'discarded_packets')}
        latencies = sorted(self.latencies)
        return {**counts, 'pending_bytes': pending, 'scheduled_unsent_packets':len(self.scheduled),
                'component_bytes': dict(self.components),
                'byte_conservation_ok': counts['generated_bytes'] == counts['delivered_bytes']+counts['dropped_bytes']+pending,
                'component_conservation_ok': counts['generated_bytes'] == sum(self.components.values()),
                'p50_latency_ms': latencies[round(.5*(len(latencies)-1))] if latencies else None,
                'p95_latency_ms': latencies[round(.95*(len(latencies)-1))] if latencies else None,
                'overlapping_subtotals': ['duplicate_bytes', 'retry_bytes', 'reordered_bytes', 'discarded_bytes'],
                'security_envelope_is_modeled_not_authenticated': True,
                'physical_mac_radio_bytes_included': False}
