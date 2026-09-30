"""Causal application session for corrected closed-loop engineering pilots.

The calibration experiment does not import this module. A caller supplies only
past/current receiver history and sender observations through two callbacks.
Every refresh, payload, query, reply and retransmission uses WireChannel bytes.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import Counter

from bces.network.wire_channel import refresh_control
from bces.protocol.bindings import sender_id_hash
from bces.protocol.bound_exchange import BoundQuery, BoundReceiver, decode_objects, decode_response


class ReferenceUnavailable(RuntimeError):
    """Known causal inability to form a reference, never an oracle-label failure."""


class BoundSession:
    def __init__(self, *, channel, policy, build_query, payload_provider, sender_id,
                 computation_delay_ms=6., timeout_ms=400., max_retries=1):
        if (not math.isfinite(computation_delay_ms) or computation_delay_ms < 0
                or not math.isfinite(timeout_ms) or timeout_ms <= 0
                or type(max_retries) is not int or max_retries < 0):
            raise ValueError('invalid frozen session timing')
        self.channel, self.policy = channel, policy
        self.build_query, self.payload_provider = build_query, payload_provider
        self.sender_id = sender_id
        self.computation_delay_ms, self.timeout_ms, self.max_retries = computation_delay_ms, timeout_ms, max_retries
        self.receiver = BoundReceiver()
        self.server_payloads = {}
        self.receiver_payload = None
        self.pending = None
        self.query_id = 0
        self.last_payload_ms = -1
        self.counts = Counter()
        self.timeline = []
        self.issue_wall_ms = []

    def send_payload(self, payload, generated_ms):
        message = decode_objects(payload)
        if message.sender_id != self.sender_id or message.timestamp_ms > generated_ms:
            raise ValueError('sender payload provenance/time mismatch')
        self.server_payloads[hashlib.sha256(payload).hexdigest()] = payload
        self.channel.send('payload',payload,generated_ms)

    def _request(self, kind, body, generated_ms, attempt=0):
        self.channel.send(kind,body,generated_ms,attempt=attempt)
        self.pending = {'kind':kind,'body':body,'sent_ms':generated_ms,'attempt':attempt}
        self.timeline.append({'event':kind+'_sent','time_ms':generated_ms,'attempt':attempt})

    def request_refresh(self, generated_ms):
        if self.pending is not None: return
        self._request('refresh',refresh_control(sender_id_hash(self.sender_id),self.query_id),generated_ms)

    def _discard(self, event, reason):
        self.channel.discard(event)
        self.counts[reason] += 1

    def _receive(self, event):
        arrival = event.delivered_ms
        self.timeline.append({'event':event.kind+'_received','time_ms':arrival,
                              'generated_ms':event.generated_ms,'attempt':event.attempt})
        if event.kind == 'refresh':
            # Provider reads its most recent observed world, not a future SUMO step.
            self.send_payload(self.payload_provider(arrival),arrival)
        elif event.kind == 'payload':
            message = decode_objects(event.body)
            if message.sender_id != self.sender_id or message.timestamp_ms > arrival:
                self._discard(event,'invalid_payload'); return
            if message.timestamp_ms <= self.last_payload_ms:
                self._discard(event,'duplicate_or_stale_payload'); return
            generated = math.ceil(arrival)
            next_id = self.query_id+1
            try:
                bound = self.build_query(event.body,next_id,generated)
            except ReferenceUnavailable:
                self.last_payload_ms = message.timestamp_ms
                self.pending = None
                self._discard(event,'reference_unavailable'); return
            bound.validate_payload(event.body)
            if (bound.query.query_id != next_id or bound.generated_ms != generated
                    or bound.payload_received_ms != generated or bound.family != self.policy.family
                    or bound.model_sha256 != self.policy.sha256):
                raise ValueError('receiver query callback violated frozen session binding')
            wire = bound.encode()
            self.receiver.register(wire)
            self.receiver_payload = event.body
            self.query_id, self.last_payload_ms = next_id, message.timestamp_ms
            self._request('query',wire,generated)
        elif event.kind == 'query':
            bound = BoundQuery.decode(event.body)
            payload = self.server_payloads.get(bound.payload_sha256)
            if payload is None:
                self._discard(event,'sender_missing_payload'); return
            start = time.perf_counter_ns()
            reply = self.policy.issue(event.body,payload)
            self.issue_wall_ms.append((time.perf_counter_ns()-start)/1e6)
            issued = arrival+self.computation_delay_ms
            self.channel.send('response',reply,issued)
            self.timeline.append({'event':'response_scheduled','query_received_ms':arrival,'time_ms':issued})
        elif event.kind == 'response':
            if self.receiver.query_wire is None:
                self._discard(event,'unsolicited_response'); return
            try:
                extension = decode_response(event.body,self.receiver.query_wire)
            except ValueError:
                self._discard(event,'stale_or_mismatched_response'); return
            if self.receiver.extension is not None:
                self._discard(event,'duplicate_response' if extension == self.receiver.extension else 'conflicting_response')
                return
            self.receiver.install(event.body,self.receiver_payload)
            if self.pending is not None and self.pending['kind']=='query' and self.pending['body']==self.receiver.query_wire:
                self.pending = None
            self.counts['installed_responses'] += 1

    def advance(self, through_ms):
        if through_ms < self.channel.clock_ms:
            raise ValueError('session cannot observe the past')
        while True:
            deadline = self.pending['sent_ms']+self.timeout_ms if self.pending is not None else math.inf
            event = self.channel.next_delivery(min(through_ms,deadline))
            if event is not None:
                self._receive(event)
                continue
            if deadline > through_ms: break
            request = self.pending
            if request['attempt'] < self.max_retries:
                self._request(request['kind'],request['body'],deadline,request['attempt']+1)
                self.counts['request_retries'] += 1
            else:
                self.pending = None
                self.counts['request_timeouts'] += 1

    def decision(self, current_state, behavior):
        now = current_state.timestamp_ms
        self.advance(now)
        decision = self.receiver.evaluate(current_state,behavior,self.policy.policy_hash,self.sender_id)
        if decision.request_refresh: self.request_refresh(now)
        self.counts['decision:'+decision.state.value] += 1
        self.counts['reason:'+decision.reason] += 1
        return decision

    def report(self):
        return {**self.channel.report(),'session_counts':dict(self.counts),
            'pending_request_kind':self.pending['kind'] if self.pending else None,
            'fixed_sender_computation_delay_ms':self.computation_delay_ms,
            'measured_sender_issue_wall_ms':self.issue_wall_ms,
            'timeline':self.timeline,
            'scope':'application engineering; no closed-loop efficacy inference from session tests'}
