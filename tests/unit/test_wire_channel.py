import pytest
from bces.network.events import NetworkCondition
from bces.network.wire_channel import WireChannel, DATAGRAM_HEADER


def drain(channel, end=10000):
    rows=[]
    while (row:=channel.next_delivery(end)) is not None:
        rows.append(row)
    return rows


def test_analytic_serialization_and_causal_response():
    channel=WireChannel(NetworkCondition(50,0,1), seed=17, security_bytes=64)
    channel.send('payload',b'x'*100,0)
    delivery=channel.next_delivery(1000)
    size=100+64+DATAGRAM_HEADER.size
    assert delivery.delivered_ms==pytest.approx(size*8/1000+50)
    with pytest.raises(ValueError): channel.send('payload',b'x',0)
    channel.send('payload',b'y'*100,delivery.delivered_ms)
    response=channel.next_delivery(1000)
    assert response.delivered_ms==pytest.approx(2*(size*8/1000+50))
    assert channel.report()['byte_conservation_ok']


def test_lost_frames_consume_bandwidth_and_bytes():
    channel=WireChannel(NetworkCondition(0,1,1), seed=17, security_bytes=0)
    channel.send('payload',b'x'*100,0)
    channel.send('payload',b'x'*100,0,attempt=1)
    assert channel.queue_ms==pytest.approx(2*110*8/1000)
    assert drain(channel)==[]
    report=channel.report()
    assert report['generated_bytes']==report['dropped_bytes']==220
    assert report['retry_bytes']==110
    assert report['byte_conservation_ok'] and report['component_conservation_ok']


def test_duplicates_are_transmitted_counted_and_discarded_separately():
    channel=WireChannel(NetworkCondition(0,0,1,duplicate_probability=1), seed=17, security_bytes=0)
    channel.send('payload',b'x'*100,0)
    rows=drain(channel)
    assert len(rows)==2 and rows[1].delivered_ms>rows[0].delivered_ms
    channel.discard(rows[1])
    report=channel.report()
    assert report['generated_bytes']==report['delivered_bytes']==220
    assert report['duplicate_bytes']==report['discarded_bytes']==110
    with pytest.raises(ValueError): channel.discard(rows[1])


def test_pending_conservation_and_fixed_seed_reproduction():
    reports=[]
    for _ in range(2):
        channel=WireChannel(NetworkCondition(100,.2,1,duplicate_probability=.5,reorder_probability=.5),seed=17,security_bytes=64)
        for i in range(10): channel.send('payload',b'x'*100,i)
        drain(channel,20)
        report=channel.report()
        assert report['byte_conservation_ok'] and report['component_conservation_ok']
        reports.append(report)
    assert reports[0]==reports[1]


def test_computation_delayed_reply_does_not_reserve_channel_before_earlier_packet():
    channel=WireChannel(NetworkCondition(0,0,1),seed=17,security_bytes=0)
    channel.send('payload',b'later',100)
    channel.send('payload',b'earlier',10)
    assert channel.report()['generated_bytes']==0
    rows=drain(channel,200)
    assert [r.body for r in rows]==[b'earlier',b'later']
    assert rows[0].generated_ms==10 and rows[1].generated_ms==100
    assert channel.report()['byte_conservation_ok']
