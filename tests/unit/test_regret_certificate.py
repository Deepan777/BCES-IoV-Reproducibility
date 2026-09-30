import itertools
import numpy as np
import pytest

from bces.geometry.regret_certificate import (conditional_regret_bound,linear_errors,
    conservative_box_certificate,verify_box_certificate)
from bces.geometry.surfaces import ExpirySurface


def test_interval_bound_covers_every_endpoint_combination_and_nonoptimal_selection():
    estimates=np.array([2.,1.,3.]);errors=np.array([.2,.1,.4]);selected=0
    bound=conditional_regret_bound(estimates,errors,selected)
    assert bound==pytest.approx(1.3)  # nominal non-optimality is retained
    for signs in itertools.product((-1,1),repeat=3):
        truth=estimates+errors*np.asarray(signs)
        assert truth[selected]-truth.min()<=bound+1e-12


def test_linear_errors_and_conservative_surface_imply_tolerance():
    estimates=[1.,1.4,1.8];base=[.03,.04,.02]
    sensitivity=np.array([[.1,.2,.0,.1,.0,.05,.2],
                          [.2,.1,.1,.0,.1,.05,.1],
                          [.1,.1,.2,.1,.0,.10,.1]])
    certificate=conservative_box_certificate(estimates,base,sensitivity,0,.25,[2.]*7)
    assert certificate.origin_permitted
    assert verify_box_certificate(certificate,estimates,base,sensitivity,0)
    surface=ExpirySurface(certificate.offsets)
    rng=np.random.default_rng(9)
    accepted=0
    for _ in range(5000):
        z=rng.uniform(-1,1,7)*np.asarray(certificate.radii)
        if surface.contains(z):
            accepted+=1
            assert conditional_regret_bound(estimates,linear_errors(base,sensitivity,z),0)<=.25+1e-12
    assert accepted>100


def test_failed_origin_yields_zero_abstention_region():
    certificate=conservative_box_certificate([2.,1.],[.1,.1],np.ones((2,7)),0,.5,[2.]*7)
    assert not certificate.origin_permitted
    assert certificate.radii==(0.,)*7 and certificate.offsets==(0.,)*16
    assert verify_box_certificate(certificate,[2.,1.],[.1,.1],np.ones((2,7)),0)


@pytest.mark.parametrize('call',[lambda:conditional_regret_bound([1],[0],1),
    lambda:linear_errors([0],[[1,-1]],[0,0]),
    lambda:conservative_box_certificate([1,2],[0,0],np.ones((2,6)),0,1,[1]*6)])
def test_certificate_rejects_invalid_contracts(call):
    with pytest.raises(ValueError): call()
