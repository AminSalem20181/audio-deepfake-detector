from sdeseg.metrics import eer

def test_eer():
    assert eer([0,0,1,1], [.1,.2,.8,.9]) == 0
    assert abs(eer([0,1], [.5,.5])-.5) < 1e-6
