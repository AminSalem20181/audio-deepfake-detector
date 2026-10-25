from sdeseg.models import Sdeseg, standard_conv_twin, count_parameters

def test_param_count():
    assert count_parameters(Sdeseg()) == 30434
    assert count_parameters(standard_conv_twin()) == 245362
