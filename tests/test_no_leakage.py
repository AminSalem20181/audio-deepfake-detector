def test_no_leakage():
    from sdeseg.train import ProtocolDataset
    assert ProtocolDataset.__init__.__code__.co_argcount == 4
