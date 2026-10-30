"""Keep small CPU tensor tests deterministic and inexpensive."""
import torch

def pytest_sessionstart(session):
    torch.set_num_threads(1)
