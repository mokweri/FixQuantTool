import os
import sys

import pytest
import torch

# Allow running the tests without an editable install
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
