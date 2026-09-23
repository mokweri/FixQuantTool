import os
import sys

import pytest
import torch

# Allow running the tests without an editable install
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))
# Tests import the CLI entry points under `tools/`. Put the repository root
# first so those resolve to this checkout and not to some other `tools`
# package that happens to be installed in the environment.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
