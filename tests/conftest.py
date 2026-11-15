from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from toy import ToyFastDLLMv2, ToyLLaDA, ToySDAR  # noqa: E402


@pytest.fixture(scope="session")
def llada() -> ToyLLaDA:
    # float64 keeps cached and uncached passes within 1e-9 of each other
    return ToyLLaDA().double()


@pytest.fixture(scope="session")
def sdar() -> ToySDAR:
    return ToySDAR().double()


@pytest.fixture(scope="session")
def v2() -> ToyFastDLLMv2:
    return ToyFastDLLMv2().double()


@pytest.fixture()
def prompt() -> torch.Tensor:
    return torch.randint(0, 90, (2, 11), generator=torch.Generator().manual_seed(0))
