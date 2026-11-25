"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

from .config import DEFAULT_SEED


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch when they are available.

    NumPy and PyTorch are imported lazily so this module can be imported
    before the environment is fully installed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
