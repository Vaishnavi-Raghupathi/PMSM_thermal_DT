import numpy as np
from src.diagnostics import rmse


def test_rmse():
    preds = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.1, 1.9, 3.0])
    val = rmse(preds, targets)
    assert abs(val - 0.08165) < 1e-3
