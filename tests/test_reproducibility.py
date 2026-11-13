import random

import numpy as np
import torch

from tools.reproducibility import set_seed


def test_set_seed_reproduces_python_numpy_and_torch_random_values():
    set_seed(123)
    first_values = (random.random(), np.random.random(), torch.rand(1))

    set_seed(123)
    second_values = (random.random(), np.random.random(), torch.rand(1))

    assert first_values[0] == second_values[0]
    assert first_values[1] == second_values[1]
    assert torch.equal(first_values[2], second_values[2])
