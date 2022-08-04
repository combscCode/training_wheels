from training_wheels.data.dataset_analysis import get_label_imbalance_array, get_max_label_imbalance

import numpy as np
import pytest


def test_label_imbalance_1():
    expected = np.asarray([
        [0, 0.375, 0.5],
        [-0.375, 0, 0.125],
        [-0.5, -0.125, 0]
    ])
    actual = get_label_imbalance_array([3,2,1,2,1,1,1,1])
    for i in range(len(expected)):
        for j in range(len(expected[i])):
            assert actual[i, j] == expected[i, j]

    expected = np.asarray([
        [0, 3, 4],
        [-3, 0, 1],
        [-4, -1, 0]
    ])
    actual = get_label_imbalance_array([3,2,1,2,1,1,1,1], raw_count=True)
    for i in range(len(expected)):
        for j in range(len(expected[i])):
            assert actual[i, j] == expected[i, j]

    assert get_max_label_imbalance([1,1,2,2,1,1,1,3,4,4]) == .4
    assert get_max_label_imbalance([1,1,2,2,1,1,1,3,4,4], raw_count=True) == 4

    with pytest.raises(RuntimeError):
        get_label_imbalance_array([1,1,1,1,1])
    
    with pytest.raises(RuntimeError):
        get_max_label_imbalance([1,1,1,1])