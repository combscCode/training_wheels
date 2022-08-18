from training_wheels.data.dataset_analysis import get_label_imbalance_array, get_max_label_imbalance, get_scale_array, get_deviation

import numpy as np
import statistics as st
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
    
def test_scale():
    expected = np.asarray([
        [(1.0, 1.0), (6.67, 7.6), (20.0, 9.5)],
        [(0.15, 0.13), (1.0, 1.0), (3.0, 1.25)],
        [(0.05, 0.11), (0.33, 0.8), (1.0, 1.0)]
    ])
    actual = get_scale_array([[100, 4, 4], [50, 12, 2], [80, 15, 20]])
    for i in range(len(expected)):
        for j in range(len(expected[i])):
            for k in range(len(expected[i][j])):
                assert actual[i][j][k] == expected[i][j][k]

def test_deviation():
    expected = [20.55, 4.64, 8.06]
    actual = get_deviation([[100, 4, 4], [50, 12, 2], [80, 15, 20]])
    for i in range(len(expected)):
        assert actual[i] == expected[i]