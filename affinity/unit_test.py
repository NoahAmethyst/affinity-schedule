import pytest
import numpy as np

from affinity.affinity import cal_affinity_and_save, cal_affinity


def test_cal_affinity_and_save():
    input_dir = '../data/input'
    saved_path = '../data/output'
    cal_affinity_and_save(input_dir, saved_path)

def test_cal_affinity():
    input_dir = '../data/input'
    pod_affinity, node_affinity = cal_affinity(input_dir)
    print(pod_affinity)
    print(node_affinity)

def test_load_affinity():
    node_affinity = np.load('../data/output/node_affinity.npy')
    print(node_affinity.shape)
