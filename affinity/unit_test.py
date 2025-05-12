import pytest
import numpy as np

from affinity.affinity import cal_affinity_and_save, cal_affinity, Graph
from affinity.gen_data import gen_pods, gen_nodes, save_resource, save_communication


def test_gen_data():
    ### 生成测试数据
    save_path = ('/Users/amethyst/PycharmProjects/affinity-schedule/data/input')
    pods, comm, platform = gen_pods(70)
    nodes = gen_nodes(5, 0)
    save_resource(pods, nodes, platform, save_path)
    save_communication(comm, save_path)


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


def test_draw_graph():
    input_dir = "/Users/amethyst/PycharmProjects/affinity-schedule/data/input"
    output_dir = "/Users/amethyst/PycharmProjects/affinity-schedule/data/others"
    g = Graph(input_dir)

    # ### 计算保存pod间亲和性
    # g.net_affinity()
    # g.command_affinity()
    # g.race_affinity()
    pod_affinity = g.pod_affinity_to_matrix(Graph.attr, Graph.weight)

    g.draw_command(output_dir)
    g.draw_pod(output_dir)
