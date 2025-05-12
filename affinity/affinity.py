"""
构建智能体画像（图）
计算亲和性，输出到data/input/affinity.npy（numpy格式）
"""
import json
import os
from enum import Enum
from operator import contains

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from affinity.resource import BasePod, Communication, BaseNode, BasePlatform
import pytest


class ScenType(Enum):
    UNKNOWN = 0
    # 群综合体嵌入：扫雷
    MINE_CLEAR = 1
    # 反无人袭扰动-侦察机协同
    ANTI_UNDISTRIBUTED_1 = 2
    # 反无人袭扰动-无人机平台
    ANTI_UNDISTRIBUTED_2 = 3
    # 反无人袭扰动-智能体嵌入，信息变化
    ANTI_UNDISTRIBUTED_3 = 4


def gen_node_name(pod: BasePod, scene_type: ScenType):
    if 'center' in pod.name:
        return '任务指挥中心'
    if scene_type == ScenType.MINE_CLEAR:
        if 'node' in pod.name:
            if '1' in pod.name:
                return f'艇群反水雷任务规划'
            if '2' in pod.name:
                return '无人艇群编队控制'
        if 'equipt' in pod.name:
            _id = int(pod.name.replace('equipt-', ''))
            if 1 < _id < 10:
                return f'无人艇航行控制-{_id}'
            if 11 < _id < 20:
                return f'灭雷行动-{_id}'
            if 21 < _id < 30:
                return f'无人艇声学识别-{_id}'
            else:
                return f'投弹灭雷-{_id}'
        if 'aigc' in pod.name:
            _id = int(pod.name.replace('aigc-', ''))
            return f'AIGC声学模型-{_id}'
    elif scene_type==ScenType.ANTI_UNDISTRIBUTED_1:
        if 'node' in pod.name:
            if '1' in pod.name:
                return f'艇群信息融合'
            if '2' in pod.name:
                return '艇群反无人袭扰任务规划'
            if '3' in pod.name:
                return '无人艇编队控制'
        if 'equipt' in pod.name:
            _id = int(pod.name.replace('equipt-', ''))
            if 1 < _id < 10:
                return f'单艇信息融合-{_id}'
            if 11 < _id < 20:
                return f'无人艇航行控制-{_id}'
            if 21 < _id < 30:
                return f'作战行动-{_id}'
            if 31 < _id < 40:
                return f'无人艇光学感知-{_id}'
            if 41 < _id < 50:
                return f'无人艇红外感知-{_id}'
            else:
                return f'无人机光学识别-{_id}'
        if 'aigc' in pod.name:
            _id = int(pod.name.replace('aigc-', ''))
        if 1 < _id < 10:
            return f'AIGC光学生成-{_id}'
        if 11 < _id < 20:
            return f'AIGC红外转换-{_id}'
        else:
            return f'AIGC无人机光学-{_id}'
    elif scene_type==ScenType.ANTI_UNDISTRIBUTED_2:
        if 'node' in pod.name:
            if '1' in pod.name:
                return f'反无人袭扰任务规划'
            if '2' in pod.name:
                return '飞行控制'
        if 'equipt' in pod.name:
            _id = int(pod.name.replace('equipt-', ''))
            if 0 < _id < 3:
                return f'无人机信息融合-{_id}'
            if 2 < _id < 5:
                return f'无人机雷达识别-{_id}'
            if 4 < _id < 7:
                return f'无人机可见光识别-{_id}'
            if 6 < _id < 8:
                return f'作战行动-{_id}'
            else:
                return f'作战行动-{_id}'
        if 'aigc' in pod.name:
            _id = int(pod.name.replace('aigc-', ''))
        if 0 < _id < 3:
            return f'AIGC光学生成-{_id}'
        if 2 < _id < 5:
            return f'AIGC雷达转换-{_id}'
        if 4 < _id < 7:
            return f'AIGC红外转换-{_id}'
        else:
            return f'AIGC光学生成-{_id}'
    else:
        return pod.name


def graph_to_tree(G, root_node=None):
    """
    将networkx的Graph转换为树形结构

    参数:
        G: networkx.Graph
        root_node: 可选，指定根节点的ID。如果未指定，将选择度最高的节点作为根

    返回:
        树形结构的字典
    """
    if not isinstance(G, nx.Graph):
        raise ValueError("输入必须是networkx的Graph")

    if root_node is None:
        # 如果没有指定根节点，选择度指挥节点作为根
        # degrees = dict(G.degree())
        # root_node = max(degrees.keys(), key=lambda x: degrees[x])
        for _node in G.nodes.keys():
            if 'center' in _node.name:
                root_node = _node
                break

    if root_node not in G:
        raise ValueError("指定的根节点不在图中")

    # 创建树结构
    tree = {
        "id": gen_node_name(root_node, ScenType.ANTI_UNDISTRIBUTED_2),
        "children": []
    }

    # 添加子节点
    _add_children(G, tree, root_node, visited=set([root_node]))

    return tree


def _add_children(G, parent_node, node_id, visited):
    """
    递归添加子节点
    """
    neighbors = list(G.neighbors(node_id))  # 排序以保证一致性

    for neighbor in neighbors:
        if neighbor not in visited:
            visited.add(neighbor)

            # 创建子节点
            child = {
                "id": gen_node_name(neighbor, scene_type=ScenType.ANTI_UNDISTRIBUTED_2),
                "children": []
            }

            # 添加节点属性（如果有）
            if G.nodes[neighbor].get("value") is not None:
                child["value"] = G.nodes[neighbor]["value"]

            # 递归添加子节点
            _add_children(G, child, neighbor, visited)

            # 如果子节点有子节点或者有value属性，则保留，否则简化为只有id
            if child["children"] or "value" in child:
                parent_node["children"].append(child)
            else:
                parent_node["children"].append({"id": child["id"]})


class Graph:
    net_affinity_name = 'net_affinity'  # 网络亲和性标签
    data_name = 'data'  # 原始输入数据标签
    command_affinity_name = 'command_affinity'  # 指挥亲和性标签
    race_affinity_name = 'race_affinity'  # 资源竞争亲和性标签
    weight = [1000, 1, 0]
    attr = [net_affinity_name, race_affinity_name]

    def __init__(self, path: str):
        self.pod_graph = nx.Graph()
        self.command_graph = nx.Graph()
        ### read pods
        data = pd.read_csv(f"{path}/pods.csv")
        self.pods = []
        self.pod2idx = {}
        for idx, row in data.iterrows():
            pod = BasePod.from_dataframe(row)
            self.pod_graph.add_node(pod)
            self.pods.append(pod)
            self.pod2idx[pod.name] = idx

        ### read communication
        data = pd.read_csv(os.path.join(path, "communication.csv"))
        for _, row in data.iterrows():
            comm = Communication.from_dataframe(row)
            self.pod_graph.add_edge(self.pods[self.pod2idx[comm.src_pod]], self.pods[self.pod2idx[comm.tgt_pod]],
                                    data=comm, kind="comm")
            self.pod_graph.add_edge(self.pods[self.pod2idx[comm.src_pod]], self.pods[self.pod2idx[comm.tgt_pod]],
                                    label=comm.to_string())

        ### read nodes
        data = pd.read_csv(os.path.join(path, "nodes.csv"))
        self.nodes = []
        for _, row in data.iterrows():
            node = BaseNode.from_dataframe(row)
            self.nodes.append(node)

        # ## read command
        # data = pd.read_csv(os.path.join(path, 'command.csv'))
        # self.name2platform = {}
        # for idx, row in data.iterrows():
        #     p = BasePlatform.from_dataframe(row)
        #     self.name2platform[p.name] = p
        #     self.command_graph.add_node(p)
        #     if p.parent is not None:
        #         self.command_graph.add_edge(self.name2platform[p.parent], p, label="")

    def draw_command(self, save_path):
        G = self.command_graph
        options = {
            "font_size": 36,
            "node_size": 3000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 5,
            "width": 5,
        }
        plt.figure(figsize=(8 * 2, 6 * 2))

        pos = nx.spring_layout(G)
        # pos = nx.spring_layout(G, k=0.2, iterations=18)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12, font_weight='bold')

        edge_labels = nx.get_edge_attributes(G, 'label')  # 获取边的标签
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        # ax.margins(0.20)
        # plt.axis("off")

        plt.savefig(os.path.join(save_path, 'command.png'))
        plt.show()

    def draw_pod(self, save_path):
        G = self.pod_graph

        # 构建数据结构
        # 转换为树形结构
        tree_data = graph_to_tree(G)
        # 打印结果（可以使用json.dumps美化输出）
        print(json.dumps(tree_data, indent=2))

        # 筛选出只保留特定属性的边，比如 color='red'
        selected_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('kind') == 'comm']

        # 创建一个只包含这些边的子图
        G = G.edge_subgraph(selected_edges)

        options = {
            "font_size": 36,
            "node_size": 3000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 5,
            "width": 5,
        }
        plt.figure(figsize=(8 * 5, 6 * 5))

        pos = nx.spring_layout(G)
        # pos = nx.spring_layout(G, k=0.2, iterations=18)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12, font_weight='bold')

        # edge_labels = nx.get_edge_attributes(G, 'label')  # 获取边的标签
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        # ax.margins(0.20)
        # plt.axis("off")
        plt.savefig(f'{save_path}/pod.png')
        plt.show()

    def net_affinity(self):
        """ 计算网络的亲和性 """
        for u, v in self.pod_graph.edges:
            d = self.pod_graph.get_edge_data(u, v)[Graph.data_name]
            self.pod_graph.add_edge(u, v, net_affinity=d.freq * d.package)

    def command_affinity(self):
        """ 指挥交互关系亲和性 """
        for x in self.pod_graph.nodes:
            for y in self.pod_graph.nodes:
                if x == y:
                    break
                x_platform = self.name2platform[x.platform]
                y_platform = self.name2platform[y.platform]
                length = nx.shortest_path_length(self.command_graph, x_platform, y_platform)
                if length == 0:
                    length = 0.1
                self.pod_graph.add_edge(x, y, command_affinity=1 / length)

    def race_affinity(self):
        """ 资源竞争亲和性 """
        for source in self.pod_graph.nodes:
            for target in self.pod_graph.nodes:
                if source == target:
                    break
                v = BasePod.race_affinity(source, target)
                self.pod_graph.add_edge(source, target, race_affinity=-v)

    # 计算节点亲和性（资源竞争，是否 > pod 需要，如果 > 就是1）
    def node_affinity(self):
        matrix = np.zeros((self.pod_graph.number_of_nodes(), len(self.nodes)), dtype=int)
        for pod in self.pod_graph.nodes:
            x = self.pod2idx[pod.name]
            for y, node in enumerate(self.nodes):
                if node >= pod:
                    matrix[x, y] = 1
                else:
                    matrix[x, y] = 0
        return matrix

    def pod_affinity_to_matrix(self, attr: [str], weight: [float], norm=True):
        matrixs = [np.zeros((self.pod_graph.number_of_nodes(), self.pod_graph.number_of_nodes()), dtype=float) for i in
                   range(len(attr))]
        for u, v, d in self.pod_graph.edges(data=True):
            i = self.pod2idx[u.name]
            j = self.pod2idx[v.name]
            for t, a in enumerate(attr):
                if d.__contains__(a):
                    matrixs[t][i][j] = d[a]
                    matrixs[t][j][i] = d[a]
        if norm:
            for i, matrix in enumerate(matrixs):
                matrixs[i] = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        result = np.zeros((self.pod_graph.number_of_nodes(), self.pod_graph.number_of_nodes()), dtype=float)
        for w, m in zip(weight, matrixs):
            result += w * m
        if norm:
            result = (result - result.min()) / (result.max() - result.min())
        return result

    @classmethod
    def save_affinity(cls, matrix: np.ndarray, save_path: str, file_name: str):
        np.save(f"{save_path}/{file_name}.npy", matrix)

    @classmethod
    def draw_hist(cls, matrix):
        max_value = np.max(matrix)
        min_value = np.min(matrix)
        median_value = np.median(matrix)
        print("最大值：", max_value)
        print("最小值：", min_value)
        print("中位数：", median_value)

        # matrix = np.log(matrix + 1e-50)
        # 计算直方图
        hist, bin_edges = np.histogram(matrix, bins=20)  # 分为 20 个区间
        print("Histogram:", hist)
        # 使用 Matplotlib 绘制直方图
        plt.hist(matrix.ravel(), bins=20, color='blue', alpha=0.7)
        plt.title("Data Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()


def cal_affinity_and_save(input_dir, saved_path):
    g = Graph(input_dir)
    # ### 计算保存pod间亲和性
    g.net_affinity()
    # g.command_affinity()
    g.race_affinity()
    pod_affinity = g.pod_affinity_to_matrix(Graph.attr, Graph.weight)
    Graph.draw_hist(pod_affinity)
    Graph.save_affinity(pod_affinity, saved_path, "pod_affinity")
    # ### 计算保存硬亲和性
    node_affinity = g.node_affinity()
    Graph.save_affinity(node_affinity, saved_path, "node_affinity")


def cal_affinity(input_dir):
    g = Graph(input_dir)
    # ### 计算保存pod间亲和性
    g.net_affinity()
    # g.command_affinity()
    g.race_affinity()
    pod_affinity = g.pod_affinity_to_matrix(Graph.attr, Graph.weight)
    node_affinity = g.node_affinity()
    return pod_affinity, node_affinity
