"""
构建智能体画像（图）
计算亲和性，输出到data/input/affinity.npy（numpy格式）
"""
import json
import os
from enum import Enum
from itertools import combinations

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from affinity.resource import BasePod, Communication, BaseNode, BasePlatform


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


def _read_csv(path: str, required_columns: list[str]) -> pd.DataFrame:
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"无法读取输入文件 {path}: {exc}") from exc

    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"输入文件 {path} 缺少列: {', '.join(missing_columns)}")
    return data


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if not np.isfinite(matrix).all():
        raise ValueError("亲和性矩阵包含 NaN 或无穷值")
    value_range = matrix.max() - matrix.min()
    if value_range == 0:
        return np.zeros_like(matrix, dtype=float)
    return (matrix - matrix.min()) / value_range


def gen_node_name(pod: BasePod, scene_type: ScenType):
    name = pod.name

    # Common case for all scene types
    if 'center' in name:
        return '任务指挥中心'

    # Scene-specific mappings
    scene_mappings = {
        ScenType.MINE_CLEAR: {
            'node': {
                '1': '艇群反水雷任务规划',
                '2': '无人艇群编队控制'
            },
            'equipt': {
                (1, 10): '无人艇航行控制-{}',
                (11, 20): '灭雷行动-{}',
                (21, 30): '无人艇声学识别-{}',
                'default': '投弹灭雷-{}'
            },
            'aigc': lambda _id: f'AIGC声学模型-{_id}'
        },
        ScenType.ANTI_UNDISTRIBUTED_1: {
            'node': {
                '1': '艇群信息融合',
                '2': '艇群反无人袭扰任务规划',
                '3': '无人艇编队控制'
            },
            'equipt': {
                (1, 10): '单艇信息融合-{}',
                (11, 20): '无人艇航行控制-{}',
                (21, 30): '作战行动-{}',
                (31, 40): '无人艇光学感知-{}',
                (41, 50): '无人艇红外感知-{}',
                'default': '无人机光学识别-{}'
            },
            'aigc': {
                (1, 10): 'AIGC光学生成-{}',
                (11, 20): 'AIGC红外转换-{}',
                'default': 'AIGC无人机光学-{}'
            }
        },
        ScenType.ANTI_UNDISTRIBUTED_2: {
            'node': {
                '1': '反无人袭扰任务规划',
                '2': '飞行控制'
            },
            'equipt': {
                (0, 3): '无人机信息融合-{}',
                (2, 5): '无人机雷达识别-{}',
                (4, 7): '无人机可见光识别-{}',
                'default': '作战行动-{}'
            },
            'aigc': {
                (0, 3): 'AIGC光学生成-{}',
                (2, 5): 'AIGC雷达转换-{}',
                (4, 7): 'AIGC红外转换-{}',
                'default': 'AIGC光学生成-{}'
            }
        }
    }

    # Get the mapping for current scene type
    mapping = scene_mappings.get(scene_type, {})

    # Check for node patterns
    if 'node' in name:
        for prefix, node_name in mapping.get('node', {}).items():
            if prefix in name:
                return node_name

    # Check for equipment patterns
    # Check for equipment patterns
    if 'equipt' in name:
        _id = int(name.replace('equipt-', ''))
        for key, eq_name in mapping.get('equipt', {}).items():
            if key == 'default':
                continue  # Skip default case here, handle after loop
            if isinstance(key, tuple) and key[0] < _id < key[1]:
                return eq_name.format(_id)
        # Handle default case after checking all ranges
        return mapping.get('equipt', {}).get('default', '{}').format(_id)


    # Check for AIGC patterns
    if 'aigc' in name:
        _id = int(name.replace('aigc-', ''))
        aigc_mapping = mapping.get('aigc', {})

        if callable(aigc_mapping):
            return aigc_mapping(_id)

        for key, aigc_name in aigc_mapping.items():
            if key == "default":
                continue
            start, end = key
            if start < _id < end:
                return aigc_name.format(_id)
        return aigc_mapping.get('default', '{}').format(_id)

    # Default case
    return name


def graph_to_tree(G, root_node=None, scene_type=ScenType.ANTI_UNDISTRIBUTED_2):
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
        "id": gen_node_name(root_node, scene_type),
        "children": []
    }

    # 添加子节点
    _add_children(G, tree, root_node, visited={root_node}, scene_type=scene_type)

    return tree


def _add_children(G, parent_node, node_id, visited, scene_type):
    """
    递归添加子节点
    """
    neighbors = list(G.neighbors(node_id))  # 排序以保证一致性

    for neighbor in neighbors:
        if neighbor not in visited:
            visited.add(neighbor)

            # 创建子节点
            child = {
                "id": gen_node_name(neighbor, scene_type=scene_type),
                "children": []
            }

            # 添加节点属性（如果有）
            if G.nodes[neighbor].get("value") is not None:
                child["value"] = G.nodes[neighbor]["value"]

            # 递归添加子节点
            _add_children(G, child, neighbor, visited, scene_type)

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
    weight = [1000, 1]
    attr = [net_affinity_name, race_affinity_name]

    def __init__(self, path: str):
        self.input_path = path
        self.pod_graph = nx.Graph()
        self.command_graph = nx.Graph()
        # read pods
        pods_path = os.path.join(path, "pods.csv")
        data = _read_csv(pods_path, BasePod.get_columns())
        duplicate_pods = data[data["name"].duplicated()]["name"].tolist()
        if duplicate_pods:
            raise ValueError(f"输入文件 {pods_path} 包含重复 Pod: {duplicate_pods}")
        self.pods = []
        self.pod2idx = {}
        for idx, (_, row) in enumerate(data.iterrows()):
            pod = BasePod.from_dataframe(row)
            self.pod_graph.add_node(pod)
            self.pods.append(pod)
            self.pod2idx[pod.name] = idx

        # read communication
        communication_path = os.path.join(path, "communication.csv")
        data = _read_csv(communication_path, Communication.get_columns())
        for row_number, (_, row) in enumerate(data.iterrows(), start=2):
            comm = Communication.from_dataframe(row)
            missing_pods = [
                pod_name
                for pod_name in (comm.src_pod, comm.tgt_pod)
                if pod_name not in self.pod2idx
            ]
            if missing_pods:
                raise ValueError(
                    f"输入文件 {communication_path} 第 {row_number} 行引用了不存在的 Pod: {missing_pods}"
                )
            source = self.pods[self.pod2idx[comm.src_pod]]
            target = self.pods[self.pod2idx[comm.tgt_pod]]
            self.pod_graph.add_edge(
                source,
                target,
                data=comm,
                kind="comm",
                label=comm.to_string(),
            )

        # read nodes
        nodes_path = os.path.join(path, "nodes.csv")
        data = _read_csv(nodes_path, BaseNode.get_columns())
        duplicate_nodes = data[data["name"].duplicated()]["name"].tolist()
        if duplicate_nodes:
            raise ValueError(f"输入文件 {nodes_path} 包含重复 Node: {duplicate_nodes}")
        self.nodes = []
        for _, row in data.iterrows():
            node = BaseNode.from_dataframe(row)
            self.nodes.append(node)

        self.name2platform = {}

    def _load_command_topology(self):
        if self.name2platform:
            return
        command_path = os.path.join(self.input_path, "command.csv")
        command_data = _read_csv(command_path, BasePlatform.get_columns())
        for _, row in command_data.iterrows():
            platform = BasePlatform.from_dataframe(row)
            if platform.name in self.name2platform:
                raise ValueError(f"输入文件 {command_path} 包含重复平台: {platform.name}")
            self.name2platform[platform.name] = platform
            self.command_graph.add_node(platform)
        for platform in self.name2platform.values():
            if platform.parent is None:
                continue
            if platform.parent not in self.name2platform:
                raise ValueError(
                    f"输入文件 {command_path} 中平台 {platform.name} 的父平台不存在: {platform.parent}"
                )
            self.command_graph.add_edge(self.name2platform[platform.parent], platform, label="")

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
        self._load_command_topology()
        if not self.name2platform:
            raise ValueError("未加载指挥关系，无法计算指挥亲和性")
        for source, target in combinations(self.pod_graph.nodes, 2):
            try:
                source_platform = self.name2platform[source.platform]
                target_platform = self.name2platform[target.platform]
                length = nx.shortest_path_length(
                    self.command_graph,
                    source_platform,
                    target_platform,
                )
            except KeyError as exc:
                raise ValueError(f"Pod 引用了不存在的平台: {exc.args[0]}") from exc
            except nx.NetworkXNoPath:
                continue
            self.pod_graph.add_edge(
                source,
                target,
                command_affinity=1 / max(length, 0.1),
            )

    def race_affinity(self):
        """ 资源竞争亲和性 """
        for source, target in combinations(self.pod_graph.nodes, 2):
            value = BasePod.race_affinity(source, target)
            self.pod_graph.add_edge(source, target, race_affinity=-value)

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

    def pod_affinity_to_matrix(self, attr: list[str], weight: list[float], norm=True):
        if len(attr) != len(weight):
            raise ValueError("亲和性属性数量必须与权重数量一致")
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
                matrixs[i] = _normalize_matrix(matrix)
        result = np.zeros((self.pod_graph.number_of_nodes(), self.pod_graph.number_of_nodes()), dtype=float)
        for w, m in zip(weight, matrixs):
            result += w * m
        if norm:
            result = _normalize_matrix(result)
        return result

    @classmethod
    def save_affinity(cls, matrix: np.ndarray, save_path: str, file_name: str):
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f"{file_name}.npy"), matrix)

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
