import copy
import json
import queue
import random
from dataclasses import dataclass
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from affinity.resource import BasePod, BaseNode, BasePlatform, Communication


class PodType(Enum):
    COMMAND_CENTER = 1
    COMMAND_NODE = 2
    EQUIPT = 3
    AIGC = 4


class CommType(Enum):
    HIGH = 1
    MIDDLE = 2
    LOW = 3


@dataclass
class CommFrequency:
    frequency_high: int
    frequency_low: int
    data_size_high: int
    data_size_low: int
    send_time: int


@dataclass
class PodAttr:
    pod_type: PodType
    comm_type: CommType
    num: int
    id: int = 0

    def __hash__(self):
        return hash((self.pod_type, self.comm_type, self.num, self.id))


def gen_base_pods() -> dict[PodType, list[BasePod]]:
    return {PodType.COMMAND_CENTER: [
        BasePod("", 3, 10, 0, 2 * 1024, None),
        BasePod("", 3, 10, 0, 3 * 1024, None),
        BasePod("", 3, 10, 0, 3 * 1024, None),
        BasePod("", 3, 10, 0, 4 * 1024, None), ],
        PodType.COMMAND_NODE: [
            BasePod("", 1, 4, 0 * 1024, 1 * 1024, None),
            BasePod("", 2, 3, 0 * 1024, 1 * 1024, None),
            BasePod("", 1, 2, 0 * 1024, 1 * 1024, None),
            BasePod("", 2, 1, 0 * 1024, 1 * 1024, None),
        ],
        PodType.EQUIPT: [
            BasePod("", 2, 4, 0 * 1024, 2 * 1024, None),
            BasePod("", 3, 5, 0 * 1024, 2 * 1024, None),
            BasePod("", 3, 6, 0 * 1024, 2 * 1024, None),
            BasePod("", 4, 7, 0 * 1024, 2 * 1024, None)],

        PodType.AIGC: [
            BasePod("", 2, 4, 0 * 1024, 2 * 1024, None),
            BasePod("", 3, 5, 0 * 1024, 2 * 1024, None),
            BasePod("", 3, 6, 0 * 1024, 2 * 1024, None),
            BasePod("", 4, 7, 0 * 1024, 2 * 1024, None)]
    }


def gen_base_nodes() -> list[BaseNode]:
    return [
        BaseNode("", 32, 240, 0, 1.5 * 1024 * 1024, 10000),  #
        BaseNode("", 32, 240, 1024 * 24, 1.5 * 1024 * 1024, 10000),  # gpu
    ]


def gen_base_communication() -> dict[CommType, CommFrequency]:
    # 通信频次、通信量、通信次数
    times = 1000 * 6
    return {
        CommType.HIGH: CommFrequency(frequency_high=10, frequency_low=6, data_size_high=5, data_size_low=1,
                                     send_time=times),
        CommType.MIDDLE: CommFrequency(frequency_high=6, frequency_low=5, data_size_high=2, data_size_low=1,
                                       send_time=times),
        CommType.LOW: CommFrequency(frequency_high=2, frequency_low=1, data_size_high=9, data_size_low=6,
                                    send_time=times),
    }


def gen_pods(G: nx.DiGraph = None) -> tuple[list[BasePod], list[Communication], list[BasePlatform]]:
    """
    生成Pod、通信关系和平台数据
    :param num: 生成的Pod数量
    :param G: 通信拓扑图
    :return: (pods列表, 通信关系列表, 平台列表)
    """
    base_pods = gen_base_pods()
    comm_frequencies = gen_base_communication()

    pods = []
    communications = []
    platforms = []

    type_counter = {
        PodType.COMMAND_CENTER: 0,
        PodType.COMMAND_NODE: 0,
        PodType.EQUIPT: 0,
        PodType.AIGC: 0
    }

    pod_dict = {pod_attr: [] for pod_attr in G.nodes.keys()}

    # 为图中的每个节点生成Pod实例
    for node_attr in G.nodes():
        pod_type = node_attr.pod_type
        num_pods = node_attr.num

        for i in range(num_pods):
            # 从基础Pod中选择一个模板
            base_pod = base_pods[pod_type][random.randint(0, len(base_pods[pod_type]) - 1)]
            pod = copy.copy(base_pod)
            pod.name = f"{pod_type.name.lower()}-{type_counter[pod_type] + 1}"
            type_counter[pod_type] += 1
            pods.append(pod)

            pod_dict[node_attr].append(pod)

            # 如果是命令中心或命令节点，添加到平台列表
            if pod_type in [PodType.COMMAND_CENTER, PodType.COMMAND_NODE]:
                platform = BasePlatform(pod.name, pod_type.value)
                platforms.append(platform)

    for src_attr, dst_attr in G.edges():
        src_pods = pod_dict[src_attr][:src_attr.num]
        dst_pods = pod_dict[dst_attr][:dst_attr.num]

        comm_freq = comm_frequencies[src_attr.comm_type]
        # 创建通信关系
        if src_attr.num == dst_attr.num:
            for i in range(src_attr.num):
                communication = Communication(
                    src=src_pods[i].name, tgt=dst_pods[i].name,
                    freq=random.randint(comm_freq.frequency_low, comm_freq.frequency_high),
                    pak=random.randint(comm_freq.data_size_low, comm_freq.data_size_high),
                    cnt=comm_freq.send_time)
                communications.append(communication)

        else:

            if dst_attr.pod_type == PodType.COMMAND_NODE:
                for src in src_pods:
                    communication = Communication(
                        src=src.name, tgt=dst_pods[0].name,
                        freq=random.randint(comm_freq.frequency_low, comm_freq.frequency_high),
                        pak=random.randint(comm_freq.data_size_low, comm_freq.data_size_high),
                        cnt=comm_freq.send_time)
                    communications.append(communication)
            else:
                _index = 0
                for dst in dst_pods:
                    communication = Communication(
                        src=src_pods[_index].name, tgt=dst.name,
                        freq=random.randint(comm_freq.frequency_low, comm_freq.frequency_high),
                        pak=random.randint(comm_freq.data_size_low, comm_freq.data_size_high),
                        cnt=comm_freq.send_time)
                    communications.append(communication)
                    _index += 1
                    if _index == len(src_pods):
                        _index = 0

    return pods, communications, platforms


def gen_nodes(num: int, gpu_num: int) -> [BaseNode]:
    node_types = gen_base_nodes()
    nodes = []
    for i in range(num):  # 生成普通node
        node = copy.copy(node_types[0])
        node.name = f"node-{len(nodes) + 1}"
        nodes.append(node)
    for i in range(gpu_num):
        node = copy.copy(node_types[1])
        node.name = f"node-{len(nodes) + 1}"
        nodes.append(node)
    return nodes


def save_communication(connections: list[Communication], save_path: str):
    """
    保存通信关系文件
    :return
    save_path/communication.yaml
    """
    data = [con.get_data() for con in connections]
    df = pd.DataFrame(data, columns=Communication.get_columns())
    df.to_csv(f"{save_path}/communication.csv", index=False)


def save_resource(pods: list[BasePod], nodes: list[BaseNode], platforms: list[BasePlatform], save_path: str):
    """
    保存资源需求文件
    :return:
    save_path/pods.csv
    save_path/nodes.csv
    """
    data = [pod.get_data() for pod in pods]
    df = pd.DataFrame(data, columns=BasePod.get_columns())
    df.to_csv(f"{save_path}/pods.csv", index=False)

    data = [node.get_data() for node in nodes]
    df = pd.DataFrame(data, columns=BaseNode.get_columns())
    df.to_csv(f'{save_path}/nodes.csv', index=False)

    data = [p.get_data() for p in platforms]
    df = pd.DataFrame(data, columns=BasePlatform.get_columns())
    df.to_csv(f'{save_path}/command.csv', index=False)


# 使用networkx构建和可视化拓扑图
def draw_graph(g: nx.DiGraph):
    # 可视化
    plt.figure(figsize=(12, 8))

    # 节点布局
    pos = nx.spring_layout(g, seed=42)

    # 获取节点属性
    node_attrs = nx.get_node_attributes(g, 'platform')

    # 绘制节点 - 使用平台信息生成颜色
    node_colors = [hash(platform) % 256 if platform else 0
                   for _, platform in node_attrs.items()]

    nx.draw_networkx_nodes(g, pos, node_size=500,
                           node_color=node_colors,
                           cmap=plt.cm.tab20,
                           alpha=0.8)

    # 获取边属性并确保有有效宽度
    edge_attrs = nx.get_edge_attributes(g, 'frequency')

    # 确保边宽度列表不为空且有有效值
    if edge_attrs:
        edge_widths = [1 + (freq / 10 if freq else 1) for freq in edge_attrs.values()]
    else:
        edge_widths = 1.0  # 默认宽度

    # 绘制边
    nx.draw_networkx_edges(g, pos, width=edge_widths,
                           arrows=True, arrowstyle='->',
                           edge_color='gray', alpha=0.6)

    # 绘制节点标签
    nx.draw_networkx_labels(g, pos, font_size=8)

    # 添加边属性标签
    edge_labels = {}
    for u, v in g.edges():
        freq = g.edges[u, v].get('frequency', 'N/A')
        pkg = g.edges[u, v].get('package_size', 'N/A')
        edge_labels[(u, v)] = f"f:{freq}\np:{pkg}"

    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                 font_size=6, label_pos=0.3)

    plt.title("POD Communication Topology")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def test_gen_data():
    # 构造通信拓扑

    # 扫雷通信
    # G = mine_clear()
    # 反无人袭扰（无人机协同）
    # G = anti_undistributed_1()
    # 反无人袭扰（平台嵌入）
    # G = anti_undistributed_2()
    # 反无人袭扰（平台嵌入 变化）
    G = anti_undistributed_3()
    # 查看节点通信拓扑关系
    draw_graph(G)
    ### 生成测试数据

    save_path = ('/Users/amethyst/PycharmProjects/affinity-schedule/data/input')
    pods, comm, platform = gen_pods(G)
    nodes = gen_nodes(8, 3)
    save_resource(pods, nodes, platform, save_path)
    save_communication(comm, save_path)


# 群综合扫雷
def mine_clear():
    # 构造基座节点
    # 指挥节点
    command_node = PodAttr(pod_type=PodType.COMMAND_CENTER, comm_type=CommType.LOW, num=1)
    # 规划控制
    plan_node1 = PodAttr(pod_type=PodType.COMMAND_NODE, comm_type=CommType.MIDDLE, num=1, id=1)
    plan_node2 = PodAttr(pod_type=PodType.COMMAND_NODE, comm_type=CommType.HIGH, num=1, id=2)
    # 设备节点
    equipt_node1 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=20, id=1)
    equipt_node2 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=10, id=2)
    equipt_node3 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=10, id=3)
    # AIGC模型
    aigc_node = PodAttr(pod_type=PodType.AIGC, comm_type=CommType.LOW, num=10)
    # 创建有向图
    G = nx.DiGraph()
    # 添加节点
    G.add_node(command_node)
    G.add_node(plan_node1)
    G.add_node(plan_node2)
    G.add_node(equipt_node1)
    G.add_node(equipt_node2)
    G.add_node(equipt_node3)
    G.add_node(aigc_node)
    # 添加边（通信关系）
    G.add_edge(command_node, plan_node1)
    G.add_edge(plan_node1, equipt_node1)
    G.add_edge(plan_node1, equipt_node2)
    G.add_edge(plan_node1, plan_node2)
    G.add_edge(equipt_node3, plan_node2)
    G.add_edge(aigc_node, equipt_node2)
    return G


# 反无人袭扰：侦察机协同配合
def anti_undistributed_1():
    # 构造基座节点
    # 指挥节点
    command_node = PodAttr(pod_type=PodType.COMMAND_CENTER, comm_type=CommType.LOW, num=1)
    # 规划控制
    # 艇群信息融合
    plan_node1 = PodAttr(pod_type=PodType.COMMAND_NODE, comm_type=CommType.MIDDLE, num=1, id=1)
    # 艇群反无人袭扰任务规划
    plan_node2 = PodAttr(pod_type=PodType.COMMAND_NODE, comm_type=CommType.MIDDLE, num=1, id=2)
    # 无人艇编队控制
    plan_node3 = PodAttr(pod_type=PodType.COMMAND_NODE, comm_type=CommType.MIDDLE, num=1, id=3)
    # 设备节点
    # 单艇信息融合
    equipt_node1 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=10, id=1)
    # 无人艇航行控制
    equipt_node2 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.MIDDLE, num=10, id=2)
    # 作战行动
    equipt_node3 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.MIDDLE, num=10, id=3)
    # 无人艇光学感知
    equipt_node4 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=10, id=4)
    # 无人艇红外感知
    equipt_node5 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=10, id=5)
    # 无人机光学识别
    equipt_node6 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=2)
    # AIGC光学
    aigc_node1 = PodAttr(pod_type=PodType.AIGC, comm_type=CommType.LOW, num=10, id=1)
    # AIGC红外
    aigc_node2 = PodAttr(pod_type=PodType.AIGC, comm_type=CommType.LOW, num=10, id=2)
    # AIGC无人机
    aigc_node3 = PodAttr(pod_type=PodType.AIGC, comm_type=CommType.LOW, num=2, id=3)
    # 创建有向图
    G = nx.DiGraph()
    # 添加节点
    G.add_node(command_node)
    G.add_node(plan_node1)
    G.add_node(plan_node2)
    G.add_node(plan_node3)
    G.add_node(equipt_node1)
    G.add_node(equipt_node2)
    G.add_node(equipt_node3)
    G.add_node(equipt_node4)
    G.add_node(equipt_node5)
    G.add_node(equipt_node6)
    G.add_node(aigc_node1)
    G.add_node(aigc_node2)
    G.add_node(aigc_node3)
    # 添加边（通信关系）
    G.add_edge(command_node, plan_node2)
    G.add_edge(command_node, equipt_node6)
    G.add_edge(plan_node2, plan_node3)
    G.add_edge(plan_node3, equipt_node2)
    G.add_edge(plan_node2, equipt_node3)
    G.add_edge(equipt_node4, equipt_node1)
    G.add_edge(equipt_node5, equipt_node1)
    G.add_edge(equipt_node1, plan_node1)
    G.add_edge(plan_node1, plan_node2)

    G.add_edge(aigc_node1, equipt_node4)
    G.add_edge(aigc_node2, equipt_node5)
    G.add_edge(aigc_node3, equipt_node6)

    return G


# 反无人袭扰：平台嵌入
def anti_undistributed_2():
    # 构造基座节点
    # 指挥节点
    command_node = PodAttr(pod_type=PodType.COMMAND_CENTER, comm_type=CommType.LOW, num=1)
    # 规划控制
    # 反无人袭扰任务规划
    plan_node1 = PodAttr(pod_type=PodType.COMMAND_NODE, comm_type=CommType.MIDDLE, num=1, id=1)
    # 飞行控制
    plan_node2 = PodAttr(pod_type=PodType.COMMAND_NODE, comm_type=CommType.MIDDLE, num=1, id=2)

    # 设备节点
    # 无人机信息融合
    equipt_node1 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=2, id=1)
    # 无人机雷达识别
    equipt_node2 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.MIDDLE, num=2, id=2)
    # 无人机可见光识别
    equipt_node3 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.MIDDLE, num=2, id=3)
    # 无人机红外识别
    equipt_node4 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=2, id=4)
    # 作战行动
    equipt_node5 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=2, id=5)
    # AIGC光学
    aigc_node1 = PodAttr(pod_type=PodType.AIGC, comm_type=CommType.LOW, num=2, id=1)
    # AIGC雷达
    aigc_node2 = PodAttr(pod_type=PodType.AIGC, comm_type=CommType.LOW, num=2, id=2)
    # AIGC红外
    aigc_node3 = PodAttr(pod_type=PodType.AIGC, comm_type=CommType.LOW, num=2, id=3)
    # 创建有向图
    G = nx.DiGraph()
    # 添加节点
    G.add_node(command_node)
    G.add_node(plan_node1)
    G.add_node(plan_node2)

    G.add_node(equipt_node1)
    G.add_node(equipt_node2)
    G.add_node(equipt_node3)
    G.add_node(equipt_node4)
    G.add_node(equipt_node5)

    G.add_node(aigc_node1)
    G.add_node(aigc_node2)
    G.add_node(aigc_node3)
    # 添加边（通信关系）
    G.add_edge(command_node, plan_node1)
    G.add_edge(plan_node1, plan_node2)
    G.add_edge(plan_node2, equipt_node3)
    G.add_edge(equipt_node1, plan_node1)
    G.add_edge(equipt_node2, equipt_node1)
    G.add_edge(equipt_node3, equipt_node1)
    G.add_edge(equipt_node4, equipt_node1)
    G.add_edge(equipt_node4, equipt_node1)
    G.add_edge(plan_node1, equipt_node5)

    G.add_edge(aigc_node1, equipt_node2)
    G.add_edge(aigc_node2, equipt_node3)
    G.add_edge(aigc_node3, equipt_node4)

    return G


# 反无人袭扰：平台嵌入（变化）
def anti_undistributed_3():
    # 构造基座节点
    # 指挥节点
    command_node = PodAttr(pod_type=PodType.COMMAND_CENTER, comm_type=CommType.LOW, num=1)
    # 规划控制
    # 反无人袭扰任务规划
    plan_node1 = PodAttr(pod_type=PodType.COMMAND_NODE, comm_type=CommType.MIDDLE, num=1, id=1)
    # 飞行控制
    plan_node2 = PodAttr(pod_type=PodType.COMMAND_NODE, comm_type=CommType.MIDDLE, num=1, id=2)

    # 设备节点
    # 无人机信息融合
    equipt_node1 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=2, id=1)
    # 无人机雷达识别
    equipt_node2 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.MIDDLE, num=2, id=2)
    # 无人机可见光识别
    equipt_node3 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.MIDDLE, num=2, id=3)
    # 无人机红外识别
    equipt_node4 = PodAttr(pod_type=PodType.EQUIPT, comm_type=CommType.HIGH, num=2, id=4)
    # AIGC光学
    aigc_node1 = PodAttr(pod_type=PodType.AIGC, comm_type=CommType.LOW, num=2, id=1)
    # AIGC雷达
    aigc_node2 = PodAttr(pod_type=PodType.AIGC, comm_type=CommType.LOW, num=2, id=2)
    # AIGC红外
    aigc_node3 = PodAttr(pod_type=PodType.AIGC, comm_type=CommType.LOW, num=2, id=3)
    # 创建有向图
    G = nx.DiGraph()
    # 添加节点
    G.add_node(command_node)
    G.add_node(plan_node1)
    G.add_node(plan_node2)

    G.add_node(equipt_node1)
    G.add_node(equipt_node2)
    G.add_node(equipt_node3)
    G.add_node(equipt_node4)

    G.add_node(aigc_node1)
    G.add_node(aigc_node2)
    G.add_node(aigc_node3)
    # 添加边（通信关系）
    G.add_edge(command_node, plan_node1)
    G.add_edge(plan_node1, plan_node2)

    G.add_edge(equipt_node1, command_node)
    G.add_edge(equipt_node2, equipt_node1)
    G.add_edge(equipt_node3, equipt_node1)
    G.add_edge(equipt_node4, equipt_node1)

    G.add_edge(aigc_node1, equipt_node2)
    G.add_edge(aigc_node2, equipt_node3)
    G.add_edge(aigc_node3, equipt_node4)

    return G