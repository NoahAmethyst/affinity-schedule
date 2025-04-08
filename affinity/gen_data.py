import copy
import queue
import pandas as pd

from affinity.resource import BasePod, BaseNode, BasePlatform, Communication


def gen_base_pods() -> list[list[BasePod]]:
    return [
        [
            BasePod("", 2, 4, 1 * 1024, 2 * 1024, None),  # 感知
            BasePod("", 3, 5, 1 * 1024, 2 * 1024, None),  # 感知
            BasePod("", 3, 6, 1 * 1024, 2 * 1024, None),  # 感知
            BasePod("", 4, 7, 1 * 1024, 2 * 1024, None),  # 感知
        ],
        [
            BasePod("", 1, 4, 1 * 1024, 1 * 1024, None),  # 认知
            BasePod("", 2, 3, 2 * 1024, 1 * 1024, None),  # 认知
            BasePod("", 1, 2, 2 * 1024, 1 * 1024, None),  # 认知
            BasePod("", 2, 1, 3 * 1024, 1 * 1024, None),  # 认知
        ],
        [
            BasePod("", 2, 9, 0, 2 * 1024, None),  # 行动
            BasePod("", 3, 10, 0, 3 * 1024, None),  # 行动
            BasePod("", 3, 10, 0, 4 * 1024, None),  # 行动
            BasePod("", 2, 11, 0, 5 * 1024, None),  # 行动
        ],
        [
            BasePod("", 3, 10, 0, 2 * 1024, None),  # 决策
            BasePod("", 3, 10, 0, 3 * 1024, None),  # 决策
            BasePod("", 3, 10, 0, 3 * 1024, None),  # 决策
            BasePod("", 3, 10, 0, 4 * 1024, None),  # 决策
        ],
    ]


def gen_base_nodes() -> list[BaseNode]:
    return [
        BaseNode("", 64, 240, 0, 1.5 * 1024 * 1024, 10000),  #
        BaseNode("", 64, 240, 1024 * 24, 1.5 * 1024 * 1024, 10000),  # gpu
    ]


def gen_base_communication() -> tuple[list[int], list[list[list[int]]]]:
    # 通信频次、通信量、通信次数
    times = 1000 * 6
    return (
        [5, 1, times],  # 指挥关系通信
        [
            [[10, 50, times], [13, 30, times], [20, 40, times]],  # 智能体通信
            [[10, 60, times], [12, 40, times], [15, 30, times]],
            [[10, 70, times], [11, 50, times], [15, 40, times]],
            [[10, 80, times], [10, 60, times], [20, 30, times]],
        ]
    )


def gen_pods(num: int) -> ([], []):
    pod_types = 4
    command_child_nums = 3  # 指挥关系 n 叉树

    def next_index_array(idx: list[int]) -> list[int]:
        n = len(idx)
        carry = 1  # 初始时模拟加 1 的操作

        for i in range(n - 1, -1, -1):
            if carry == 0:
                break
            idx[i] += carry
            if idx[i] >= pod_types:  # 超过最大值时，产生进位并将当前位清零
                idx[i] -= pod_types
                carry = 1
            else:
                carry = 0  # 没有进位，停止处理
        return idx

    pods_type = gen_base_pods()
    command_communication, communication_type = gen_base_communication()
    communication = []

    pods_type_idx = [0, 0, 0, 0]  # 表示每一组的 pod 类型
    communication_type_idx = 0
    pods = []
    pod_idx = 0
    platform = []
    platform_idx = 0

    def gen_one_platform(parent: BasePlatform) -> BasePlatform:
        nonlocal communication_type_idx
        nonlocal communication_type
        nonlocal platform
        nonlocal platform_idx
        nonlocal pod_idx
        nonlocal pods_type_idx
        nonlocal pods_type
        nonlocal command_communication
        # 生成新的 platform
        p = BasePlatform(f'platform-{platform_idx + 1}')
        platform_idx += 1
        platform.append(p)
        if parent is not None:
            parent.add_child(p)
        p.add_parent(parent)
        # 生成第一个 pod
        pod = copy.copy(pods_type[0][pods_type_idx[0]])
        pod.name = f"pod-{pod_idx + 1}"
        pod_idx += 1
        pod.platform = p.name
        pod_list = [pod]
        # 生成parent最后一个pod到新的第一个pod的通信
        comm_list = []
        if parent is not None:
            comm = Communication(parent.pods[-1].name, pod.name, *command_communication)
            comm_list.append(comm)
        # 生成其余的pod
        for i in range(1, len(pods_type_idx)):
            # 创建 pod
            pod = copy.copy(pods_type[i][pods_type_idx[i]])
            pod.name = f"pod-{pod_idx + 1}"
            pod.platform = p.name
            pod_idx += 1
            p.add_pod(pod)
            pod_list.append(pod)
            # 获取通信
            comm = communication_type[communication_type_idx][i - 1]
            communication_type_idx = (communication_type_idx + 1) % len(communication_type)
            # 增加 connection
            comm_list.append(Communication(pod_list[-2].name, pod.name, *comm))
        # 将生成的pod添加到pod
        pods.extend(pod_list)
        communication.extend(comm_list)
        pods_type_idx = next_index_array(pods_type_idx)
        return p

    q = queue.Queue()
    root = gen_one_platform(None)
    q.put(root)
    while pod_idx < num:
        parent = q.get()
        for i in range(command_child_nums):
            child = gen_one_platform(parent)
            q.put(child)
            if pod_idx > num:
                break

    return pods, communication, platform


def gen_nodes(num: int, gpu_num: int) -> [BaseNode]:
    node_types = gen_base_nodes()
    nodes = [
        # BaseNode("pasak8s-14", 64, 1024 * 1024, 0, 1.5 * 1024 * 1024, 10000),  #
        # BaseNode("pasak8s-15", 64, 1024 * 1024, 0, 1.5 * 1024 * 1024, 10000),  #
        # BaseNode("pasak8s-16", 64, 1024 * 1024, 0, 1.5 * 1024 * 1024, 10000),  #
        # BaseNode("pasak8s-17", 64, 1024 * 1024, 0, 1.5 * 1024 * 1024, 10000),  #
        # BaseNode("pasak8s-18", 64, 1024 * 1024, 24 * 1024, 1.5 * 1024 * 1024, 10000),  # gpu
        # BaseNode("pasak8s-19", 64, 1024 * 1024, 24 * 1024, 1.5 * 1024 * 1024, 10000),  # gpu
        # BaseNode("pasak8s-20", 64, 1024 * 1024, 24 * 1024, 1.5 * 1024 * 1024, 10000),  # gpu
        # BaseNode("pasak8s-21", 64, 1024 * 1024, 24 * 1024, 1.5 * 1024 * 1024, 10000),  # gpu
    ]
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


if __name__ == '__main__':
    ### 生成测试数据
    save_path = ('/Users/amethyst/PycharmProjects/affinity-schedule/data/input')
    pods, comm, platform = gen_pods(1000)
    nodes = gen_nodes(5, 3)
    save_resource(pods, nodes, platform, save_path)
    save_communication(comm, save_path)
