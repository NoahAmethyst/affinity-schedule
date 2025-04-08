# 静态调度框架
import logging
import math
import os
import random
from random import shuffle
from sched import scheduler
from typing import Any
import pandas as pd
import yaml
import numpy as np

from affinity.resource import BaseNode, BasePod, BaseObject, SingleSchedulerPlan
from util.logger import init_logger, logger

init_logger()


class Scheduler:
    # podIdx2name: [str]
    # nodeIdx2name: [str]
    # podName2idx: dict[str, int]
    # nodeName2idx: dict[str, int]

    affinity_weight = 1
    avg_usage_weight = 1
    var_usage_weight = 1

    def __init__(self, input_path: str, pod_affinity, node_affinity):
        self.pods: [BasePod] = []
        self.nodes: [BaseNode] = []
        self.pod_affinity = pod_affinity
        self.node_affinity = node_affinity
        self.read_input(input_path)
        self.scheduler_name = ""

    def schedule(self) -> [int]:
        raise NotImplementedError

    def affinity(self, plan: [int]) -> int:
        """ 计算节点直接亲和性 """
        res = 0
        affinity_pod = self.pod_affinity
        ### calc affinity between pods
        for pod1_idx in range(len(self.pods)):
            for pod2_idx in range(pod1_idx + 1, len(self.pods)):
                # affinity
                if plan[pod1_idx] == plan[pod2_idx]:
                    res += affinity_pod[pod1_idx, pod2_idx]
                if plan[pod1_idx] != plan[pod2_idx]:
                    res -= affinity_pod[pod1_idx, pod2_idx]
        return res

    def calc_cost(self, plan: [int]) -> float:
        min_usage, max_usage, usage = self.usage(plan)

        affinity_cost = self.affinity(plan)
        avg_usage_cost = np.average(max_usage)
        var_usage_cost = np.var(max_usage)
        logger.info(f'affinity: {affinity_cost}, avg: {avg_usage_cost}, var: {var_usage_cost}')
        return -(var_usage_cost * self.var_usage_weight +
                 affinity_cost * self.affinity_weight +
                 avg_usage_cost * self.avg_usage_weight)

    def usage(self, plan: np.ndarray) -> ([float], [float], [BaseObject]):
        """ 输出每个节点的 资源最低利用率 最高利用率 和 每个资源的利用率 """
        occupied = [None for _ in range(len(self.nodes))]
        for pod_idx, node_idx in enumerate(plan):
            if occupied[node_idx] is not None:
                tmp = occupied[node_idx]
            else:
                tmp = BaseObject("", 0, 0, 0, 0)
            tmp += self.pods[pod_idx]
            occupied[node_idx] = tmp

        min_usage = [0 for _ in range(len(occupied))]
        max_usage = [0 for _ in range(len(occupied))]
        usage = [BaseObject("", 0, 0, 0, 0) for _ in range(len(occupied))]
        for node_idx, u in enumerate(occupied):
            if u is None:
                min_usage[node_idx] = 0
                continue
            min_usage[node_idx] = self.nodes[node_idx].min_usage(u)
            max_usage[node_idx] = self.nodes[node_idx].max_usage(u)
            usage[node_idx] = self.nodes[node_idx].usage(u)
        return min_usage, max_usage, usage

    def used(self, plan: np.ndarray) -> [BaseObject]:
        """ 每个节点的资源使用量 """
        used = [BaseObject() for _ in range(len(self.nodes))]
        for pod_idx, node_idx in enumerate(plan):
            tmp = used[node_idx]
            tmp += self.pods[pod_idx]
            used[node_idx] = tmp
        return used

    def read_pod_yamls(self, pods_dir: str) -> dict[str, Any]:
        entries = os.listdir(pods_dir)
        pods = {}
        for entry in entries:
            with open(os.path.join(pods_dir, entry), 'r') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
                pods[data['metadata']['name']] = data
        self.pod_yaml = pods
        return pods

    def save_pod_yamls(self, dir: str):
        if self.plan is None:
            logger.info('call schedule first')
            return
        pod_yaml = self.pod_yaml
        plan = self.plan

        for pod_name, pod_idx in self.podName2idx.items():
            with open(os.path.join(dir, pod_name + '.yaml'), 'w') as file:
                one_yaml = pod_yaml[self.podIdx2name[pod_idx]]
                one_yaml['metadata']['nodeName'] = self.nodeIdx2name[plan[pod_idx]]
                yaml.dump(one_yaml, file, default_flow_style=False)

    def read_input(self, input_path: str):
        """ 读取输入数据 """
        ### read pods
        pods, pod2idx = read_pods_csv(f"{input_path}/pods.csv")
        self.pods = pods
        self.podName2idx = pod2idx

        ### read nodes
        nodes, node2idx = read_nodes_csv(f"{input_path}/nodes.csv")
        self.nodes = nodes
        self.nodeName2idx = node2idx

        ### read affinity
        if self.node_affinity is None or self.pod_affinity is None:
            init_logger()
            logger.warning(f'node and pod affinity are None.Load affinity from {input_path}')
            self.pod_affinity = np.load(f"{input_path}/pod_affinity.npy")
            self.node_affinity = np.load(f"{input_path}/node_affinity.npy")
        ### shuffle
        self.shuffle()

    def shuffle(self):
        """ 将输入的pod顺序打乱 """
        index = list(range(len(self.pods)))  # index[i]表示原位置
        random.seed(42)
        shuffle(index)
        pods = [self.pods[i] for i in index]
        ### 转换pod affinity
        pod_affinity = np.copy(self.pod_affinity)
        for i in range(pod_affinity.shape[0]):
            for j in range(pod_affinity.shape[1]):
                pod_affinity[i, j] = self.pod_affinity[index[i], index[j]]
        ### 转换node affinity
        node_affinity = np.copy(self.node_affinity)
        for i in range(node_affinity.shape[0]):
            node_affinity[i, :] = self.node_affinity[index[i], :]

        self.pods = pods
        self.pod_affinity = pod_affinity
        self.node_affinity = node_affinity

    def get_node_num(self):
        return len(self.nodes)

    def get_pod_num(self):
        return len(self.pods)

    def check(self, plan: np.ndarray) -> bool:
        """ 检查放置方案是否合法 """
        used = self.used(plan)
        for u, node in zip(used, self.nodes):
            tmp = node - u
            if not tmp.is_not_empty():
                return False
        return True

    def save_plan(self, save_path: str, plan: []):
        """ 保存调度结果 """
        data = [[self.pods[pod].name, self.nodes[node].name] for pod, node in enumerate(plan)]
        df = pd.DataFrame(data, columns=["name", "node"])
        df.to_csv(os.path.join(save_path, f'{self.scheduler_name}.csv'), index=False)

    @classmethod
    def check_and_output(cls, scheduler, save_path: str, plan: [int]):
        ### check
        result = scheduler.check(plan)
        if not result:
            logger.info('check failed')
            return

        ### 计算cost
        cost = scheduler.calc_cost(plan)
        logger.info(f'cost: {cost}')

        ### 计算利用率
        min_usage, max_usage, usage = scheduler.usage(plan)
        for i, v in enumerate(zip(min_usage, max_usage, usage)):
            min_u, max_u, u = v
            logger.info(f'node({i}): min_usage:{min_u}, max_usage:{max_u}, usage:[{u.to_string()}]')

        ### 保存结果
        scheduler.save_plan(save_path, plan)

    def check_and_gen(self, scheduler, plan: [int]) -> list[SingleSchedulerPlan] | None:
        """Validate a scheduling plan and generate execution details.

        Args:
            scheduler: Scheduler instance
            plan: List of node indices for each pod

        Returns:
            List of SingleSchedulerPlan objects if valid, None otherwise
        """
        # Early return if plan is invalid
        if not scheduler.check(plan):
            logger.info('Plan validation failed')
            return None

        # Log cost information
        cost = scheduler.calc_cost(plan)
        logger.info(f'Total cost: {cost}')

        # Log usage statistics with cleaner formatting
        min_usage, max_usage, usage = scheduler.usage(plan)
        for node_idx, (min_u, max_u, u) in enumerate(zip(min_usage, max_usage, usage)):
            logger.info(
                f'Node {node_idx}: Min={min_u:.2f}, Max={max_u:.2f}, '
                f'Usage={u.to_string()}'
            )

        # Generate plan objects using list comprehension
        return [
            SingleSchedulerPlan(self.pods[pod].name, self.nodes[node].name)
            for pod, node in enumerate(plan)
        ]


def read_pods_csv(path: str) -> ([], {}):
    ### read pods
    data = pd.read_csv(path)
    pods = []
    pod2idx = {}
    for idx, row in data.iterrows():
        pod = BasePod.from_dataframe(row)
        pods.append(pod)
        pod2idx[pod.name] = idx
    return pods, pod2idx


def read_nodes_csv(path: str) -> ([], {}):
    data = pd.read_csv(path)
    nodes = []
    node2idx = {}
    for idx, row in data.iterrows():
        node = BaseNode.from_dataframe(row)
        nodes.append(node)
        node2idx[node.name] = idx
    return nodes, node2idx


if __name__ == '__main__':
    scheduler = Scheduler("/offline-scheduler/data/input")
    scheduler.shuffle()
